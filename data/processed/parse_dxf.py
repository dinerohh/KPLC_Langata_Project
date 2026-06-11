"""
parse_dxf.py
============
Parses the Langata-Feeders.dxf AutoCAD R11/R12 network topology file into
three structured CSV outputs for use in modelling and dashboard visualisation.

Coordinate system: UTM Zone 37S (EPSG:32737) → converted to WGS84 (EPSG:4326)

Outputs (saved to data/processed/)
-------
    feeder_lines.csv       — polyline segments per feeder with WGS84 coords
    network_nodes.csv      — secondary substations and switches with coords
    feeder_topology.csv    — per-feeder aggregate statistics (model features)

Usage
-----
    python parse_dxf.py
    python parse_dxf.py --dxf /path/to/Langata-Feeders.dxf --out_dir /path/to/output
"""

import argparse
import warnings
from pathlib import Path
from math import sqrt

import pandas as pd
from ezdxf import recover
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ── coordinate transformer: UTM 37S → WGS84 ──────────────────────────────────
# EPSG:32737 = WGS 84 / UTM zone 37S
# EPSG:4326  = WGS 84 geographic (lat/lon)
_TRANSFORMER = Transformer.from_crs("EPSG:32737", "EPSG:4326", always_xy=True)

# ── feeder name inference ─────────────────────────────────────────────────────
# The DXF has no explicit feeder attribution per polyline.
# We match feeder names from the incidence data using spatial proximity
# to known feeder centroids derived from incidence outage locations.
# Fallback: assign "UNKNOWN" — topology is still usable for dashboard map.
FEEDER_APPROX_CENTROIDS_UTM = {
    "SOWETO EX LANGATA":         (250_500, 9_851_500),
    "NGEI EX LANGATA":           (251_800, 9_852_800),
    "MAGADI  EX LANGATA":        (253_500, 9_851_000),
    "HARDY EX LANGATA":          (254_500, 9_853_500),
    "KUWINDA EX LANGATA":        (252_200, 9_853_800),
    "KAREN HOSPITAL EX LANGATA": (255_000, 9_853_200),
    "NDALATI EX LANGATA":        (252_800, 9_854_500),
    "OTIENDE EX LANGATA":        (251_400, 9_851_900),
}


def _utm_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Convert UTM Zone 37S coordinates to (lon, lat) WGS84."""
    lon, lat = _TRANSFORMER.transform(x, y)
    return round(lon, 7), round(lat, 7)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _nearest_feeder(x: float, y: float) -> str:
    """Assign a feeder name based on proximity to known feeder centroids."""
    return min(
        FEEDER_APPROX_CENTROIDS_UTM,
        key=lambda f: _distance(x, y, *FEEDER_APPROX_CENTROIDS_UTM[f])
    )


def _segment_length_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in metres (valid in UTM)."""
    return round(_distance(x1, y1, x2, y2), 2)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_feeder_lines(msp) -> pd.DataFrame:
    """
    Extract all POLYLINE entities from the MV_LINE_SECTION layer.
    Each polyline is decomposed into sequential (start, end) segments.
    Each segment is assigned to the nearest feeder by centroid proximity.

    Returns one row per line segment.
    """
    rows = []
    seg_id = 0

    for entity in msp:
        if entity.dxftype() != "POLYLINE":
            continue
        if entity.dxf.layer != "MV_LINE_SECTION":
            continue

        try:
            verts = [(v.dxf.location.x, v.dxf.location.y)
                     for v in entity.vertices]
        except Exception:
            continue

        if len(verts) < 2:
            continue

        # centroid of this polyline for feeder assignment
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        feeder = _nearest_feeder(cx, cy)

        for i in range(len(verts) - 1):
            x1, y1 = verts[i]
            x2, y2 = verts[i + 1]
            lon1, lat1 = _utm_to_wgs84(x1, y1)
            lon2, lat2 = _utm_to_wgs84(x2, y2)
            length_m = _segment_length_m(x1, y1, x2, y2)

            rows.append({
                "segment_id":   f"SEG_{seg_id:05d}",
                "feeder_name":  feeder,
                "lon_start":    lon1,
                "lat_start":    lat1,
                "lon_end":      lon2,
                "lat_end":      lat2,
                "length_m":     length_m,
                "polyline_idx": seg_id // max(len(verts) - 1, 1),
            })
            seg_id += 1

    return pd.DataFrame(rows)


def extract_network_nodes(msp) -> pd.DataFrame:
    """
    Extract INSERT entities representing network nodes:
      - SECONDARY_SUBSTATION  — distribution transformers
      - SECONDARY_SWITCH_ISOLATOR — line switches and fuses
      - PRIMARY_SUBSTATION    — Langata 66kV/11kV anchor point

    Returns one row per node with coordinates and attributes.
    """
    rows = []
    node_id = 0

    layer_type_map = {
        "SECONDARY_SUBSTATION":      "secondary_substation",
        "SECONDARY_SWITCH_ISOLATOR": "switch_isolator",
        "PRIMARY_SUBSTATION":        "primary_substation",
        "BDIV10_INFO_SECONDARY_SWITC": "switch_isolator",
    }

    for entity in msp:
        if entity.dxftype() != "INSERT":
            continue
        layer = entity.dxf.layer
        if layer not in layer_type_map:
            continue

        x = entity.dxf.insert.x
        y = entity.dxf.insert.y
        lon, lat = _utm_to_wgs84(x, y)
        node_type = layer_type_map[layer]
        feeder = _nearest_feeder(x, y)

        # extract attribute text (switch number, substation name)
        attrib_text = ""
        try:
            attribs = list(entity.attribs)
            if attribs:
                attrib_text = "; ".join(
                    f"{a.dxf.tag}={a.dxf.text}"
                    for a in attribs
                    if a.dxf.text.strip()
                )
        except Exception:
            pass

        rows.append({
            "node_id":      f"NODE_{node_id:04d}",
            "node_type":    node_type,
            "feeder_name":  feeder if node_type != "primary_substation" else "LANGATA SUBSTATION",
            "lon":          lon,
            "lat":          lat,
            "utm_x":        round(x, 2),
            "utm_y":        round(y, 2),
            "attributes":   attrib_text,
            "block_name":   entity.dxf.name,
        })
        node_id += 1

    return pd.DataFrame(rows)


def compute_feeder_topology(
    lines_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-feeder topology statistics from extracted lines and nodes.
    These become additional static model features in the feeder-day frame.

    Features produced
    -----------------
    total_length_km     — total route length of all feeder segments
    n_segments          — number of line segments (proxy for network complexity)
    n_secondary_subs    — number of distribution transformers on feeder
    n_switches          — number of sectionalising switches / fuses
    bbox_area_km2       — bounding box area of feeder service territory
    avg_segment_m       — average segment length (short = dense urban network)
    """
    rows = []

    all_feeders = list(FEEDER_APPROX_CENTROIDS_UTM.keys())

    for feeder in all_feeders:
        fl = lines_df[lines_df["feeder_name"] == feeder]
        fn = nodes_df[nodes_df["feeder_name"] == feeder]

        total_length_m = fl["length_m"].sum()
        n_segments     = len(fl)
        n_subs         = (fn["node_type"] == "secondary_substation").sum()
        n_switches     = (fn["node_type"] == "switch_isolator").sum()
        avg_seg_m      = fl["length_m"].mean() if n_segments > 0 else 0

        # bounding box area using lat/lon (approximate km² at Nairobi latitude)
        if len(fl) > 0:
            lat_all = list(fl["lat_start"]) + list(fl["lat_end"])
            lon_all = list(fl["lon_start"]) + list(fl["lon_end"])
            # 1° lat ≈ 111 km; 1° lon ≈ 111 × cos(-1.32°) ≈ 110.97 km
            bbox_lat_km = (max(lat_all) - min(lat_all)) * 111.0
            bbox_lon_km = (max(lon_all) - min(lon_all)) * 110.97
            bbox_area   = round(bbox_lat_km * bbox_lon_km, 4)
        else:
            bbox_area = 0.0

        rows.append({
            "feeder_name":      feeder,
            "total_length_km":  round(total_length_m / 1000, 4),
            "n_segments":       n_segments,
            "n_secondary_subs": int(n_subs),
            "n_switches":       int(n_switches),
            "avg_segment_m":    round(avg_seg_m, 2),
            "bbox_area_km2":    bbox_area,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_dxf(dxf_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading  : {dxf_path}")
    print(f"Output   : {out_dir}")
    print()

    # recover mode handles the duplicate entity handles in this R11/R12 file
    doc, _ = recover.readfile(str(dxf_path))
    msp = doc.modelspace()

    # ── feeder lines ──────────────────────────────────────────────────────────
    print("Extracting feeder line segments ...")
    lines_df = extract_feeder_lines(msp)
    lines_path = out_dir / "feeder_lines.csv"
    lines_df.to_csv(lines_path, index=False)
    print(f"  {len(lines_df):,} segments across {lines_df['feeder_name'].nunique()} feeders")

    # ── network nodes ─────────────────────────────────────────────────────────
    print("Extracting network nodes ...")
    nodes_df = extract_network_nodes(msp)
    nodes_path = out_dir / "network_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    print(f"  {len(nodes_df):,} nodes  "
          f"({(nodes_df['node_type']=='secondary_substation').sum()} substations, "
          f"{(nodes_df['node_type']=='switch_isolator').sum()} switches)")

    # ── feeder topology summary ───────────────────────────────────────────────
    print("Computing per-feeder topology statistics ...")
    topo_df = compute_feeder_topology(lines_df, nodes_df)
    topo_path = out_dir / "feeder_topology.csv"
    topo_df.to_csv(topo_path, index=False)

    print()
    print("=== FEEDER TOPOLOGY SUMMARY ===")
    print(topo_df.to_string(index=False))

    print()
    print("Saved:")
    print(f"  {lines_path}    ({len(lines_df):,} rows)")
    print(f"  {nodes_path}    ({len(nodes_df):,} rows)")
    print(f"  {topo_path}     ({len(topo_df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Langata feeder DXF topology")
    parser.add_argument(
        "--dxf",
        default="data/raw/Langata-Feeders.dxf",
        help="Path to DXF file",
    )
    parser.add_argument(
        "--out_dir",
        default="data/processed",
        help="Output directory for CSV files",
    )
    args = parser.parse_args()
    parse_dxf(Path(args.dxf), Path(args.out_dir))


if __name__ == "__main__":
    main()
