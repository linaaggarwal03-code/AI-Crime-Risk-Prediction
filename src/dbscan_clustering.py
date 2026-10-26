"""
Geographic Crime Hotspot Detection using DBSCAN.
Implements haversine spherical distance clustering on latitude/longitude coordinates,
systematic parameter sensitivity experiments, and analytical cluster summarization.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Volumetric mean Earth radius in kilometers (IUGG standard / WGS-84 reference)
EARTH_RADIUS_KM = 6371.0088


class DBSCANHotspotDetector:
    """
    Geographic Hotspot Detector utilizing DBSCAN with the Haversine spherical distance metric.
    Prevents Cartesian projection distortion on unprojected spherical angular coordinates.
    """

    def __init__(
        self,
        eps_meters: float = 500.0,
        min_samples: int = 50,
        earth_radius_km: float = EARTH_RADIUS_KM,
    ):
        self.eps_meters = eps_meters
        self.eps_km = eps_meters / 1000.0
        self.min_samples = min_samples
        self.earth_radius_km = earth_radius_km
        # eps in radians for haversine metric: eps_rad = distance_km / Earth_radius_km
        self.eps_radians = self.eps_km / self.earth_radius_km

        self.model: Optional[DBSCAN] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_summary: Optional[pd.DataFrame] = None
        self.experiment_results: Optional[pd.DataFrame] = None

    def fit_predict(
        self,
        df: pd.DataFrame,
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> np.ndarray:
        """
        Executes DBSCAN on geographic coordinates using haversine metric.
        Requires coordinates in radians, in [latitude, longitude] order.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty; cannot perform clustering.")

        if lat_col not in df.columns or lon_col not in df.columns:
            raise KeyError(f"Coordinate columns '{lat_col}' and/or '{lon_col}' not found in DataFrame.")

        # Extract coordinates and convert degrees to spherical radians
        coords = df[[lat_col, lon_col]].dropna()
        if len(coords) < self.min_samples:
            raise ValueError(
                f"Valid records ({len(coords)}) are fewer than min_samples ({self.min_samples})."
            )

        coords_rad = np.radians(coords.to_numpy())

        # Initialize scikit-learn DBSCAN with metric='haversine' and algorithm='ball_tree'
        self.model = DBSCAN(
            eps=self.eps_radians,
            min_samples=self.min_samples,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=-1,
        )

        labels = self.model.fit_predict(coords_rad)
        self.cluster_labels = labels
        return labels

    def compute_cluster_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        id_col: Optional[str] = "ID",
        cat_col: Optional[str] = "Primary Type",
        area_col: Optional[str] = "Community Area",
    ) -> pd.DataFrame:
        """
        Computes geographic centroids, bounding envelopes, and dominant crime categories per cluster.
        Excludes noise (-1) from positive cluster identification.
        """
        labeled_df = df.copy()
        labeled_df["cluster_id"] = labels

        clusters = sorted([c for c in np.unique(labels) if c != -1])
        summary_rows = []

        for cid in clusters:
            c_data = labeled_df[labeled_df["cluster_id"] == cid]
            count = len(c_data)

            # Representative geographic centroid (mean coordinates)
            mean_lat = float(c_data[lat_col].mean())
            mean_lon = float(c_data[lon_col].mean())

            # Spatial extent
            min_lat = float(c_data[lat_col].min())
            max_lat = float(c_data[lat_col].max())
            min_lon = float(c_data[lon_col].min())
            max_lon = float(c_data[lon_col].max())

            # Top crime category
            top_category = (
                str(c_data[cat_col].mode().iloc[0])
                if cat_col and cat_col in c_data.columns and not c_data[cat_col].empty
                else "Unknown"
            )

            # Top community area
            top_area = (
                str(int(c_data[area_col].mode().iloc[0]))
                if area_col and area_col in c_data.columns and not c_data[area_col].dropna().empty
                else "N/A"
            )

            summary_rows.append({
                "cluster_id": int(cid),
                "incident_count": int(count),
                "percentage_of_total": round(count / len(df) * 100, 2),
                "centroid_latitude": round(mean_lat, 6),
                "centroid_longitude": round(mean_lon, 6),
                "min_latitude": round(min_lat, 6),
                "max_latitude": round(max_lat, 6),
                "min_longitude": round(min_lon, 6),
                "max_longitude": round(max_lon, 6),
                "primary_crime_type": top_category,
                "dominant_community_area": top_area,
            })

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by="incident_count", ascending=False).reset_index(drop=True)
        self.cluster_summary = summary_df
        return summary_df

    def run_parameter_experiments(
        self,
        df: pd.DataFrame,
        eps_values_meters: List[float],
        min_samples_values: List[int],
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
    ) -> pd.DataFrame:
        """
        Evaluates sensitivity of DBSCAN across systematic combinations of eps and min_samples.
        """
        results = []
        total_points = len(df)
        coords_rad = np.radians(df[[lat_col, lon_col]].to_numpy())

        print(f"[DBSCAN] Starting parameter experiments ({len(eps_values_meters)} eps x {len(min_samples_values)} min_samples)...")

        for eps_m in eps_values_meters:
            eps_km = eps_m / 1000.0
            eps_rad = eps_km / self.earth_radius_km

            for ms in min_samples_values:
                db = DBSCAN(eps=eps_rad, min_samples=ms, metric="haversine", algorithm="ball_tree", n_jobs=-1)
                labels = db.fit_predict(coords_rad)

                n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
                n_noise = int(np.sum(labels == -1))
                noise_pct = round(n_noise / total_points * 100, 2)
                clustered_points = total_points - n_noise
                clustered_pct = round(clustered_points / total_points * 100, 2)

                cluster_sizes = [int(np.sum(labels == c)) for c in set(labels) if c != -1]
                max_size = int(max(cluster_sizes)) if cluster_sizes else 0
                min_size = int(min(cluster_sizes)) if cluster_sizes else 0
                mean_size = round(float(np.mean(cluster_sizes)), 1) if cluster_sizes else 0.0

                results.append({
                    "eps_meters": float(eps_m),
                    "eps_km": round(eps_km, 3),
                    "eps_radians": round(eps_rad, 7),
                    "min_samples": int(ms),
                    "num_clusters": n_clusters,
                    "noise_points": n_noise,
                    "noise_percentage": noise_pct,
                    "clustered_points": clustered_points,
                    "clustered_percentage": clustered_pct,
                    "max_cluster_size": max_size,
                    "min_cluster_size": min_size,
                    "mean_cluster_size": mean_size,
                })

        exp_df = pd.DataFrame(results)
        self.experiment_results = exp_df
        return exp_df


def generate_dbscan_visualizations(
    df: pd.DataFrame,
    labels: np.ndarray,
    summary_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    output_dir: str = "outputs",
) -> Dict[str, str]:
    """
    Generates publication-quality static charts and interactive maps for DBSCAN results.
    """
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_paths = {}

    labeled_df = df.copy()
    labeled_df["cluster_id"] = labels
    n_clusters = len(summary_df)
    n_noise = int(np.sum(labels == -1))

    # Figure 1: DBSCAN Hotspot Clusters Map
    fig, ax = plt.subplots(figsize=(9, 11))
    
    # Plot noise points first in subtle light gray
    noise_pts = labeled_df[labeled_df["cluster_id"] == -1]
    ax.scatter(
        noise_pts[lon_col],
        noise_pts[lat_col],
        c="#cbd5e1",
        alpha=0.35,
        s=4,
        label=f"Noise / Dispersed Incidents ({n_noise:,}; {n_noise/len(df)*100:.1f}%)",
        edgecolors="none",
    )

    # Plot top non-noise clusters with distinct colors
    top_cluster_ids = summary_df["cluster_id"].head(8).tolist()
    cmap = plt.get_cmap("tab10", len(top_cluster_ids))

    for idx, cid in enumerate(top_cluster_ids):
        c_pts = labeled_df[labeled_df["cluster_id"] == cid]
        c_info = summary_df[summary_df["cluster_id"] == cid].iloc[0]
        ax.scatter(
            c_pts[lon_col],
            c_pts[lat_col],
            color=cmap(idx),
            alpha=0.75,
            s=12,
            label=f"Cluster #{cid} ({c_info['incident_count']:,} pts; Area {c_info['dominant_community_area']})",
            edgecolors="none",
        )
        # Centroid marker
        ax.scatter(
            [c_info["centroid_longitude"]],
            [c_info["centroid_latitude"]],
            marker="*",
            s=130,
            color="black",
            edgecolors="yellow",
            linewidth=1,
            zorder=6,
        )

    ax.set_title(
        f"Geographic Hotspot Detection via DBSCAN\n"
        f"({n_clusters} Dense Clusters Identified, Haversine Spherical Metric)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude (Degrees)", fontsize=11)
    ax.set_ylabel("Latitude (Degrees)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    plt.tight_layout()
    cluster_map_path = os.path.join(figures_dir, "dbscan_hotspot_clusters.png")
    plt.savefig(cluster_map_path, dpi=200)
    plt.close(fig)
    fig_paths["cluster_map"] = cluster_map_path

    # Figure 2: Cluster Size Distribution (Top 15 Clusters)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_summary = summary_df.head(15)
    cluster_labels = [f"Cluster {int(cid)}" for cid in top_summary["cluster_id"]]

    sns.barplot(
        x=cluster_labels,
        y=top_summary["incident_count"].values,
        hue=cluster_labels,
        palette="flare",
        ax=ax,
        legend=False,
    )
    ax.set_title("Incident Volume by Identified DBSCAN Hotspot Cluster (Top 15)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Hotspot Cluster ID", fontsize=11)
    ax.set_ylabel("Number of Assigned Incidents", fontsize=11)
    plt.xticks(rotation=30)

    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h + 5),
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    size_chart_path = os.path.join(figures_dir, "dbscan_cluster_sizes.png")
    plt.savefig(size_chart_path, dpi=200)
    plt.close(fig)
    fig_paths["cluster_sizes"] = size_chart_path

    # Figure 3: Parameter Sensitivity Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    sns.lineplot(
        data=exp_df,
        x="eps_meters",
        y="num_clusters",
        hue="min_samples",
        marker="o",
        palette="viridis",
        ax=ax1,
    )
    ax1.set_title("Sensitivity: Number of Clusters vs. Epsilon", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Neighborhood Radius Epsilon (meters)")
    ax1.set_ylabel("Number of Detected Clusters")
    ax1.grid(True, linestyle="--", alpha=0.5)

    sns.lineplot(
        data=exp_df,
        x="eps_meters",
        y="noise_percentage",
        hue="min_samples",
        marker="s",
        palette="rocket",
        ax=ax2,
    )
    ax2.set_title("Sensitivity: Noise Percentage vs. Epsilon", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Neighborhood Radius Epsilon (meters)")
    ax2.set_ylabel("Unassigned Noise Percentage (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("DBSCAN Hyperparameter Optimization and Sensitivity Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    tuning_path = os.path.join(figures_dir, "dbscan_parameter_tuning.png")
    plt.savefig(tuning_path, dpi=200)
    plt.close(fig)
    fig_paths["parameter_tuning"] = tuning_path

    # Figure 4: Interactive Hotspot HTML Map
    map_path = generate_interactive_hotspot_html_map(df, labels, summary_df, lat_col, lon_col, output_dir)
    fig_paths["interactive_hotspots"] = map_path

    return fig_paths


def generate_interactive_hotspot_html_map(
    df: pd.DataFrame,
    labels: np.ndarray,
    summary_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    output_dir: str = "outputs",
    sample_size: int = 2500,
) -> str:
    """
    Creates an interactive HTML Leaflet map displaying DBSCAN hotspot clusters,
    noise points, and cluster centroid cards.
    """
    labeled_df = df.copy()
    labeled_df["cluster_id"] = labels

    sample_pts = labeled_df.sample(n=min(len(labeled_df), sample_size), random_state=42)
    mean_lat = float(df[lat_col].mean())
    mean_lon = float(df[lon_col].mean())

    palette = [
        "#e11d48", "#2563eb", "#16a34a", "#d97706",
        "#9333ea", "#0891b2", "#ea580c", "#4f46e5",
    ]

    cluster_color_map = {}
    for idx, cid in enumerate(summary_df["cluster_id"].head(len(palette))):
        cluster_color_map[int(cid)] = palette[idx]

    points_data = []
    for _, r in sample_pts.iterrows():
        cid = int(r["cluster_id"])
        color = cluster_color_map.get(cid, "#94a3b8" if cid == -1 else "#64748b")
        points_data.append({
            "lat": float(r[lat_col]),
            "lon": float(r[lon_col]),
            "cid": cid,
            "color": color,
            "type": str(r.get("Primary Type", "Incident")),
        })

    centroids_data = []
    for _, c in summary_df.head(10).iterrows():
        cid = int(c["cluster_id"])
        centroids_data.append({
            "cid": cid,
            "lat": float(c["centroid_latitude"]),
            "lon": float(c["centroid_longitude"]),
            "count": int(c["incident_count"]),
            "pct": float(c["percentage_of_total"]),
            "type": str(c["primary_crime_type"]),
            "area": str(c["dominant_community_area"]),
            "color": cluster_color_map.get(cid, "#e11d48"),
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Chicago DBSCAN Crime Hotspot Intelligence Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        .hud-card {{
            position: absolute; top: 12px; left: 12px; z-index: 1000;
            background: rgba(15, 23, 42, 0.92); color: #f8fafc; padding: 14px 18px;
            border-radius: 10px; box-shadow: 0 4px 14px rgba(0,0,0,0.3);
            max-width: 320px; font-size: 13px; backdrop-filter: blur(4px);
        }}
        .hud-card h3 {{ margin: 0 0 6px 0; font-size: 15px; color: #38bdf8; }}
        .legend-row {{ display: flex; align-items: center; margin: 4px 0; font-size: 12px; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
    </style>
</head>
<body>
    <div class="hud-card">
        <h3>DBSCAN Hotspot Detection</h3>
        <div><strong>Algorithm:</strong> DBSCAN (Haversine Metric)</div>
        <div><strong>Clusters Detected:</strong> {len(summary_df):,}</div>
        <div><strong>Total Points Analyzed:</strong> {len(df):,}</div>
        <div style="margin-top: 8px; border-top: 1px solid #334155; padding-top: 6px;">
            <div class="legend-row"><span class="legend-dot" style="background: #94a3b8;"></span> Noise / Dispersed Incidents</div>
            <div class="legend-row"><span class="legend-dot" style="background: #e11d48;"></span> Top Hotspot Cluster (Centroid ★)</div>
            <div class="legend-row"><span class="legend-dot" style="background: #2563eb;"></span> Secondary Hotspots</div>
        </div>
        <div style="font-size: 11px; color: #94a3b8; margin-top: 8px;">
            * Clusters indicate historical incident density, not guaranteed future crime.
        </div>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{mean_lat}, {mean_lon}], 11);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);

        var points = {json.dumps(points_data)};
        points.forEach(function(p) {{
            var r = (p.cid === -1) ? 2 : 5;
            var op = (p.cid === -1) ? 0.35 : 0.8;
            L.circleMarker([p.lat, p.lon], {{
                radius: r,
                fillColor: p.color,
                color: p.color,
                weight: 1,
                opacity: op,
                fillOpacity: op
            }}).addTo(map).bindPopup('<strong>' + (p.cid === -1 ? 'Noise Incident' : 'Cluster #' + p.cid) + '</strong><br>' + p.type);
        }});

        var centroids = {json.dumps(centroids_data)};
        centroids.forEach(function(c) {{
            var starIcon = L.divIcon({{
                className: 'custom-star',
                html: '<div style=\"font-size: 20px; color: #fbbf24; text-shadow: 0 0 3px #000;\">★</div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            }});
            L.marker([c.lat, c.lon], {{ icon: starIcon }}).addTo(map)
                .bindPopup('<strong>Hotspot Cluster #' + c.cid + ' Centroid</strong><br>' +
                           'Incidents: ' + c.count + ' (' + c.pct + '%)<br>' +
                           'Dominant Area: ' + c.area + '<br>' +
                           'Top Crime: ' + c.type);
        }});
    </script>
</body>
</html>"""

    map_path = os.path.join(output_dir, "interactive_hotspot_map.html")
    with open(map_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[DBSCAN] Generated Interactive Hotspot Leaflet Map: {map_path}")
    return map_path


def execute_full_dbscan_pipeline(
    filepath: Optional[str] = None,
    output_dir: str = "outputs",
    selected_eps_meters: float = 500.0,
    selected_min_samples: int = 50,
) -> Dict[str, Any]:
    """
    Executes end-to-end DBSCAN analysis:
    1. Loads spatially clean records.
    2. Runs systematic parameter sensitivity experiments.
    3. Runs clustering under the primary selected configuration.
    4. Generates cluster summary table and analytical artifacts.
    5. Saves output CSVs, JSON summary, and visualization figures.
    """
    from src.data_loader import load_and_validate_crime_data

    os.makedirs(output_dir, exist_ok=True)
    clean_df, col_map, audit = load_and_validate_crime_data(filepath)

    lat_col = col_map["latitude"]
    lon_col = col_map["longitude"]
    id_col = col_map.get("id", "ID")
    cat_col = col_map.get("category", "Primary Type")
    area_col = col_map.get("area", "Community Area")

    detector = DBSCANHotspotDetector(
        eps_meters=selected_eps_meters,
        min_samples=selected_min_samples,
    )

    # 1. Run parameter sensitivity experiments
    eps_candidates = [200.0, 350.0, 500.0, 750.0, 1000.0]
    min_samples_candidates = [15, 25, 50, 75, 100]
    exp_df = detector.run_parameter_experiments(
        clean_df,
        eps_values_meters=eps_candidates,
        min_samples_values=min_samples_candidates,
        lat_col=lat_col,
        lon_col=lon_col,
    )
    exp_csv_path = os.path.join(output_dir, "parameter_experiments.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    print(f"[DBSCAN] Saved parameter experiments table: {exp_csv_path}")

    # 2. Run primary clustering model
    print(f"[DBSCAN] Executing primary model (eps={selected_eps_meters}m, min_samples={selected_min_samples})...")
    labels = detector.fit_predict(clean_df, lat_col=lat_col, lon_col=lon_col)

    # 3. Compute cluster summary
    summary_df = detector.compute_cluster_summary(
        clean_df,
        labels,
        lat_col=lat_col,
        lon_col=lon_col,
        id_col=id_col,
        cat_col=cat_col,
        area_col=area_col,
    )
    summary_csv_path = os.path.join(output_dir, "cluster_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[DBSCAN] Saved cluster summary table: {summary_csv_path} ({len(summary_df)} clusters)")

    # 4. Save record-level hotspot assignments
    results_df = clean_df.copy()
    results_df["cluster_id"] = labels
    results_csv_path = os.path.join(output_dir, "hotspot_results.csv")
    # Export essential columns for lightweight output
    export_cols = [c for c in [id_col, "Date", cat_col, area_col, lat_col, lon_col, "cluster_id"] if c in results_df.columns]
    results_df[export_cols].to_csv(results_csv_path, index=False)
    print(f"[DBSCAN] Saved record-level hotspot assignments: {results_csv_path}")

    # 5. Generate visualizations
    fig_paths = generate_dbscan_visualizations(
        clean_df,
        labels,
        summary_df,
        exp_df,
        lat_col=lat_col,
        lon_col=lon_col,
        output_dir=output_dir,
    )

    # 6. Build and save comprehensive analysis summary JSON
    total_records = len(clean_df)
    noise_count = int(np.sum(labels == -1))
    clustered_count = total_records - noise_count

    analysis_summary = {
        "project": "AI-Powered Urban Crime Risk & Hotspot Prediction System",
        "author": "Aditi Verma (Member 2)",
        "algorithm": "DBSCAN (Density-Based Spatial Clustering of Applications with Noise)",
        "distance_metric": "Haversine (spherical angular radians)",
        "earth_radius_km": EARTH_RADIUS_KM,
        "selected_hyperparameters": {
            "eps_meters": selected_eps_meters,
            "eps_km": selected_eps_meters / 1000.0,
            "eps_radians": round((selected_eps_meters / 1000.0) / EARTH_RADIUS_KM, 7),
            "min_samples": selected_min_samples,
        },
        "results_summary": {
            "total_records_analyzed": total_records,
            "num_clusters_identified": int(len(summary_df)),
            "clustered_incidents": clustered_count,
            "clustered_percentage": round(clustered_count / total_records * 100, 2),
            "noise_incidents": noise_count,
            "noise_percentage": round(noise_count / total_records * 100, 2),
            "largest_cluster_size": int(summary_df["incident_count"].max()) if not summary_df.empty else 0,
            "smallest_cluster_size": int(summary_df["incident_count"].min()) if not summary_df.empty else 0,
            "median_cluster_size": float(summary_df["incident_count"].median()) if not summary_df.empty else 0.0,
        },
        "top_clusters": summary_df.head(5).to_dict(orient="records") if not summary_df.empty else [],
        "artifacts_generated": {
            "hotspot_results_csv": results_csv_path,
            "cluster_summary_csv": summary_csv_path,
            "parameter_experiments_csv": exp_csv_path,
            "cluster_map_figure": fig_paths.get("cluster_map"),
            "cluster_sizes_figure": fig_paths.get("cluster_sizes"),
            "parameter_tuning_figure": fig_paths.get("parameter_tuning"),
            "interactive_hotspots_map": fig_paths.get("interactive_hotspots"),
        },
    }

    summary_json_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"[DBSCAN] Saved comprehensive analysis summary JSON: {summary_json_path}")

    return analysis_summary


if __name__ == "__main__":
    summary = execute_full_dbscan_pipeline()
    print("[DBSCAN] Pipeline finished successfully.")
