"""
Spatial Exploratory Data Analysis (EDA) Module.
Generates descriptive spatial statistics, geographic bounding box analysis,
and analytical visualizations (Static PNGs and Interactive HTML Maps).
"""

import os
import sys
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SpatialEDA:
    """
    Performs rigorous spatial exploratory analysis on validated crime records.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        column_mapping: Dict[str, str],
        audit_info: Dict[str, Any],
        output_dir: str = "outputs",
    ):
        self.df = df.copy()
        self.col_map = column_mapping
        self.audit = audit_info
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.lat_col = self.col_map["latitude"]
        self.lon_col = self.col_map["longitude"]
        self.cat_col = self.col_map.get("category")
        self.area_col = self.col_map.get("area")
        self.date_col = self.col_map.get("date")

        # Visual styling
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Computes spatial descriptive statistics, extents, and categorical metrics.
        """
        stats: Dict[str, Any] = {
            "spatial_audit": self.audit,
            "total_analyzed_incidents": len(self.df),
            "geographic_extent": {
                "min_latitude": float(self.df[self.lat_col].min()),
                "max_latitude": float(self.df[self.lat_col].max()),
                "latitude_span_deg": float(self.df[self.lat_col].max() - self.df[self.lat_col].min()),
                "min_longitude": float(self.df[self.lon_col].min()),
                "max_longitude": float(self.df[self.lon_col].max()),
                "longitude_span_deg": float(self.df[self.lon_col].max() - self.df[self.lon_col].min()),
                "mean_latitude": float(self.df[self.lat_col].mean()),
                "mean_longitude": float(self.df[self.lon_col].mean()),
            },
        }

        # Categories
        if self.cat_col and self.cat_col in self.df.columns:
            cat_counts = self.df[self.cat_col].value_counts()
            stats["categories"] = {
                "distinct_count": int(self.df[self.cat_col].nunique()),
                "top_10": {str(k): int(v) for k, v in cat_counts.head(10).items()},
            }

        # Community Areas
        if self.area_col and self.area_col in self.df.columns:
            area_counts = self.df[self.area_col].value_counts()
            stats["community_areas"] = {
                "distinct_count": int(self.df[self.area_col].nunique()),
                "top_10": {str(k): int(v) for k, v in area_counts.head(10).items()},
            }

        # Date range
        if self.date_col and self.date_col in self.df.columns:
            try:
                parsed_dates = pd.to_datetime(self.df[self.date_col], errors="coerce").dropna()
                if not parsed_dates.empty:
                    stats["temporal_extent"] = {
                        "start_date": str(parsed_dates.min()),
                        "end_date": str(parsed_dates.max()),
                        "days_span": int((parsed_dates.max() - parsed_dates.min()).days),
                    }
            except Exception as e:
                stats["temporal_extent"] = {"error": str(e)}

        return stats

    def generate_all_visualizations(self) -> Dict[str, str]:
        """
        Generates visualizations A, B, C, D, E and saves them to outputs/figures.
        """
        fig_paths: Dict[str, str] = {}

        # Vis A: Historical Crime Locations
        fig_paths["vis_a"] = self.plot_historical_locations()

        # Vis B: Crime Density Heatmap (2D KDE / Hexbin)
        fig_paths["vis_b"] = self.plot_density_heatmap()

        # Vis C: Crime Category Distribution
        if self.cat_col and self.cat_col in self.df.columns:
            fig_paths["vis_c"] = self.plot_category_distribution()

        # Vis D: Area-level Distribution
        if self.area_col and self.area_col in self.df.columns:
            fig_paths["vis_d"] = self.plot_area_distribution()

        # Vis E: Geographic Coverage and Extent
        fig_paths["vis_e"] = self.plot_geographic_coverage()

        # Interactive HTML map
        fig_paths["interactive_map"] = self.generate_interactive_html_map()

        return fig_paths

    def plot_historical_locations(self) -> str:
        """
        Visualization A: Scatter distribution of historical crime incidents across Chicago.
        """
        fig, ax = plt.subplots(figsize=(8, 10))
        # Subsample if large for crisp vector rendering
        plot_df = self.df.sample(n=min(len(self.df), 15000), random_state=42)

        scatter = ax.scatter(
            plot_df[self.lon_col],
            plot_df[self.lat_col],
            c="#1f77b4",
            alpha=0.25,
            s=4,
            edgecolors="none",
        )
        ax.set_title("Visualization A: Geographic Distribution of Historical Incidents\n(City of Chicago)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (Degrees)", fontsize=11)
        ax.set_ylabel("Latitude (Degrees)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Highlight Chicago Loop reference coordinates
        ax.scatter([-87.6298], [41.8781], color="red", marker="x", s=80, label="Chicago Downtown / Loop (41.88°N, -87.63°W)", zorder=5)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "vis_a_historical_locations.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[SpatialEDA] Saved Visualization A: {path}")
        return path

    def plot_density_heatmap(self) -> str:
        """
        Visualization B: 2D Hexagonal Binning density representation of crime concentration.
        """
        fig, ax = plt.subplots(figsize=(9, 10))
        
        hb = ax.hexbin(
            self.df[self.lon_col],
            self.df[self.lat_col],
            gridsize=55,
            cmap="inferno",
            mincnt=1,
            bins="log",
        )
        cb = fig.colorbar(hb, ax=ax, orientation="vertical", shrink=0.7)
        cb.set_label("Reported Incident Frequency (log scale)", fontsize=10)

        ax.set_title("Visualization B: Historical Crime Density Heatmap\n(Spatial Concentration via 2D Binning)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (Degrees)", fontsize=11)
        ax.set_ylabel("Latitude (Degrees)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "vis_b_crime_density_heatmap.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[SpatialEDA] Saved Visualization B: {path}")
        return path

    def plot_category_distribution(self) -> str:
        """
        Visualization C: Crime-category breakdown.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        top_cats = self.df[self.cat_col].value_counts().head(12)
        total = len(self.df)

        sns.barplot(
            y=top_cats.index,
            x=top_cats.values,
            hue=top_cats.index,
            palette="mako",
            ax=ax,
            legend=False,
        )
        ax.set_title("Visualization C: Distribution of Top 12 Primary Crime Categories", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Incident Count", fontsize=11)
        ax.set_ylabel("Crime Classification", fontsize=11)

        for p in ax.patches:
            w = p.get_width()
            pct = (w / total) * 100
            ax.annotate(f"{int(w):,} ({pct:.1f}%)", (w + total * 0.005, p.get_y() + p.get_height() / 2.),
                        ha="left", va="center", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "vis_c_crime_category_distribution.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[SpatialEDA] Saved Visualization C: {path}")
        return path

    def plot_area_distribution(self) -> str:
        """
        Visualization D: Area-level crime volume distribution.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        top_areas = self.df[self.area_col].dropna().value_counts().head(10)
        area_labels = [f"Area {int(a)}" for a in top_areas.index]

        sns.barplot(
            x=area_labels,
            y=top_areas.values,
            hue=area_labels,
            palette="crest",
            ax=ax,
            legend=False,
        )
        ax.set_title("Visualization D: Incident Counts by Top 10 Community Areas", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Community Area ID", fontsize=11)
        ax.set_ylabel("Incident Count", fontsize=11)
        plt.xticks(rotation=25)

        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{int(h):,}", (p.get_x() + p.get_width() / 2., h + 10),
                        ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.figures_dir, "vis_d_area_level_distribution.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[SpatialEDA] Saved Visualization D: {path}")
        return path

    def plot_geographic_coverage(self) -> str:
        """
        Visualization E: Study-area bounding box extent and coordinate distribution.
        """
        fig, (ax_lat, ax_lon) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Latitude distribution
        sns.histplot(self.df[self.lat_col], bins=40, kde=True, color="#2b5c8f", ax=ax_lat)
        ax_lat.set_title("Latitude Coordinate Distribution", fontsize=12, fontweight="bold")
        ax_lat.set_xlabel("Latitude (°N)")
        ax_lat.set_ylabel("Incident Frequency")
        ax_lat.axvline(self.df[self.lat_col].median(), color="red", linestyle="--", label=f"Median: {self.df[self.lat_col].median():.4f}°N")
        ax_lat.legend()

        # Longitude distribution
        sns.histplot(self.df[self.lon_col], bins=40, kde=True, color="#b24d3b", ax=ax_lon)
        ax_lon.set_title("Longitude Coordinate Distribution", fontsize=12, fontweight="bold")
        ax_lon.set_xlabel("Longitude (°W)")
        ax_lon.set_ylabel("Incident Frequency")
        ax_lon.axvline(self.df[self.lon_col].median(), color="red", linestyle="--", label=f"Median: {self.df[self.lon_col].median():.4f}°W")
        ax_lon.legend()

        fig.suptitle("Visualization E: Chicago Study-Area Geographic Extent & Coordinate Profiles", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.figures_dir, "vis_e_geographic_coverage.png")
        plt.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[SpatialEDA] Saved Visualization E: {path}")
        return path

    def generate_interactive_html_map(self, sample_size: int = 1500) -> str:
        """
        Generates a standalone, dependency-free interactive HTML map using OpenStreetMap and Leaflet.js.
        """
        sample_df = self.df.sample(n=min(len(self.df), sample_size), random_state=42)
        mean_lat = float(self.df[self.lat_col].mean())
        mean_lon = float(self.df[self.lon_col].mean())

        markers_data = []
        for _, row in sample_df.iterrows():
            markers_data.append({
                "lat": float(row[self.lat_col]),
                "lon": float(row[self.lon_col]),
                "type": str(row.get(self.cat_col, "Crime Incident")),
                "date": str(row.get(self.date_col, "")),
                "area": str(row.get(self.area_col, "N/A")),
            })

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Chicago Historical Crime Incidents Map</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; font-family: sans-serif; }}
        #map {{ height: 100vh; width: 100vw; }}
        .info-panel {{
            position: absolute; top: 10px; right: 10px; z-index: 1000;
            background: rgba(255, 255, 255, 0.95); padding: 12px 16px;
            border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            max-width: 320px; font-size: 13px; line-height: 1.4;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <h4 style="margin: 0 0 8px 0; color: #1e3a8a;">Chicago Crime Spatial Distribution</h4>
        <div><strong>Total Valid Records:</strong> {len(self.df):,}</div>
        <div><strong>Sample Displayed:</strong> {len(markers_data):,} incidents</div>
        <div><strong>Center:</strong> {mean_lat:.4f}°N, {mean_lon:.4f}°W</div>
        <p style="margin: 6px 0 0 0; color: #64748b; font-size: 11px;">
            Points represent recorded historical incidents. Point density indicates spatial concentration of historical reports.
        </p>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{mean_lat}, {mean_lon}], 11);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);

        var markers = {json.dumps(markers_data)};
        markers.forEach(function(m) {{
            var circle = L.circleMarker([m.lat, m.lon], {{
                radius: 4,
                fillColor: '#2563eb',
                color: '#1d4ed8',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.5
            }}).addTo(map);
            circle.bindPopup('<strong>' + m.type + '</strong><br>Area: ' + m.area + '<br>Date: ' + m.date);
        }});
    </script>
</body>
</html>"""

        map_path = os.path.join(self.output_dir, "interactive_spatial_map.html")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[SpatialEDA] Generated Interactive Leaflet HTML Map: {map_path}")
        return map_path


def run_spatial_eda(filepath: Optional[str] = None, output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Executes full Spatial EDA workflow and returns structured results.
    """
    from src.data_loader import load_and_validate_crime_data

    clean_df, col_map, audit = load_and_validate_crime_data(filepath)
    eda = SpatialEDA(clean_df, col_map, audit, output_dir=output_dir)

    summary_stats = eda.compute_summary_statistics()
    fig_paths = eda.generate_all_visualizations()
    summary_stats["figures"] = fig_paths

    # Save summary stats JSON
    summary_json_path = os.path.join(output_dir, "spatial_eda_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"[SpatialEDA] Saved summary metrics to: {summary_json_path}")

    return summary_stats


if __name__ == "__main__":
    results = run_spatial_eda()
    print("[SpatialEDA] Completed successfully.")
