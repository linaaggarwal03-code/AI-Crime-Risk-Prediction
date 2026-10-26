"""
Helper script to generate clean, reproducible Jupyter Notebooks for:
1. Spatial Exploratory Data Analysis (02_spatial_eda.ipynb)
2. DBSCAN Hotspot Analysis & Parameter Tuning (03_dbscan_hotspot_analysis.ipynb)
"""

import json
import os


def create_spatial_eda_notebook(out_path="notebooks/02_spatial_eda.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Powered Urban Crime Risk & Hotspot Prediction System\n",
                "## Spatial Exploratory Data Analysis (Member 2: Aditi Verma)\n",
                "\n",
                "This notebook implements comprehensive geographic validation, spatial boundary auditing,\n",
                "and descriptive spatial analysis on City of Chicago crime records."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "from src.data_loader import load_and_validate_crime_data\n",
                "from src.spatial_eda import SpatialEDA, run_spatial_eda\n",
                "\n",
                "print(\"Spatial analysis libraries loaded successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Ingestion and Coordinate Boundary Validation\n",
                "Loading raw records, discovering coordinate fields dynamically, and auditing exclusion criteria."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "clean_df, col_map, audit = load_and_validate_crime_data()\n",
                "print(\"Column Mapping:\", col_map)\n",
                "print(\"Exclusion Audit:\", audit)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Descriptive Spatial Summary Statistics\n",
                "Computing geographic bounding box extents, center coordinates, and categorical volume."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "eda = SpatialEDA(clean_df, col_map, audit, output_dir='../outputs')\n",
                "stats = eda.compute_summary_statistics()\n",
                "print(\"Geographic Extent:\", stats['geographic_extent'])\n",
                "print(\"Top Categories:\", stats['categories']['top_10'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Spatial Visualizations (Visualizations A - E)\n",
                "Generating scatter distribution, 2D density heatmap, category charts, and coverage histograms."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig_paths = eda.generate_all_visualizations()\n",
                "for k, v in fig_paths.items():\n",
                "    print(f\"{k}: {v}\")"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path}")


def create_dbscan_notebook(out_path="notebooks/03_dbscan_hotspot_analysis.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AI-Powered Urban Crime Risk & Hotspot Prediction System\n",
                "## DBSCAN Crime Hotspot Detection & Parameter Tuning (Member 2: Aditi Verma)\n",
                "\n",
                "This notebook implements density-based spatial clustering of historical crime incidents\n",
                "using scikit-learn's DBSCAN with the spherical **Haversine** distance metric."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "from src.dbscan_clustering import execute_full_dbscan_pipeline, DBSCANHotspotDetector\n",
                "from src.data_loader import load_and_validate_crime_data\n",
                "\n",
                "print(\"DBSCAN clustering libraries loaded successfully.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Execute End-to-End DBSCAN Hotspot Pipeline\n",
                "Runs systematic hyperparameter sensitivity testing across eps (200m–1000m) and min_samples (15–100),\n",
                "executes primary clustering, and exports cluster summary metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "summary = execute_full_dbscan_pipeline(output_dir='../outputs')\n",
                "print(\"Identified Clusters:\", summary['results_summary']['num_clusters_identified'])\n",
                "print(\"Clustered Percentage:\", summary['results_summary']['clustered_percentage'])\n",
                "print(\"Noise Percentage:\", summary['results_summary']['noise_percentage'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Inspect Identified Top Hotspot Clusters\n",
                "Viewing the representative centroids, bounding envelopes, and dominant offense types."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cluster_df = pd.read_csv('../outputs/cluster_summary.csv')\n",
                "cluster_df.head(10)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Hyperparameter Sensitivity Evaluation\n",
                "Inspecting the trade-offs between neighborhood radius (eps) and minimum cluster density (min_samples)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "exp_df = pd.read_csv('../outputs/parameter_experiments.csv')\n",
                "exp_df.head(12)"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_path}")


if __name__ == "__main__":
    create_spatial_eda_notebook()
    create_dbscan_notebook()
