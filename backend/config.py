"""
Configuration module for the Urban Crime Risk Prediction Backend API.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

SAMPLE_DATA_PATH = DATA_DIR / "Crimes_Sample_2020_2025.csv"
FULL_DATA_PATH = DATA_DIR / "Crimes_-_2001_to_Present_20260820.csv"

CLUSTER_SUMMARY_PATH = OUTPUTS_DIR / "cluster_summary.csv"
HOTSPOT_RESULTS_PATH = OUTPUTS_DIR / "hotspot_results.csv"
ANALYSIS_SUMMARY_PATH = OUTPUTS_DIR / "analysis_summary.json"
PARAMETER_EXPERIMENTS_PATH = OUTPUTS_DIR / "parameter_experiments.csv"

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "*",
]
