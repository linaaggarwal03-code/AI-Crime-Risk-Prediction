"""
Data preprocessing module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Handles raw incident loading, spatial mapping audit, dense space-time grid construction,
and chronological dataset splitting.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd


# Explicit empirical beat-to-area mappings verified to be 100% unambiguous
UNAMBIGUOUS_BEAT_TO_COMMUNITY_AREA = {
    1654: 76,  # 100.0% of historical Beat 1654 records are in Area 76 (O'Hare)
    2411: 2,   # 100.0% of historical Beat 2411 records are in Area 2 (West Ridge)
}

# Beats with genuine boundary ambiguity across multiple Community Areas
AMBIGUOUS_BEATS = {
    1614: [76, 10, 17],  # Spans O'Hare (85.5%), Norwood Park (11.4%), Dunning (3.2%)
    2212: [75, 72],      # Spans Morgan Park (77.1%), Beverly (22.9%)
}


def load_and_audit_incidents(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw Chicago crime incidents and performs a rigorous spatial audit.
    
    Returns:
        resolved_df: Valid incident records ready for area-date aggregation.
        unresolved_df: Records with ambiguous spatial assignments, preserved for audit.
        audit_metrics: Dictionary tracking complete incident-count conservation.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw crime dataset not found at: {csv_path}")

    # Load required incident columns
    usecols = [
        'ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type',
        'Description', 'Location Description', 'Arrest', 'Domestic',
        'Beat', 'District', 'Ward', 'Community Area', 'Latitude', 'Longitude'
    ]
    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    total_raw_incidents = len(df)

    # 1. Parse incident dates
    df['parsed_date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')
    df['date'] = df['parsed_date'].dt.date

    # 2. Audit missing Community Area records
    is_ca_missing = df['Community Area'].isna()
    missing_ca_df = df[is_ca_missing].copy()
    valid_ca_df = df[~is_ca_missing].copy()
    valid_ca_df['community_area'] = valid_ca_df['Community Area'].astype(int)

    # Resolve unambiguous beats
    can_resolve = missing_ca_df['Beat'].isin(UNAMBIGUOUS_BEAT_TO_COMMUNITY_AREA)
    resolved_missing_df = missing_ca_df[can_resolve].copy()
    resolved_missing_df['community_area'] = resolved_missing_df['Beat'].map(UNAMBIGUOUS_BEAT_TO_COMMUNITY_AREA).astype(int)

    # Isolate ambiguous records
    unresolved_df = missing_ca_df[~can_resolve].copy()

    # Combine all verified, resolved incident records
    resolved_df = pd.concat([valid_ca_df, resolved_missing_df], ignore_index=True)

    # 3. Incident Conservation Audit
    audit_metrics = {
        'total_raw_incidents': total_raw_incidents,
        'initially_valid_community_area': len(valid_ca_df),
        'missing_community_area_total': len(missing_ca_df),
        'resolved_unambiguous_incidents': len(resolved_missing_df),
        'unresolved_ambiguous_incidents': len(unresolved_df),
        'total_resolved_for_grid': len(resolved_df),
        'conservation_verified': (len(resolved_df) + len(unresolved_df)) == total_raw_incidents,
        'missing_coordinates_count': int(resolved_df['Latitude'].isna().sum()),
        'missing_coordinates_pct': float((resolved_df['Latitude'].isna().sum() / len(resolved_df)) * 100),
    }

    if not audit_metrics['conservation_verified']:
        raise ValueError("Incident conservation check failed during raw loading!")

    return resolved_df, unresolved_df, audit_metrics


def build_spatial_temporal_grid(
    resolved_df: pd.DataFrame,
    start_date: str = '2020-01-01',
    end_date: str = '2025-12-31'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregates point incidents into a complete, dense Cartesian product grid
    of (Community Area x Calendar Date), imputing zero-crime area-days.
    
    Returns:
        grid_df: Dense DataFrame sorted strictly by ['community_area', 'date'].
        grid_stats: Verification metrics including zero-crime sparsity counts.
    """
    # Aggregate observed daily crime counts
    daily_counts = resolved_df.groupby(['community_area', 'date']).size().rename('crime_count')

    # Construct the full Cartesian product grid across all 77 Community Areas
    all_areas = list(range(1, 78))
    date_range = pd.date_range(start_date, end_date).date
    full_index = pd.MultiIndex.from_product(
        [all_areas, date_range],
        names=['community_area', 'date']
    )

    # Reindex to guarantee dense coverage; fill zero-crime days with 0
    grid_series = daily_counts.reindex(full_index, fill_value=0)
    grid_df = grid_series.reset_index()

    # CRUCIAL SAFEGUARD: Guarantee strict ascending sort order
    grid_df = grid_df.sort_values(by=['community_area', 'date']).reset_index(drop=True)

    # Verification: incident conservation across the grid
    total_grid_crimes = int(grid_df['crime_count'].sum())
    total_resolved_crimes = len(resolved_df)
    if total_grid_crimes != total_resolved_crimes:
        raise ValueError(
            f"Grid aggregation mismatch: sum={total_grid_crimes:,} vs resolved={total_resolved_crimes:,}"
        )

    zero_crime_cells = int((grid_df['crime_count'] == 0).sum())
    grid_stats = {
        'total_cells': len(grid_df),
        'expected_cells': len(all_areas) * len(date_range),
        'unique_areas': grid_df['community_area'].nunique(),
        'unique_dates': grid_df['date'].nunique(),
        'min_date': str(grid_df['date'].min()),
        'max_date': str(grid_df['date'].max()),
        'zero_crime_cells': zero_crime_cells,
        'zero_crime_pct': float((zero_crime_cells / len(grid_df)) * 100),
        'total_aggregated_crimes': total_grid_crimes,
        'mean_daily_crimes_per_area': float(grid_df['crime_count'].mean()),
        'median_daily_crimes_per_area': float(grid_df['crime_count'].median()),
        'max_daily_crimes_in_single_area': int(grid_df['crime_count'].max()),
    }

    return grid_df, grid_stats


def split_chronological(
    grid_df: pd.DataFrame,
    warmup_end_date: str = '2020-01-30',
    train_end_date: str = '2023-12-31',
    val_end_date: str = '2024-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the space-time grid into strictly chronological splits:
    - warmup_df: 2020-01-01 to 2020-01-30 (used to initialize 30-day lags; pruned from training)
    - train_df: 2020-01-31 to 2023-12-31 (used for feature parameter fitting and model training)
    - val_df: 2024-01-01 to 2024-12-31 (used for validation and tuning)
    - test_df: 2025-01-01 to 2025-12-31 (held-out final evaluation)
    """
    warmup_end = pd.to_datetime(warmup_end_date).date()
    train_end = pd.to_datetime(train_end_date).date()
    val_end = pd.to_datetime(val_end_date).date()

    warmup_mask = grid_df['date'] <= warmup_end
    train_mask = (grid_df['date'] > warmup_end) & (grid_df['date'] <= train_end)
    val_mask = (grid_df['date'] > train_end) & (grid_df['date'] <= val_end)
    test_mask = grid_df['date'] > val_end

    warmup_df = grid_df[warmup_mask].copy().reset_index(drop=True)
    train_df = grid_df[train_mask].copy().reset_index(drop=True)
    val_df = grid_df[val_mask].copy().reset_index(drop=True)
    test_df = grid_df[test_mask].copy().reset_index(drop=True)

    return warmup_df, train_df, val_df, test_df
