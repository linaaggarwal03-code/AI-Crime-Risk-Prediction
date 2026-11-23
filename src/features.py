"""
Feature engineering module for AI-Powered Urban Crime Risk & Hotspot Prediction System.
Generates deterministic calendar signals, US federal holiday indicators,
autoregressive lags, strictly past-only rolling statistics, and cross-area spatial context.
"""

from datetime import date, timedelta
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def get_us_federal_holidays(years: List[int]) -> Set[date]:
    """
    Computes a reproducible, standard calendar of observed US federal holidays.
    Holidays included:
    - New Year's Day (Jan 1)
    - Martin Luther King Jr. Day (3rd Monday in January)
    - Washington's Birthday / Presidents' Day (3rd Monday in February)
    - Memorial Day (last Monday in May)
    - Juneteenth National Independence Day (June 19)
    - Independence Day (July 4)
    - Labor Day (1st Monday in September)
    - Columbus Day / Indigenous Peoples' Day (2nd Monday in October)
    - Veterans Day (November 11)
    - Thanksgiving Day (4th Thursday in November)
    - Christmas Day (December 25)
    - Holiday Eves with high urban mobility: July 3rd, New Year's Eve (Dec 31)
    """
    holidays = set()
    for y in years:
        # Fixed-date holidays
        holidays.add(date(y, 1, 1))    # New Year's Day
        if y >= 2021:
            holidays.add(date(y, 6, 19))   # Juneteenth National Independence Day (Federal holiday since 2021)
        holidays.add(date(y, 7, 3))    # July 3rd (Eve)
        holidays.add(date(y, 7, 4))    # Independence Day
        holidays.add(date(y, 11, 11))  # Veterans Day
        holidays.add(date(y, 12, 25))  # Christmas Day
        holidays.add(date(y, 12, 31))  # New Year's Eve

        # Floating holidays
        # MLK Day: 3rd Monday in Jan
        mondays_jan = [date(y, 1, d) for d in range(1, 32) if date(y, 1, d).weekday() == 0]
        holidays.add(mondays_jan[2])

        # Presidents' Day: 3rd Monday in Feb
        mondays_feb = [date(y, 2, d) for d in range(1, 29 + (1 if y % 4 == 0 else 0)) if date(y, 2, d).weekday() == 0]
        holidays.add(mondays_feb[2])

        # Memorial Day: Last Monday in May
        mondays_may = [date(y, 5, d) for d in range(1, 32) if date(y, 5, d).weekday() == 0]
        holidays.add(mondays_may[-1])

        # Labor Day: 1st Monday in Sept
        mondays_sep = [date(y, 9, d) for d in range(1, 31) if date(y, 9, d).weekday() == 0]
        holidays.add(mondays_sep[0])

        # Columbus Day: 2nd Monday in Oct
        mondays_oct = [date(y, 10, d) for d in range(1, 32) if date(y, 10, d).weekday() == 0]
        holidays.add(mondays_oct[1])

        # Thanksgiving Day: 4th Thursday in Nov
        thurs_nov = [date(y, 11, d) for d in range(1, 31) if date(y, 11, d).weekday() == 3]
        holidays.add(thurs_nov[3])

    return holidays


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates deterministic temporal and calendar signals from the 'date' column.
    All features represent strictly known exogenous time coordinates.
    """
    out = df.copy()
    dates_series = pd.to_datetime(out['date'])

    out['day_of_week'] = dates_series.dt.dayofweek  # 0=Mon, 6=Sun
    out['is_weekend'] = out['day_of_week'].isin([5, 6]).astype(int)
    out['month'] = dates_series.dt.month
    out['day_of_month'] = dates_series.dt.day
    out['day_of_year'] = dates_series.dt.dayofyear

    # Cyclical trigonometric transforms for continuous seasonal coordinates
    out['sin_dow'] = np.sin(2 * np.pi * out['day_of_week'] / 7.0)
    out['cos_dow'] = np.cos(2 * np.pi * out['day_of_week'] / 7.0)
    out['sin_month'] = np.sin(2 * np.pi * (out['month'] - 1) / 12.0)
    out['cos_month'] = np.cos(2 * np.pi * (out['month'] - 1) / 12.0)
    out['sin_doy'] = np.sin(2 * np.pi * (out['day_of_year'] - 1) / 365.25)
    out['cos_doy'] = np.cos(2 * np.pi * (out['day_of_year'] - 1) / 365.25)

    # US Federal Holiday Indicator
    unique_years = sorted(list(set(dates_series.dt.year)))
    holiday_dates = get_us_federal_holidays(unique_years)
    out['is_holiday'] = out['date'].apply(lambda d: 1 if d in holiday_dates else 0)

    return out


def add_lag_features(df: pd.DataFrame, lags: Tuple[int, ...] = (1, 2, 3, 7, 14, 28)) -> pd.DataFrame:
    """
    Computes autoregressive lag features per Community Area.
    
    CRITICAL SORTING & TEMPORAL INTEGRITY SAFEGUARD:
    Guarantees df is sorted by ['community_area', 'date'] before applying shift(k).
    Each lag_k represents strictly y(t-k), perfectly known prior to target day t.
    """
    out = df.copy()
    out = out.sort_values(by=['community_area', 'date']).reset_index(drop=True)

    grouped = out.groupby('community_area')['crime_count']
    for k in lags:
        out[f'lag_{k}'] = grouped.shift(k)

    return out


def add_rolling_features(df: pd.DataFrame, windows: Tuple[int, ...] = (7, 14, 30)) -> pd.DataFrame:
    """
    Computes rolling window statistics (mean, std, min, max) per Community Area.
    
    CRITICAL ANTI-LEAKAGE SAFEGUARD:
    Applies shift(1) BEFORE the rolling window calculation.
    For day t, rolling_mean_w = mean(y(t-1), y(t-2), ..., y(t-w)).
    Target day y(t) is NEVER included in the rolling aggregation.
    """
    out = df.copy()
    out = out.sort_values(by=['community_area', 'date']).reset_index(drop=True)

    # Shift by 1 day first so day t is strictly excluded
    shifted = out.groupby('community_area')['crime_count'].shift(1)

    for w in windows:
        roll = shifted.groupby(out['community_area']).rolling(w, min_periods=w)
        out[f'rolling_mean_{w}'] = roll.mean().reset_index(level=0, drop=True)
        out[f'rolling_std_{w}'] = roll.std().reset_index(level=0, drop=True)
        out[f'rolling_min_{w}'] = roll.min().reset_index(level=0, drop=True)
        out[f'rolling_max_{w}'] = roll.max().reset_index(level=0, drop=True)

    # Momentum ratio: compares immediate yesterday (lag_1) to 7-day average baseline
    out['ratio_lag1_to_rolling7'] = (out['lag_1'] + 1.0) / (out['rolling_mean_7'] + 1.0)

    return out


def add_spatial_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes citywide cross-area aggregate crime features at lag 1.
    Provides macroeconomic / citywide crime shock context without spatial target leakage.
    """
    out = df.copy()

    # Citywide total crime per date
    citywide_totals = out.groupby('date')['crime_count'].sum().sort_index()

    # Citywide past statistics (strictly shifted by 1)
    citywide_lag_1 = citywide_totals.shift(1).rename('citywide_lag_1')
    citywide_rolling_7 = citywide_totals.shift(1).rolling(7, min_periods=7).mean().rename('citywide_rolling_7')

    out = out.merge(citywide_lag_1, on='date', how='left')
    out = out.merge(citywide_rolling_7, on='date', how='left')

    return out


def build_feature_pipeline(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline sequentially.
    Ensures correct sorting and anti-leakage constraints at each stage.
    """
    df = add_calendar_features(grid_df)
    df = add_lag_features(df, lags=(1, 2, 3, 7, 14, 28))
    df = add_rolling_features(df, windows=(7, 14, 30))
    df = add_spatial_context_features(df)
    return df


def compute_training_risk_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    """
    Derives risk classification thresholds STRICTLY from the Training split.
    Validation and test splits never influence these threshold values.
    """
    counts = train_df['crime_count']
    q25 = float(counts.quantile(0.25))
    q50 = float(counts.median())
    q75 = float(counts.quantile(0.75))

    thresholds = {
        'low_threshold_q25': q25,
        'median_threshold_q50': q50,
        'high_threshold_q75': q75,
    }
    return thresholds


def assign_risk_tiers(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """
    Applies fixed training-derived thresholds to assign discrete risk tiers.
    """
    out = df.copy()
    q25 = thresholds['low_threshold_q25']
    q75 = thresholds['high_threshold_q75']

    def _tier(val):
        if val <= q25:
            return 'Low'
        elif val < q75:
            return 'Normal'
        else:
            return 'High'

    out['risk_level'] = out['crime_count'].apply(_tier)
    out['is_high_risk'] = (out['crime_count'] >= q75).astype(int)
    return out
