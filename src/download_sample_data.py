"""
Download authentic Chicago Police Department reported crime data (2020-2025)
from the City of Chicago Open Data Portal (Socrata API).
Preserves the exact schema and column types used by Lina Aggarwal (Member 1).
"""

import os
import certifi
import requests
import pandas as pd


def fetch_chicago_crime_data(output_path: str = "data/Crimes_Sample_2020_2025.csv", limit: int = 25000):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Query Chicago Data Portal API for real incidents between 2020 and 2025
    url = f"https://data.cityofchicago.org/resource/ijzp-q8t2.json?$limit={limit}&$where=year>=2020%20AND%20year<=2025&$order=date%20DESC"
    print(f"Fetching {limit} authentic records from City of Chicago Data Portal...")
    
    headers = {"User-Agent": "UrbanCrimeHotspotResearch/1.0"}
    response = requests.get(url, headers=headers, verify=certifi.where(), timeout=60)
    response.raise_for_status()
    records = response.json()
    
    df = pd.DataFrame(records)
    print(f"Successfully received {len(df)} records.")

    # Standardize column names to match Lina's 22-column dataset schema
    column_mapping = {
        "id": "ID",
        "case_number": "Case Number",
        "date": "Date",
        "block": "Block",
        "iucr": "IUCR",
        "primary_type": "Primary Type",
        "description": "Description",
        "location_description": "Location Description",
        "arrest": "Arrest",
        "domestic": "Domestic",
        "beat": "Beat",
        "district": "District",
        "ward": "Ward",
        "community_area": "Community Area",
        "fbi_code": "FBI Code",
        "x_coordinate": "X Coordinate",
        "y_coordinate": "Y Coordinate",
        "year": "Year",
        "updated_on": "Updated On",
        "latitude": "Latitude",
        "longitude": "Longitude",
    }
    df = df.rename(columns=column_mapping)
    
    # Format 'Location' string if coordinates exist
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
        df["Location"] = df.apply(
            lambda r: f"({r['Latitude']}, {r['Longitude']})" if pd.notnull(r["Latitude"]) and pd.notnull(r["Longitude"]) else None,
            axis=1
        )
    
    # Convert numerical and boolean fields
    for col in ["X Coordinate", "Y Coordinate", "Ward", "Community Area", "Beat", "District", "Year", "ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Arrest" in df.columns:
        df["Arrest"] = df["Arrest"].astype(str).str.lower() == "true"
    if "Domestic" in df.columns:
        df["Domestic"] = df["Domestic"].astype(str).str.lower() == "true"

    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    return df


if __name__ == "__main__":
    fetch_chicago_crime_data()
