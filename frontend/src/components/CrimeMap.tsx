"use client";

import React, { useEffect, useRef, useState } from "react";

interface HotspotCluster {
  cluster_id: number;
  incident_count: number;
  percentage_of_total: number;
  centroid_latitude: number;
  centroid_longitude: number;
  min_latitude: number;
  max_latitude: number;
  min_longitude: number;
  max_longitude: number;
  primary_crime_type: string;
  dominant_community_area: string | number;
}

interface MlHotspotCluster {
  hotspot_rank: number;
  hotspot_id: number;
  incident_count_total: number;
  incident_count_in_area: number;
  is_dominant_in_this_area: boolean;
  centroid: {
    latitude: number;
    longitude: number;
  };
  radius_meters: number;
  top_crime_types: string;
}

interface KdeSurfaceData {
  grid_size: number;
  bandwidth_meters: number;
  target_year: number;
  lat_grid: number[][];
  lon_grid: number[][];
  density_grid: number[][];
}

interface IncidentPoint {
  id?: string;
  date?: string;
  primary_type?: string;
  community_area?: string;
  latitude: number;
  longitude: number;
  cluster_id?: number | null;
}

interface CrimeMapProps {
  clusters: HotspotCluster[];
  mlClusters?: MlHotspotCluster[];
  kdeSurface?: KdeSurfaceData | null;
  incidents?: IncidentPoint[];
  showNoise?: boolean;
  showIncidents?: boolean;
  showSpatialClusters?: boolean;
  showMlHotspots?: boolean;
  showKdeSurface?: boolean;
  selectedAreaId?: number | null;
  filterClusterId?: number | null;
  onSelectCluster?: (cluster: HotspotCluster | null) => void;
}

export default function CrimeMap({
  clusters,
  mlClusters = [],
  kdeSurface = null,
  incidents = [],
  showNoise = true,
  showIncidents = true,
  showSpatialClusters = true,
  showMlHotspots = false,
  showKdeSurface = false,
  selectedAreaId = null,
  filterClusterId = null,
  onSelectCluster,
}: CrimeMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const tileLayersRef = useRef<any[]>([]);
  const layersRef = useRef<any[]>([]);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapTheme, setMapTheme] = useState<"dark" | "light">("dark");

  // 1. Initialize map & swap basemap tiles when mapTheme changes
  useEffect(() => {
    if (typeof window === "undefined" || !mapContainerRef.current) return;

    let isMounted = true;

    import("leaflet").then((L) => {
      if (!isMounted || !mapContainerRef.current) return;

      if (!mapInstanceRef.current) {
        const map = L.map(mapContainerRef.current, {
          zoomControl: true,
          preferCanvas: true,
        }).setView([41.8781, -87.6298], 11);
        mapInstanceRef.current = map;
        setMapLoaded(true);

        // Crucial fix: invalidate size after mount and layout settlement
        setTimeout(() => {
          if (mapInstanceRef.current) {
            mapInstanceRef.current.invalidateSize();
          }
        }, 200);

        // Keep map responsive to container resize
        if (typeof ResizeObserver !== "undefined" && mapContainerRef.current) {
          const ro = new ResizeObserver(() => {
            if (mapInstanceRef.current) {
              mapInstanceRef.current.invalidateSize();
            }
          });
          ro.observe(mapContainerRef.current);
        }
      }

      const map = mapInstanceRef.current;

      // Remove existing basemap tile layers
      tileLayersRef.current.forEach((tl) => map.removeLayer(tl));
      tileLayersRef.current = [];

      if (mapTheme === "dark") {
        const darkBase = L.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
          {
            attribution: "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
            maxZoom: 16,
          }
        ).addTo(map);
        const darkRef = L.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
          {
            maxZoom: 16,
          }
        ).addTo(map);
        darkBase.bringToBack();
        tileLayersRef.current.push(darkBase, darkRef);
      } else {
        const lightStreet = L.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
          {
            attribution: "Tiles &copy; Esri &mdash; Source: Esri, HERE, Garmin, OpenStreetMap",
            maxZoom: 18,
          }
        ).addTo(map);
        lightStreet.bringToBack();
        tileLayersRef.current.push(lightStreet);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [mapTheme]);

  // 2. Render clusters and filtered incidents
  useEffect(() => {
    if (typeof window === "undefined" || !mapInstanceRef.current) return;

    let isMounted = true;

    import("leaflet").then((L) => {
      if (!isMounted || !mapInstanceRef.current) return;

      const map = mapInstanceRef.current;

      // Clear existing vector layers
      layersRef.current.forEach((l) => map.removeLayer(l));
      layersRef.current = [];

      // 1. Render KDE density surface (50x50 macroscopic ML grid)
      if (
        showKdeSurface &&
        kdeSurface &&
        Array.isArray(kdeSurface.density_grid) &&
        kdeSurface.density_grid.length > 1
      ) {
        const { lat_grid, lon_grid, density_grid } = kdeSurface;
        let maxDensity = 0;
        for (let i = 0; i < density_grid.length; i++) {
          for (let j = 0; j < density_grid[i].length; j++) {
            if (density_grid[i][j] > maxDensity) maxDensity = density_grid[i][j];
          }
        }

        if (maxDensity > 0) {
          const dLat = Math.abs(lat_grid[1][0] - lat_grid[0][0]) || 0.007;
          const dLon = Math.abs(lon_grid[0][1] - lon_grid[0][0]) || 0.009;

          for (let i = 0; i < density_grid.length; i++) {
            for (let j = 0; j < density_grid[i].length; j++) {
              const val = density_grid[i][j];
              const norm = val / maxDensity;
              if (norm < 0.03) continue;

              const fillColor =
                norm >= 0.7
                  ? "#dc2626"
                  : norm >= 0.4
                  ? "#ea580c"
                  : norm >= 0.15
                  ? "#f59e0b"
                  : "#fde047";

              const lat = lat_grid[i][j];
              const lon = lon_grid[i][j];
              const cell = L.rectangle(
                [
                  [lat - dLat / 2, lon - dLon / 2],
                  [lat + dLat / 2, lon + dLon / 2],
                ],
                {
                  stroke: false,
                  fillColor,
                  fillOpacity: Math.min(0.62, 0.14 + norm * 0.48),
                }
              ).addTo(map);

              cell.bindPopup(
                `<div style="font-family:sans-serif;font-size:12px;">
                  <strong>KDE Density Cell (${kdeSurface.target_year})</strong><br/>
                  Relative Intensity: ${(norm * 100).toFixed(1)}%<br/>
                  Density: ${val.toExponential(2)}<br/>
                  Center: ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°W
                </div>`
              );
              layersRef.current.push(cell);
            }
          }
        }
      }

      // 2. Render sampled/filtered historical incidents if enabled
      if (showIncidents) {
        incidents.forEach((pt) => {
          const isNoise = pt.cluster_id === -1 || pt.cluster_id === null;
          if (!showNoise && isNoise) return;
          const dot = L.circleMarker([pt.latitude, pt.longitude], {
            radius: 3,
            fillColor: isNoise ? "#94a3b8" : "#0284c7",
            color: isNoise ? "#64748b" : "#0369a1",
            weight: 0.5,
            opacity: 0.75,
            fillOpacity: 0.6,
          }).addTo(map);
          dot.bindPopup(
            `<div style="font-family:sans-serif;font-size:12px;">
              <strong>${pt.primary_type || "Incident"}</strong><br/>
              Area: ${pt.community_area || "N/A"} | Cluster: ${pt.cluster_id === -1 ? "Noise" : "#" + pt.cluster_id}<br/>
              <span style="color:#94a3b8">${pt.date || ""}</span>
            </div>`
          );
          layersRef.current.push(dot);
        });
      }

      // 3. Render 29 spatial DBSCAN clusters (Member 2 sample pipeline)
      if (showSpatialClusters) {
        const palette = [
          "#f43f5e", "#3b82f6", "#10b981", "#f59e0b",
          "#a855f7", "#06b6d4", "#f97316", "#6366f1",
          "#ec4899", "#84cc16", "#14b8a6", "#8b5cf6"
        ];

        const filteredClusters = filterClusterId !== null
          ? clusters.filter((c) => c.cluster_id === filterClusterId)
          : clusters;

        filteredClusters.forEach((c, idx) => {
          const color = palette[idx % palette.length];

          const bounds: [[number, number], [number, number]] = [
            [c.min_latitude, c.min_longitude],
            [c.max_latitude, c.max_longitude],
          ];
          const rect = L.rectangle(bounds, {
            color: color,
            weight: 1.5,
            dashArray: "4, 4",
            fillOpacity: 0.12,
          }).addTo(map);
          layersRef.current.push(rect);

          const marker = L.circleMarker([c.centroid_latitude, c.centroid_longitude], {
            radius: Math.min(18, Math.max(7, Math.log(c.incident_count) * 2.2)),
            fillColor: color,
            color: "#ffffff",
            weight: 1.5,
            opacity: 0.9,
            fillOpacity: 0.8,
          }).addTo(map);

          const popupContent = `
            <div style="font-family: sans-serif; font-size: 13px;">
              <div style="font-weight: bold; font-size: 14px; color: ${color}; margin-bottom: 4px;">
                Spatial DBSCAN Cluster #${c.cluster_id} (29-Cluster System)
              </div>
              <div><strong>Incident Count:</strong> ${c.incident_count.toLocaleString()} (${c.percentage_of_total}%)</div>
              <div><strong>Primary Crime:</strong> ${c.primary_crime_type}</div>
              <div><strong>Community Area:</strong> Area ${c.dominant_community_area}</div>
              <div><strong>Coordinates:</strong> ${c.centroid_latitude.toFixed(4)}°N, ${c.centroid_longitude.toFixed(4)}°W</div>
              <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                Density Envelope: [${c.min_latitude.toFixed(3)}, ${c.min_longitude.toFixed(3)}] to [${c.max_latitude.toFixed(3)}, ${c.max_longitude.toFixed(3)}]
              </div>
            </div>
          `;
          marker.bindPopup(popupContent);

          marker.on("click", () => {
            if (onSelectCluster) onSelectCluster(c);
          });

          layersRef.current.push(marker);
        });
      }

      // 4. Render 617-Cluster ML Cross-Boundary DBSCAN Hotspots for selected Community Area
      if (showMlHotspots && mlClusters.length > 0) {
        mlClusters.forEach((mc) => {
          const ring = L.circle([mc.centroid.latitude, mc.centroid.longitude], {
            radius: Math.min(mc.radius_meters, 2800),
            color: "#9333ea",
            weight: 1.5,
            fillColor: "#a855f7",
            fillOpacity: 0.14,
          }).addTo(map);
          layersRef.current.push(ring);

          const marker = L.circleMarker([mc.centroid.latitude, mc.centroid.longitude], {
            radius: Math.min(16, Math.max(6, Math.log(Math.max(1, mc.incident_count_in_area)) * 1.8)),
            fillColor: mc.is_dominant_in_this_area ? "#7c3aed" : "#c084fc",
            color: "#ffffff",
            weight: 2,
            opacity: 0.95,
            fillOpacity: 0.9,
          }).addTo(map);

          marker.bindPopup(
            `<div style="font-family: sans-serif; font-size: 13px;">
              <div style="font-weight: bold; font-size: 14px; color: #7c3aed; margin-bottom: 4px;">
                ML Hotspot #${mc.hotspot_id} (Rank ${mc.hotspot_rank} &middot; 617-Cluster System)
              </div>
              <div><strong>In Area ${selectedAreaId ?? ""}:</strong> ${mc.incident_count_in_area.toLocaleString()} incidents</div>
              <div><strong>Total Cluster:</strong> ${mc.incident_count_total.toLocaleString()} incidents</div>
              <div><strong>Boundary Status:</strong> ${
                mc.is_dominant_in_this_area
                  ? "Dominant in this Area"
                  : "Cross-Boundary (Dominant elsewhere)"
              }</div>
              <div><strong>Radius:</strong> ${mc.radius_meters.toFixed(1)} m</div>
              <div style="font-size: 11px; color: #475569; margin-top: 4px;">
                <strong>Top Crimes:</strong> ${mc.top_crime_types}
              </div>
            </div>`
          );
          layersRef.current.push(marker);
        });
      }
    });

    return () => {
      isMounted = false;
    };
  }, [
    clusters,
    mlClusters,
    kdeSurface,
    incidents,
    filterClusterId,
    showNoise,
    showIncidents,
    showSpatialClusters,
    showMlHotspots,
    showKdeSurface,
    selectedAreaId,
    onSelectCluster,
    mapLoaded,
  ]);

  return (
    <div className="relative w-full h-full min-h-[500px] rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      <div ref={mapContainerRef} className="w-full h-full min-h-[500px]" />

      {/* Dark / Light Mode Map Toggle */}
      <div className="absolute top-3 right-3 z-[400] flex items-center bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-lg p-1 shadow-lg text-xs">
        <button
          type="button"
          onClick={() => setMapTheme("dark")}
          className={`px-2.5 py-1 rounded-md font-medium transition ${
            mapTheme === "dark"
              ? "bg-indigo-600 text-white shadow"
              : "text-slate-300 hover:text-white"
          }`}
        >
          Dark Map
        </button>
        <button
          type="button"
          onClick={() => setMapTheme("light")}
          className={`px-2.5 py-1 rounded-md font-medium transition ${
            mapTheme === "light"
              ? "bg-amber-500 text-slate-950 shadow font-semibold"
              : "text-slate-300 hover:text-white"
          }`}
        >
          Light Map
        </button>
      </div>

      {!mapLoaded && (
        <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center text-slate-400">
          Loading interactive spatial layers...
        </div>
      )}
      <div className="absolute bottom-4 right-4 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700/60 text-slate-200 px-3 py-2 rounded-lg text-xs shadow-lg pointer-events-none space-y-1">
        {showSpatialClusters && (
          <div>
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-500 mr-2"></span>
            29 Spatial DBSCAN Clusters (&epsilon;=500m, Sample)
          </div>
        )}
        {showMlHotspots && (
          <div>
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-purple-500 border border-white mr-2"></span>
            617-Cluster ML Hotspots ({mlClusters.length} in Area {selectedAreaId ?? "N/A"})
          </div>
        )}
        {showKdeSurface && kdeSurface && (
          <div>
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-gradient-to-r from-yellow-300 to-red-600 mr-2"></span>
            KDE Density Surface ({kdeSurface.grid_size}&times;{kdeSurface.grid_size}, {kdeSurface.target_year})
          </div>
        )}
        {showIncidents && incidents.length > 0 && (
          <div>
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-sky-400 mr-2"></span>
            Historical Incidents ({incidents.length.toLocaleString()} pts)
          </div>
        )}
      </div>
    </div>
  );
}
