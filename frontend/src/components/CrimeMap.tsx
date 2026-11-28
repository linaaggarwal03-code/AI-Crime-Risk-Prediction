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
  incidents?: IncidentPoint[];
  showNoise?: boolean;
  filterClusterId?: number | null;
  onSelectCluster?: (cluster: HotspotCluster | null) => void;
}

export default function CrimeMap({
  clusters,
  incidents = [],
  showNoise = true,
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

      // Render sampled/filtered historical incidents if provided
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
              DBSCAN Hotspot Cluster #${c.cluster_id}
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
    });

    return () => {
      isMounted = false;
    };
  }, [clusters, incidents, filterClusterId, showNoise, onSelectCluster, mapLoaded]);

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
        <div>
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-500 mr-2"></span>
          Hotspot Centroid Radius &prop; Incident Frequency
        </div>
        {incidents.length > 0 && (
          <div>
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-sky-400 mr-2"></span>
            Filtered Incident Sample ({incidents.length.toLocaleString()} pts)
          </div>
        )}
      </div>
    </div>
  );
}
