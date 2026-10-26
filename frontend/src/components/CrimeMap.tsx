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

interface CrimeMapProps {
  clusters: HotspotCluster[];
  showNoise?: boolean;
  filterClusterId?: number | null;
  onSelectCluster?: (cluster: HotspotCluster | null) => void;
}

export default function CrimeMap({
  clusters,
  showNoise = true,
  filterClusterId = null,
  onSelectCluster,
}: CrimeMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const layersRef = useRef<any[]>([]);
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !mapContainerRef.current) return;

    let isMounted = true;

    // Dynamically import Leaflet
    import("leaflet").then((L) => {
      if (!isMounted || !mapContainerRef.current) return;

      if (!mapInstanceRef.current) {
        const map = L.map(mapContainerRef.current).setView([41.8781, -87.6298], 11);

        L.tileLayer(
          "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
          {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; OpenStreetMap',
            maxZoom: 18,
          }
        ).addTo(map);

        mapInstanceRef.current = map;
        setMapLoaded(true);
      }

      const map = mapInstanceRef.current;

      // Clear existing layers
      layersRef.current.forEach((l) => map.removeLayer(l));
      layersRef.current = [];

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

        // 1. Draw Bounding Envelope Rectangle
        const bounds: [[number, number], [number, number]] = [
          [c.min_latitude, c.min_longitude],
          [c.max_latitude, c.max_longitude],
        ];
        const rect = L.rectangle(bounds, {
          color: color,
          weight: 1,
          dashArray: "4, 4",
          fillOpacity: 0.12,
        }).addTo(map);
        layersRef.current.push(rect);

        // 2. Draw Centroid Circle Marker
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
  }, [clusters, filterClusterId, showNoise, onSelectCluster]);

  return (
    <div className="relative w-full h-full min-h-[500px] rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      <div ref={mapContainerRef} className="w-full h-full min-h-[500px]" />
      {!mapLoaded && (
        <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center text-slate-400">
          Loading interactive spatial layers...
        </div>
      )}
      <div className="absolute bottom-4 right-4 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700/60 text-slate-200 px-3 py-2 rounded-lg text-xs shadow-lg pointer-events-none">
        <span className="inline-block w-2.5 h-2.5 rounded-full bg-rose-500 mr-2"></span>
        Hotspot Centroid Radius ∝ Incident Frequency
      </div>
    </div>
  );
}
