"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import {
  LayoutDashboard,
  MapPin,
  Flame,
  BarChart3,
  ShieldAlert,
  BookOpen,
  Layers,
  Sparkles,
  Compass,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Database,
} from "lucide-react";

import analysisData from "../data/analysisData.json";

// Dynamically import CrimeMap to ensure client-only mounting
const CrimeMap = dynamic(() => import("../components/CrimeMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[550px] bg-slate-900 border border-slate-800 rounded-xl flex items-center justify-center text-slate-400">
      Initializing Spatial Map Engine...
    </div>
  ),
});

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "map" | "hotspots" | "analytics" | "prediction" | "methodology"
  >("dashboard");

  const [selectedCluster, setSelectedCluster] = useState<any>(null);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);

  // Prediction Form State
  const [predLat, setPredLat] = useState<number>(41.8781);
  const [predLon, setPredLon] = useState<number>(-87.6298);
  const [predDate, setPredDate] = useState<string>("2026-10-27");
  const [predTime, setPredTime] = useState<string>("14:30");
  const [predResult, setPredResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  const { summary, clusters, experiments, eda } = analysisData;
  const kpis = summary.results_summary;
  const params = summary.selected_hyperparameters;

  // Check live backend connectivity
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((res) => {
        if (res.ok) setBackendOnline(true);
      })
      .catch(() => setBackendOnline(false));
  }, []);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);

    try {
      const res = await fetch("http://localhost:8000/api/predict-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonBody({
          latitude: predLat,
          longitude: predLon,
          date: predDate,
          time: predTime,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setPredResult(data);
        setIsPredicting(false);
        return;
      }
    } catch {
      // Local fallback calculation if backend is offline
    }

    // Local client-side calculation fallback
    let nearestC = clusters[0];
    let minD = 999999;
    clusters.forEach((c: any) => {
      const dLat = ((c.centroid_latitude - predLat) * Math.PI) / 180;
      const dLon = ((c.centroid_longitude - predLon) * Math.PI) / 180;
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos((predLat * Math.PI) / 180) *
          Math.cos((c.centroid_latitude * Math.PI) / 180) *
          Math.sin(dLon / 2) *
          Math.sin(dLon / 2);
      const dist = 6371.0088 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      if (dist < minD) {
        minD = dist;
        nearestC = c;
      }
    });

    let tier = "Low / Baseline (Dispersed Location > 1.5km from Hotspots)";
    if (minD <= 0.5) {
      tier = "High (Proximity to Historic Hotspot < 500m)";
    } else if (minD <= 1.5) {
      tier = "Moderate (Proximity to Historic Hotspot 0.5km - 1.5km)";
    }

    setPredResult({
      status: "PROVISIONAL_SPATIAL_PROXIMITY_ONLY",
      query_coordinates: { latitude: predLat, longitude: predLon },
      nearest_hotspot: nearestC,
      distance_to_hotspot_km: parseFloat(minD.toFixed(3)),
      provisional_spatial_risk_tier: tier,
      model_integration_status: "PENDING_LINA_SUPERVISED_MODEL_EXPORT",
      disclaimer:
        "Notice: This is a geometric proximity calculation relative to historical DBSCAN clusters. Final supervised classification and count forecasting are pending completion of Member 1's (Lina) model testing and artifact serialization. This does NOT represent a confirmed future crime forecast.",
    });
    setIsPredicting(false);
  };

  function jsonBody(obj: any) {
    return JSON.stringify(obj);
  }

  const setPresetLocation = (name: string, lat: number, lon: number) => {
    setPredLat(lat);
    setPredLon(lon);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col justify-between p-4 z-20">
        <div>
          {/* Logo / Header */}
          <div className="flex items-center gap-3 px-2 py-3 mb-6 border-b border-slate-800">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-rose-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-900/30">
              <ShieldAlert className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-wide text-slate-100">
                Urban Crime AI
              </h1>
              <p className="text-[11px] text-slate-400">
                Spatial Hotspot Intelligence
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            {[
              { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
              { id: "map", label: "Historical Crime Map", icon: MapPin },
              { id: "hotspots", label: "Hotspot Analysis", icon: Flame },
              { id: "analytics", label: "Analytics", icon: BarChart3 },
              { id: "prediction", label: "Risk Prediction", icon: ShieldAlert },
              { id: "methodology", label: "Methodology", icon: BookOpen },
            ].map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id as any)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 font-semibold"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? "text-indigo-400" : "text-slate-500"
                    }`}
                  />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Member & Academic Info */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 text-[11px] space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Contributor:</span>
            <span className="font-medium text-slate-200">
              Aditi (Member 2)
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-400">Milestone:</span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 font-mono">
              Milestone 2
            </span>
          </div>
          <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
            <span className="text-slate-400">Backend API:</span>
            <span
              className={`flex items-center gap-1.5 font-medium ${
                backendOnline ? "text-emerald-400" : "text-amber-400"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
                }`}
              />
              {backendOnline ? ":8000 Live" : "Standalone"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-y-auto">
        {/* Top Bar */}
        <header className="h-16 border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-10">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              {activeTab === "dashboard" && "Executive Spatial Intelligence Dashboard"}
              {activeTab === "map" && "Geographic Distribution & Density Map"}
              {activeTab === "hotspots" && "DBSCAN Crime Hotspot Analysis (Haversine Metric)"}
              {activeTab === "analytics" && "Urban Crime Statistical & Area Analytics"}
              {activeTab === "prediction" && "Provisional Spatial Risk Prediction Interface"}
              {activeTab === "methodology" && "Methodology, Mathematical Formulations & Ethical Safeguards"}
            </h2>
            <p className="text-xs text-slate-400">
              Study Domain: City of Chicago Reported Crimes (2020–2025)
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700/60 text-slate-300">
              <span className="text-slate-400 mr-1.5">Analyzed Records:</span>
              <strong className="text-indigo-400 font-mono">
                {kpis.total_records_analyzed.toLocaleString()}
              </strong>
            </div>
            <div className="px-3 py-1.5 rounded-md bg-slate-800 border border-slate-700/60 text-slate-300">
              <span className="text-slate-400 mr-1.5">Clusters:</span>
              <strong className="text-rose-400 font-mono">
                {kpis.num_clusters_identified}
              </strong>
            </div>
          </div>
        </header>

        {/* Tab View Container */}
        <div className="p-8 flex-1">
          {/* TAB 1: DASHBOARD */}
          {activeTab === "dashboard" && (
            <div className="space-y-6">
              {/* Metric Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-medium">Valid Crime Incidents</span>
                    <Database className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {kpis.total_records_analyzed.toLocaleString()}
                  </div>
                  <div className="text-[11px] text-emerald-400 mt-2 flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    99.84% Geographic completeness
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-medium">Hotspot Clusters</span>
                    <Flame className="w-4 h-4 text-rose-400" />
                  </div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {kpis.num_clusters_identified}
                  </div>
                  <div className="text-[11px] text-slate-400 mt-2">
                    Radius &epsilon; = {params.eps_meters}m | min_samples = {params.min_samples}
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-medium">Clustered Incidents</span>
                    <Layers className="w-4 h-4 text-blue-400" />
                  </div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {kpis.clustered_incidents.toLocaleString()}
                  </div>
                  <div className="text-[11px] text-blue-400 mt-2">
                    {kpis.clustered_percentage}% assigned to spatial hotspots
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <div className="flex items-center justify-between text-slate-400 mb-2">
                    <span className="text-xs font-medium">Noise / Dispersed</span>
                    <Compass className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {kpis.noise_incidents.toLocaleString()}
                  </div>
                  <div className="text-[11px] text-amber-400/90 mt-2">
                    {kpis.noise_percentage}% isolated historical incidents
                  </div>
                </div>
              </div>

              {/* Map & Cluster Breakdown Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-200">
                        Primary Spatial Hotspots in Chicago
                      </h3>
                      <p className="text-xs text-slate-400">
                        Identified via DBSCAN using Spherical Haversine Metric
                      </p>
                    </div>
                    <button
                      onClick={() => setActiveTab("map")}
                      className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1 font-medium"
                    >
                      Full Interactive Map <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <div className="h-[420px]">
                    <CrimeMap
                      clusters={clusters}
                      filterClusterId={clusterFilter}
                      onSelectCluster={setSelectedCluster}
                    />
                  </div>
                </div>

                {/* Top Hotspots List */}
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200 mb-1">
                      Highest-Density Clusters
                    </h3>
                    <p className="text-xs text-slate-400 mb-4">
                      Ranked by historical incident concentration
                    </p>

                    <div className="space-y-2.5 max-h-[360px] overflow-y-auto pr-1">
                      {clusters.slice(0, 6).map((c: any) => (
                        <div
                          key={c.cluster_id}
                          onClick={() => {
                            setClusterFilter(
                              clusterFilter === c.cluster_id ? null : c.cluster_id
                            );
                            setSelectedCluster(c);
                          }}
                          className={`p-3 rounded-lg border text-xs cursor-pointer transition-all ${
                            clusterFilter === c.cluster_id
                              ? "bg-indigo-950/60 border-indigo-500"
                              : "bg-slate-950/50 border-slate-800 hover:border-slate-700"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-slate-200">
                              Cluster #{c.cluster_id} (Area {c.dominant_community_area})
                            </span>
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-rose-950 text-rose-300 font-mono font-medium">
                              {c.incident_count.toLocaleString()} pts
                            </span>
                          </div>
                          <div className="text-[11px] text-slate-400 flex items-center justify-between">
                            <span>Top: {c.primary_crime_type}</span>
                            <span>{c.percentage_of_total}% share</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-slate-800 text-[11px] text-slate-400">
                    * Click any cluster card to filter map bounding envelope.
                  </div>
                </div>
              </div>

              {/* Crime Type & Area Distribution Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-sm font-semibold text-slate-200 mb-1">
                    Top Primary Crime Classifications
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Observed historical incident counts across study area
                  </p>
                  <div className="space-y-3">
                    {Object.entries(eda.categories.top_10)
                      .slice(0, 6)
                      .map(([name, count]: any) => {
                        const pct = (
                          (count / kpis.total_records_analyzed) *
                          100
                        ).toFixed(1);
                        return (
                          <div key={name}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-slate-300">{name}</span>
                              <span className="text-slate-400 font-mono">
                                {count.toLocaleString()} ({pct}%)
                              </span>
                            </div>
                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-indigo-500 h-full rounded-full"
                                style={{ width: `${Math.min(100, parseFloat(pct) * 3)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                  <h3 className="text-sm font-semibold text-slate-200 mb-1">
                    Top Chicago Community Areas by Volume
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Area codes corresponding to official city community areas
                  </p>
                  <div className="space-y-3">
                    {Object.entries(eda.community_areas.top_10)
                      .slice(0, 6)
                      .map(([area, count]: any) => {
                        const areaNames: Record<string, string> = {
                          "8.0": "Area 8 (Near North Side)",
                          "25.0": "Area 25 (Austin)",
                          "28.0": "Area 28 (Near West Side)",
                          "43.0": "Area 43 (South Shore)",
                          "32.0": "Area 32 (The Loop / Downtown)",
                          "24.0": "Area 24 (West Town)",
                        };
                        const label = areaNames[area] || `Community Area ${area}`;
                        const pct = (
                          (count / kpis.total_records_analyzed) *
                          100
                        ).toFixed(1);
                        return (
                          <div key={area}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-slate-300">{label}</span>
                              <span className="text-slate-400 font-mono">
                                {count.toLocaleString()} ({pct}%)
                              </span>
                            </div>
                            <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-emerald-500 h-full rounded-full"
                                style={{ width: `${Math.min(100, parseFloat(pct) * 15)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: HISTORICAL CRIME MAP */}
          {activeTab === "map" && (
            <div className="space-y-4">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <MapPin className="w-5 h-5 text-indigo-400" />
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Chicago Spatial Crime Map & DBSCAN Clusters
                    </h3>
                    <p className="text-xs text-slate-400">
                      Visualizing verified incident concentrations and bounding density envelopes
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <select
                    value={clusterFilter ?? ""}
                    onChange={(e) =>
                      setClusterFilter(
                        e.target.value === "" ? null : parseInt(e.target.value)
                      )
                    }
                    className="bg-slate-950 border border-slate-700 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-indigo-500"
                  >
                    <option value="">Show All {clusters.length} Clusters</option>
                    {clusters.map((c: any) => (
                      <option key={c.cluster_id} value={c.cluster_id}>
                        Cluster #{c.cluster_id} (Area {c.dominant_community_area}) - {c.incident_count} pts
                      </option>
                    ))}
                  </select>

                  {clusterFilter !== null && (
                    <button
                      onClick={() => setClusterFilter(null)}
                      className="text-xs text-slate-400 hover:text-slate-200 underline"
                    >
                      Reset Filter
                    </button>
                  )}
                </div>
              </div>

              <div className="h-[650px] relative">
                <CrimeMap
                  clusters={clusters}
                  filterClusterId={clusterFilter}
                  onSelectCluster={setSelectedCluster}
                />
              </div>

              {selectedCluster && (
                <div className="bg-slate-900 border border-indigo-500/40 rounded-xl p-4 flex items-center justify-between text-xs">
                  <div>
                    <strong className="text-indigo-400 text-sm">
                      Selected: Cluster #{selectedCluster.cluster_id}
                    </strong>
                    <div className="text-slate-300 mt-1">
                      Centroid: {selectedCluster.centroid_latitude.toFixed(4)}°N,{" "}
                      {selectedCluster.centroid_longitude.toFixed(4)}°W | Incidents:{" "}
                      {selectedCluster.incident_count.toLocaleString()} (
                      {selectedCluster.percentage_of_total}%) | Dominant Type:{" "}
                      {selectedCluster.primary_crime_type}
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedCluster(null)}
                    className="px-2.5 py-1 bg-slate-800 text-slate-300 rounded hover:bg-slate-700"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: HOTSPOT ANALYSIS */}
          {activeTab === "hotspots" && (
            <div className="space-y-6">
              {/* Algorithm & Parameter Banner */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                <div className="flex items-center gap-3 mb-3">
                  <Flame className="w-5 h-5 text-rose-500" />
                  <h3 className="text-sm font-semibold text-slate-100">
                    DBSCAN Hyperparameter Configuration & Sensitivity Matrix
                  </h3>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
                  DBSCAN identifies clusters where geographic density exceeds a critical threshold without
                  enforcing spherical geometry. Distance is computed using the <strong>Haversine</strong> spherical
                  formula on radians (&epsilon; = 500m &rArr; &epsilon;_rad &approx; 7.85 &times; 10&minus;5).
                  Points failing to find at least {params.min_samples} neighbors are labeled as <strong>noise (-1)</strong>.
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Neighborhood Radius (&epsilon;)</span>
                    <strong className="text-rose-400 font-mono text-sm">{params.eps_meters} meters</strong>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Min Samples (min_samples)</span>
                    <strong className="text-rose-400 font-mono text-sm">{params.min_samples} incidents</strong>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Earth Radius Model</span>
                    <strong className="text-slate-200 font-mono text-sm">6,371.0088 km</strong>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-slate-400 block mb-0.5">Metric</span>
                    <strong className="text-slate-200 font-mono text-sm">Haversine (Radians)</strong>
                  </div>
                </div>
              </div>

              {/* Parameter Experiment Grid Table */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-200 mb-1">
                  Hyperparameter Sensitivity Experiments (Tested Grid)
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Evaluating trade-offs between neighborhood radius (&epsilon;) and density threshold (min_samples)
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400">
                        <th className="py-2.5 px-3">Epsilon (meters)</th>
                        <th className="py-2.5 px-3">Min Samples</th>
                        <th className="py-2.5 px-3">Detected Clusters</th>
                        <th className="py-2.5 px-3">Clustered Incidents</th>
                        <th className="py-2.5 px-3">Clustered %</th>
                        <th className="py-2.5 px-3">Noise %</th>
                        <th className="py-2.5 px-3">Mean Cluster Size</th>
                        <th className="py-2.5 px-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {experiments.slice(0, 15).map((exp: any, i: number) => {
                        const isSelected =
                          exp.eps_meters === params.eps_meters &&
                          exp.min_samples === params.min_samples;
                        return (
                          <tr
                            key={i}
                            className={`hover:bg-slate-800/40 ${
                              isSelected ? "bg-indigo-950/40 text-indigo-300 font-semibold" : ""
                            }`}
                          >
                            <td className="py-2 px-3">{exp.eps_meters}m</td>
                            <td className="py-2 px-3">{exp.min_samples}</td>
                            <td className="py-2 px-3">{exp.num_clusters}</td>
                            <td className="py-2 px-3">{exp.clustered_points.toLocaleString()}</td>
                            <td className="py-2 px-3">{exp.clustered_percentage}%</td>
                            <td className="py-2 px-3">{exp.noise_percentage}%</td>
                            <td className="py-2 px-3">{exp.mean_cluster_size}</td>
                            <td className="py-2 px-3">
                              {isSelected ? (
                                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px]">
                                  SELECTED
                                </span>
                              ) : (
                                <span className="text-slate-500 text-[10px]">Evaluation</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Cluster Summary Table */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-200 mb-1">
                  Detected Hotspot Clusters Summary Table
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Representative centroids, coordinate boundaries, and dominant legal classifications
                </p>
                <div className="overflow-x-auto max-h-[400px]">
                  <table className="w-full text-left text-xs">
                    <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400">
                      <tr>
                        <th className="py-2.5 px-3">Cluster ID</th>
                        <th className="py-2.5 px-3">Incident Count</th>
                        <th className="py-2.5 px-3">Share %</th>
                        <th className="py-2.5 px-3">Centroid (Lat, Lon)</th>
                        <th className="py-2.5 px-3">Dominant Crime</th>
                        <th className="py-2.5 px-3">Community Area</th>
                        <th className="py-2.5 px-3">Bounding Box Extent</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                      {clusters.map((c: any) => (
                        <tr key={c.cluster_id} className="hover:bg-slate-800/40">
                          <td className="py-2 px-3 font-bold text-rose-400">
                            #{c.cluster_id}
                          </td>
                          <td className="py-2 px-3">{c.incident_count.toLocaleString()}</td>
                          <td className="py-2 px-3">{c.percentage_of_total}%</td>
                          <td className="py-2 px-3">
                            {c.centroid_latitude.toFixed(4)}°N, {c.centroid_longitude.toFixed(4)}°W
                          </td>
                          <td className="py-2 px-3 font-sans">{c.primary_crime_type}</td>
                          <td className="py-2 px-3">Area {c.dominant_community_area}</td>
                          <td className="py-2 px-3 text-[11px] text-slate-400">
                            [{c.min_latitude.toFixed(3)}, {c.min_longitude.toFixed(3)}] to [
                            {c.max_latitude.toFixed(3)}, {c.max_longitude.toFixed(3)}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: ANALYTICS */}
          {activeTab === "analytics" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                    Spatial Span & Bounding Box
                  </h3>
                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Min Latitude:</span>
                      <span>{eda.geographic_extent.min_latitude.toFixed(4)}°N</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Max Latitude:</span>
                      <span>{eda.geographic_extent.max_latitude.toFixed(4)}°N</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Min Longitude:</span>
                      <span>{eda.geographic_extent.min_longitude.toFixed(4)}°W</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Max Longitude:</span>
                      <span>{eda.geographic_extent.max_longitude.toFixed(4)}°W</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-400 font-sans">Mean Center:</span>
                      <span className="text-indigo-400">
                        {eda.geographic_extent.mean_latitude.toFixed(4)}°N,{" "}
                        {eda.geographic_extent.mean_longitude.toFixed(4)}°W
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                    Temporal Span & Coverage
                  </h3>
                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Date Range:</span>
                      <span>2020 through 2025</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Diurnal Trough:</span>
                      <span>5:00 AM (Lowest volume)</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Diurnal Peak:</span>
                      <span>12:00 PM – 10:00 PM</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-400 font-sans">Seasonality:</span>
                      <span className="text-amber-400">Summer Peak (+30% vs Feb)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                    Cluster Concentration Metrics
                  </h3>
                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Largest Hotspot:</span>
                      <span className="text-rose-400">{kpis.largest_cluster_size.toLocaleString()} incidents</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Smallest Hotspot:</span>
                      <span>{kpis.smallest_cluster_size} incidents</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800">
                      <span className="text-slate-400 font-sans">Median Size:</span>
                      <span>{kpis.median_cluster_size} incidents</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-400 font-sans">Noise Fraction:</span>
                      <span>{kpis.noise_percentage}% of data</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visualizations Gallery */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-200 mb-1">
                  Generated Analytical Visualizations
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Visualizations exported to outputs/figures/ from actual Chicago crime data
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                    <h4 className="font-semibold text-slate-200 mb-1">
                      Visualization A: Historical Crime Incident Scatter
                    </h4>
                    <p className="text-slate-400 text-[11px] mb-2">
                      Points represent individual recorded offenses. Demonstrates heavy spatial clustering in the Downtown/Loop corridor and Western districts.
                    </p>
                    <div className="text-indigo-400 font-mono text-[10px]">
                      File: outputs/figures/vis_a_historical_locations.png
                    </div>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                    <h4 className="font-semibold text-slate-200 mb-1">
                      Visualization B: Spatial Density Heatmap (2D Hexbin)
                    </h4>
                    <p className="text-slate-400 text-[11px] mb-2">
                      Log-scaled incident frequency per spatial bin, highlighting chronic density corridors across the urban core.
                    </p>
                    <div className="text-indigo-400 font-mono text-[10px]">
                      File: outputs/figures/vis_b_crime_density_heatmap.png
                    </div>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                    <h4 className="font-semibold text-slate-200 mb-1">
                      Visualization C: Primary Crime Categories
                    </h4>
                    <p className="text-slate-400 text-[11px] mb-2">
                      THEFT (21.8%) and BATTERY (18.0%) comprise nearly 40% of all incidents across Chicago.
                    </p>
                    <div className="text-indigo-400 font-mono text-[10px]">
                      File: outputs/figures/vis_c_crime_category_distribution.png
                    </div>
                  </div>

                  <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                    <h4 className="font-semibold text-slate-200 mb-1">
                      Visualization D: Community Area Volume
                    </h4>
                    <p className="text-slate-400 text-[11px] mb-2">
                      Austin (#25), Near North Side (#8), Near West Side (#28), and Loop (#32) register highest volumes.
                    </p>
                    <div className="text-indigo-400 font-mono text-[10px]">
                      File: outputs/figures/vis_d_area_level_distribution.png
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: RISK PREDICTION */}
          {activeTab === "prediction" && (
            <div className="max-w-4xl mx-auto space-y-6">
              {/* Integration Status Notice */}
              <div className="bg-amber-950/40 border border-amber-600/50 rounded-xl p-5 shadow-lg">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div className="text-xs">
                    <h3 className="font-semibold text-amber-200 text-sm mb-1">
                      Supervised ML Prediction Pipeline: Integration Pending
                    </h3>
                    <p className="text-amber-300/90 leading-relaxed">
                      Lina (Member 1) has completed data preprocessing and core model training. Model validation and
                      final testing on the 2025 held-out split are currently pending on her side.
                    </p>
                    <p className="text-amber-400/90 mt-2 font-medium">
                      The form below calculates <strong>provisional spatial exposure</strong> (Haversine distance d_cluster
                      to the nearest historical DBSCAN hotspot). It does <em>not</em> claim to generate an automated future crime forecast.
                    </p>
                  </div>
                </div>
              </div>

              {/* Prediction Input Form */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-200 mb-1">
                  Query Geographic Spatial Risk
                </h3>
                <p className="text-xs text-slate-400 mb-5">
                  Select a test location or input custom coordinates to evaluate proximity to historical hotspot centroids.
                </p>

                {/* Location Presets */}
                <div className="mb-5">
                  <span className="text-xs text-slate-400 block mb-2 font-medium">
                    Quick Geographic Presets:
                  </span>
                  <div className="flex flex-wrap gap-2 text-xs">
                    {[
                      { name: "Downtown / Chicago Loop", lat: 41.8781, lon: -87.6298 },
                      { name: "Near North Side (Area 8)", lat: 41.8996, lon: -87.6334 },
                      { name: "Austin (Area 25)", lat: 41.8900, lon: -87.7600 },
                      { name: "South Shore (Area 43)", lat: 41.7600, lon: -87.5700 },
                      { name: "Far South / Low Density", lat: 41.6600, lon: -87.5600 },
                    ].map((preset) => (
                      <button
                        key={preset.name}
                        type="button"
                        onClick={() => setPresetLocation(preset.name, preset.lat, preset.lon)}
                        className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700/60 transition"
                      >
                        {preset.name}
                      </button>
                    ))}
                  </div>
                </div>

                <form onSubmit={handlePredict} className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Latitude (°N)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={predLat}
                        onChange={(e) => setPredLat(parseFloat(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 font-mono focus:border-indigo-500 focus:outline-none"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Longitude (°W)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={predLon}
                        onChange={(e) => setPredLon(parseFloat(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 font-mono focus:border-indigo-500 focus:outline-none"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Target Date
                      </label>
                      <input
                        type="date"
                        value={predDate}
                        onChange={(e) => setPredDate(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 font-mono focus:border-indigo-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Target Time
                      </label>
                      <input
                        type="time"
                        value={predTime}
                        onChange={(e) => setPredTime(e.target.value)}
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 font-mono focus:border-indigo-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={isPredicting}
                    className="w-full mt-4 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow-lg shadow-indigo-900/30 transition flex items-center justify-center gap-2"
                  >
                    {isPredicting ? "Calculating Proximity..." : "Evaluate Spatial Risk Exposure"}
                  </button>
                </form>
              </div>

              {/* Prediction Result Display */}
              {predResult && (
                <div className="bg-slate-900 border border-indigo-500/50 rounded-xl p-6 shadow-2xl space-y-4">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-indigo-400" />
                      <h4 className="text-sm font-bold text-slate-100">
                        Provisional Spatial Analysis Result
                      </h4>
                    </div>
                    <span className="px-2.5 py-1 rounded text-[11px] bg-indigo-950 text-indigo-300 border border-indigo-800 font-mono font-medium">
                      Status: {predResult.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block mb-1">Spatial Risk Tier</span>
                      <strong className="text-rose-400 font-semibold text-sm">
                        {predResult.provisional_spatial_risk_tier}
                      </strong>
                    </div>

                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block mb-1">Distance to Nearest Hotspot</span>
                      <strong className="text-slate-200 font-mono text-sm">
                        {predResult.distance_to_hotspot_km} km
                      </strong>
                    </div>

                    <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block mb-1">Nearest Cluster ID</span>
                      <strong className="text-slate-200 font-mono text-sm">
                        Cluster #{predResult.nearest_hotspot.cluster_id} (
                        {predResult.nearest_hotspot.incident_count} pts)
                      </strong>
                    </div>
                  </div>

                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800 text-[11px] text-slate-400 leading-relaxed">
                    <strong className="text-slate-300">Methodological Note: </strong>
                    {predResult.disclaimer}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 6: METHODOLOGY */}
          {activeTab === "methodology" && (
            <div className="max-w-4xl mx-auto space-y-6 text-xs text-slate-300 leading-relaxed">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
                <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
                  <BookOpen className="w-5 h-5 text-indigo-400" />
                  <h3 className="text-sm font-bold text-slate-100">
                    Spatial Analysis Methodology & Algorithmic Rationale
                  </h3>
                </div>

                <div className="space-y-3">
                  <h4 className="font-semibold text-indigo-400 text-xs uppercase tracking-wider">
                    1. The Spherical Haversine Distance Metric
                  </h4>
                  <p>
                    Latitude and Longitude represent angular coordinates on an oblate spheroid. At Chicago&apos;s
                    latitude (&approx; 41.88&deg;N), one degree of latitude measures approximately 111.0 km,
                    whereas one degree of longitude measures only &approx; 111.0 &times; cos(41.88&deg;) &approx; 82.7 km.
                    Treating geographic degrees as planar Euclidean Cartesian coordinates introduces an asymmetric
                    distortion error of over <strong>34%</strong>.
                  </p>
                  <p>
                    We circumvent this distortion by transforming angular coordinates to radians and calculating
                    spherical great-circle distances via the <strong>Haversine formula</strong>:
                  </p>
                  <div className="p-3 bg-slate-950 rounded-lg font-mono text-[11px] text-indigo-300 border border-slate-800 overflow-x-auto">
                    d = 2R &middot; arcsin(&radic;(sin&sup2;(&Delta;&phi;/2) + cos(&phi;&sub1;)cos(&phi;&sub2;)sin&sup2;(&Delta;&lambda;/2)))
                  </div>
                  <p>
                    Where R = 6,371.0088 km represents the standard IUGG volumetric mean Earth radius.
                    The neighborhood radius &epsilon; is passed to scikit-learn&apos;s DBSCAN as an angular radian:
                    &epsilon;_rad = &epsilon;_km / R.
                  </p>
                </div>

                <div className="space-y-3 pt-3 border-t border-slate-800">
                  <h4 className="font-semibold text-indigo-400 text-xs uppercase tracking-wider">
                    2. DBSCAN Clustering & Parameter Selection
                  </h4>
                  <p>
                    Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is an unsupervised
                    learning algorithm that discovers clusters of arbitrary non-convex geometric shapes based on local
                    spatial density. We systematically evaluated 25 hyperparameter combinations across
                    &epsilon; &isin; [200m, 1000m] and min_samples &isin; [15, 100].
                  </p>
                  <p>
                    <strong>Selected Configuration:</strong> &epsilon; = 500 meters, min_samples = 50.
                    This balances urban street-block granularity without causing excessive spatial fragmentation
                    or collapsing the entire municipality into an uninformative mega-cluster.
                  </p>
                </div>

                <div className="space-y-3 pt-3 border-t border-slate-800">
                  <h4 className="font-semibold text-indigo-400 text-xs uppercase tracking-wider">
                    3. Distinction: Historical Hotspots vs. Future Risk Prediction
                  </h4>
                  <p>
                    A DBSCAN cluster is purely a <strong>retrospective summary</strong> of where recorded incidents
                    historically accumulated. Hotspot clustering alone does not predict when, where, or why future offenses
                    will occur. Prospective forecasting requires supervised regression and classification models
                    integrating calendar cycles, weather, land use, and temporal lags (Member 1&apos;s contribution).
                  </p>
                </div>

                <div className="space-y-3 pt-3 border-t border-slate-800">
                  <h4 className="font-semibold text-rose-400 text-xs uppercase tracking-wider">
                    4. Security, Fairness & Ethical Safeguards
                  </h4>
                  <ul className="list-disc list-inside space-y-1.5 text-slate-300">
                    <li>
                      <strong>No Sensitive Personal Attributes:</strong> Race, ethnicity, religion, and personal
                      identifiers are strictly excluded from the analytical pipeline.
                    </li>
                    <li>
                      <strong>Reporting & Enforcement Bias Awareness:</strong> Police records reflect reported
                      incidents and active enforcement patterns, which can over-represent proactive policing zones.
                      Outputs must never be interpreted as inherent community criminality.
                    </li>
                    <li>
                      <strong>Strict Data Leakage Prevention:</strong> Post-investigation fields (`Arrest`, `IUCR`,
                      `FBI Code`, `Description`) are quarantined and prohibited from prospective feature pipelines.
                    </li>
                    <li>
                      <strong>Decision Support Context:</strong> System outputs are designed strictly for resource
                      allocation and public safety support, not automated punitive actions.
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
