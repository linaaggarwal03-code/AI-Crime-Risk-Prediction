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
  History,
  Filter,
  TrendingUp,
  Clock,
  Activity,
  RefreshCw,
  Search,
  ExternalLink,
  Info,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

import analysisData from "../data/analysisData.json";

// Dynamically import CrimeMap — client-only with smooth fallback
const CrimeMap = dynamic(() => import("../components/CrimeMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] flex flex-col items-center justify-center gap-3 text-sky-400 bg-[#09111e] rounded-2xl border border-sky-500/20">
      <RefreshCw className="w-6 h-6 animate-spin" />
      <span className="text-xs font-medium tracking-wide text-slate-300">
        Mounting Leaflet GIS Engine…
      </span>
    </div>
  ),
});

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "map" | "hotspots" | "analytics" | "prediction" | "methodology"
  >("dashboard");

  // Map and filter states
  const [selectedCluster, setSelectedCluster] = useState<any>(null);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [timeWindowFilter, setTimeWindowFilter] = useState<string>("");
  const [mapIncidents, setMapIncidents] = useState<any[]>([]);
  const [loadingMapIncidents, setLoadingMapIncidents] = useState<boolean>(false);

  // Hotspot sub-tab
  const [hotspotSubTab, setHotspotSubTab] = useState<"clusters" | "experiments" | "params">("clusters");
  const [clusterSearchQuery, setClusterSearchQuery] = useState<string>("");

  // Risk prediction form
  const [predLat, setPredLat] = useState<number>(41.8781);
  const [predLon, setPredLon] = useState<number>(-87.6298);
  const [predDate, setPredDate] = useState<string>("2026-11-25");
  const [predTime, setPredTime] = useState<string>("14:30");
  const [predResult, setPredResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [predHistory, setPredHistory] = useState<any[]>([]);
  const [historyBackend, setHistoryBackend] = useState<string>("SQLite");

  const { summary, clusters, experiments, eda, spatial_temporal } = analysisData as any;
  const kpis = summary.results_summary;
  const params = summary.selected_hyperparameters;

  // Fetch prediction query history
  const fetchPredictionHistory = () => {
    fetch("http://localhost:8000/api/prediction-history?limit=8")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.history) {
          setPredHistory(d.history);
          setHistoryBackend(d.database_backend || "SQLite");
        }
      })
      .catch(() => {});
  };

  // Initial check on backend health
  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((r) => {
        if (r.ok) {
          setBackendOnline(true);
          fetchPredictionHistory();
        }
      })
      .catch(() => setBackendOnline(false));
  }, []);

  // Fetch filtered incidents when map filters change
  useEffect(() => {
    if (activeTab !== "map" || !backendOnline) return;
    setLoadingMapIncidents(true);
    const qs = new URLSearchParams({ limit: "600" });
    if (clusterFilter !== null) qs.set("cluster_id", String(clusterFilter));
    if (categoryFilter) qs.set("category", categoryFilter);
    if (timeWindowFilter) qs.set("time_window", timeWindowFilter);

    fetch(`http://localhost:8000/api/historical-incidents?${qs.toString()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.incidents) setMapIncidents(d.incidents);
      })
      .catch(() => setMapIncidents([]))
      .finally(() => setLoadingMapIncidents(false));
  }, [activeTab, backendOnline, clusterFilter, categoryFilter, timeWindowFilter]);

  // Handle spatial proximity evaluation
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);

    try {
      const res = await fetch("http://localhost:8000/api/predict-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: predLat,
          longitude: predLon,
          date: predDate,
          time: predTime,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setPredResult(data);
        fetchPredictionHistory();
        setIsPredicting(false);
        return;
      }
    } catch {
      // Local fallback calculation if backend is disconnected
    }

    let nearestC = clusters[0];
    let minD = 999999;
    clusters.forEach((c: any) => {
      const dLat = ((c.centroid_latitude - predLat) * Math.PI) / 180;
      const dLon = ((c.centroid_longitude - predLon) * Math.PI) / 180;
      const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos((predLat * Math.PI) / 180) *
          Math.cos((c.centroid_latitude * Math.PI) / 180) *
          Math.sin(dLon / 2) ** 2;
      const dist = 6371.0088 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      if (dist < minD) {
        minD = dist;
        nearestC = c;
      }
    });

    let tier = "Low / Baseline (Dispersed Location > 1.5km from Hotspots)";
    if (minD <= 0.5) tier = "High (Proximity to Historic Hotspot < 500m)";
    else if (minD <= 1.5) tier = "Moderate (Proximity to Historic Hotspot 0.5km - 1.5km)";

    setPredResult({
      status: "PROVISIONAL_SPATIAL_PROXIMITY_ONLY",
      query_coordinates: { latitude: predLat, longitude: predLon },
      nearest_hotspot: nearestC,
      distance_to_hotspot_km: parseFloat(minD.toFixed(3)),
      provisional_spatial_risk_tier: tier,
      model_integration_status: "PENDING_LINA_SUPERVISED_MODEL_EXPORT",
      disclaimer:
        "Notice: This is a geometric proximity calculation relative to historical DBSCAN clusters. Final supervised classification and count forecasting are pending Member 1's (Lina) model testing. This does NOT represent a confirmed future crime forecast.",
    });
    setIsPredicting(false);
  };

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "map", label: "Crime Map", icon: MapPin },
    { id: "hotspots", label: "Hotspot Analysis", icon: Flame },
    { id: "analytics", label: "Analytics", icon: BarChart3 },
    { id: "prediction", label: "Risk Prediction", icon: ShieldAlert },
    { id: "methodology", label: "Methodology", icon: BookOpen },
  ];

  // Filter clusters for search query in Hotspots tab
  const filteredClusters = clusters.filter((c: any) => {
    if (!clusterSearchQuery) return true;
    const q = clusterSearchQuery.toLowerCase();
    return (
      c.cluster_id.toString().includes(q) ||
      c.primary_crime_type.toLowerCase().includes(q) ||
      c.dominant_community_area.toString().includes(q)
    );
  });

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080e18] text-[#f8fafc] font-sans">
      {/* ─── SIDEBAR ─────────────────────────────────────────── */}
      <aside className="w-64 flex-shrink-0 flex flex-col justify-between p-5 bg-[#0c1524] border-r border-[#1a2942] z-20">
        <div>
          {/* Logo & Project Tag */}
          <div className="flex items-center gap-3 pb-6 mb-6 border-b border-[#1a2942]">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-sky-700 flex items-center justify-center shadow-lg shadow-sky-500/20">
              <ShieldAlert className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-sm tracking-tight text-white flex items-center gap-1.5">
                Urban Crime AI
              </h1>
              <span className="text-[11px] text-sky-400 font-medium">
                Spatial Intelligence
              </span>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="space-y-1.5">
            {navItems.map(({ id, label, icon: Icon }) => {
              const active = activeTab === id;
              return (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as any)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                    active
                      ? "bg-sky-500/15 text-sky-300 border border-sky-500/30 shadow-sm shadow-sky-500/10"
                      : "text-slate-400 hover:text-slate-200 hover:bg-[#121f35]"
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 transition-colors ${
                      active ? "text-sky-400" : "text-slate-500"
                    }`}
                  />
                  <span>{label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer Meta / Project Badge */}
        <div className="pt-4 border-t border-[#1a2942] space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Backend Engine</span>
            <span
              className={`flex items-center gap-1.5 font-medium ${
                backendOnline ? "text-emerald-400" : "text-amber-400"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  backendOnline ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
                }`}
              />
              {backendOnline ? ":8000 Live" : "Standalone"}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-[#09111e] border border-[#1b2b45] text-[11px] space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-400">Author:</span>
              <span className="font-semibold text-slate-200">Aditi Verma</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Phase:</span>
              <span className="text-sky-400 font-medium">Milestone 3 (Completed)</span>
            </div>
          </div>
        </div>
      </aside>

      {/* ─── MAIN CONTENT AREA ───────────────────────────────── */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Header Bar */}
        <header className="h-16 flex-shrink-0 px-8 flex items-center justify-between border-b border-[#1a2942] bg-[#0c1524]/70 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-bold tracking-tight text-white capitalize">
              {activeTab === "dashboard" && "Executive Spatial Intelligence Dashboard"}
              {activeTab === "map" && "Geographic Distribution & Density Explorer"}
              {activeTab === "hotspots" && "DBSCAN Crime Hotspot Analysis"}
              {activeTab === "analytics" && "Urban Crime Statistical & Temporal Analytics"}
              {activeTab === "prediction" && "Provisional Spatial Risk Evaluation"}
              {activeTab === "methodology" && "Methodology & Ethical Safeguards"}
            </h2>
            <span className="pill-sky hidden sm:inline-flex">
              Chicago 2020–2025
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="pill-cloudy">
              <Database className="w-3.5 h-3.5 text-sky-400" />
              <span>{kpis.total_records_analyzed.toLocaleString()} records</span>
            </span>
            <span className="pill-red">
              <Flame className="w-3.5 h-3.5 text-red-400" />
              <span>{kpis.num_clusters_identified} clusters</span>
            </span>
          </div>
        </header>

        {/* Content Pane */}
        <div className="flex-1 overflow-y-auto p-8">
          {/* ════════════════ TAB 1: DASHBOARD ════════════════ */}
          {activeTab === "dashboard" && (
            <div className="max-w-7xl mx-auto space-y-7">
              {/* 4 Spacious KPI Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                <div className="glass-panel p-5 space-y-3">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Validated Incidents
                    </span>
                    <span className="p-2 rounded-lg bg-sky-500/10 text-sky-400">
                      <Database className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-white tracking-tight">
                    {kpis.total_records_analyzed.toLocaleString()}
                  </div>
                  <div className="text-xs text-emerald-400 flex items-center gap-1.5 font-medium">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    99.84% Geographic completeness
                  </div>
                </div>

                <div className="glass-panel p-5 space-y-3">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Detected Hotspots
                    </span>
                    <span className="p-2 rounded-lg bg-red-500/10 text-red-400">
                      <Flame className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-white tracking-tight">
                    {kpis.num_clusters_identified}
                  </div>
                  <div className="text-xs text-slate-400">
                    Radius &epsilon;={params.eps_meters}m &middot; min_pts={params.min_samples}
                  </div>
                </div>

                <div className="glass-panel p-5 space-y-3">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Clustered Density
                    </span>
                    <span className="p-2 rounded-lg bg-sky-500/10 text-sky-400">
                      <Layers className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-sky-400 tracking-tight">
                    {kpis.clustered_percentage}%
                  </div>
                  <div className="text-xs text-slate-400">
                    {kpis.clustered_incidents.toLocaleString()} crimes inside hotspots
                  </div>
                </div>

                <div className="glass-panel p-5 space-y-3">
                  <div className="flex items-center justify-between text-slate-400">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                      Dispersed Noise
                    </span>
                    <span className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
                      <Compass className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-amber-400 tracking-tight">
                    {kpis.noise_percentage}%
                  </div>
                  <div className="text-xs text-slate-400">
                    {kpis.noise_incidents.toLocaleString()} isolated baseline points
                  </div>
                </div>
              </div>

              {/* Main Featured Row: Map Preview + Top 5 Clusters */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Map Preview Card */}
                <div className="lg:col-span-2 glass-panel p-6 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-base font-bold text-white">
                          Spatial Hotspot Distribution
                        </h3>
                        <p className="text-xs text-slate-400 mt-0.5">
                          High-density clusters detected via DBSCAN Haversine metric
                        </p>
                      </div>
                      <button
                        onClick={() => setActiveTab("map")}
                        className="px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold flex items-center gap-1.5 transition shadow-sm shadow-sky-600/30"
                      >
                        Full Interactive Map <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="rounded-xl overflow-hidden border border-[#1b2b45] h-[400px]">
                      <CrimeMap
                        clusters={clusters}
                        filterClusterId={clusterFilter}
                        onSelectCluster={setSelectedCluster}
                      />
                    </div>
                  </div>

                  <div className="mt-3 pt-3 border-t border-[#1b2b45] flex items-center justify-between text-xs text-slate-400">
                    <span>* Toggle Dark/Light map views inside the map control panel</span>
                    <span className="text-sky-400 font-mono">Chicago Spatial Envelope: 41.64°N - 42.02°N</span>
                  </div>
                </div>

                {/* Top Hotspots List */}
                <div className="glass-panel p-6 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-base font-bold text-white">
                          Top Hotspots
                        </h3>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Highest historical concentration
                        </p>
                      </div>
                      <span className="pill-red">High Density</span>
                    </div>

                    <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
                      {clusters.slice(0, 5).map((c: any) => (
                        <div
                          key={c.cluster_id}
                          onClick={() => {
                            setClusterFilter(
                              clusterFilter === c.cluster_id ? null : c.cluster_id
                            );
                            setSelectedCluster(c);
                          }}
                          className={`p-3.5 rounded-xl border text-xs cursor-pointer transition-all ${
                            clusterFilter === c.cluster_id
                              ? "bg-sky-500/10 border-sky-400 shadow-sm shadow-sky-500/20"
                              : "bg-[#09111e] border-[#1b2b45] hover:border-sky-500/40"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1.5">
                            <span className="font-bold text-slate-100">
                              Cluster #{c.cluster_id} &middot; Area {c.dominant_community_area}
                            </span>
                            <span className="pill-red font-mono font-bold">
                              {c.incident_count.toLocaleString()} pts
                            </span>
                          </div>
                          <div className="flex items-center justify-between text-slate-400 text-[11px]">
                            <span>Primary: {c.primary_crime_type}</span>
                            <span className="font-mono text-sky-400 font-semibold">{c.percentage_of_total}% share</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-400 pt-3 border-t border-[#1b2b45]">
                    Click any cluster card to isolate it on the map.
                  </p>
                </div>
              </div>

              {/* Bottom 2-Column Summary: Categories & Areas */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Top Crime Classifications
                      </h3>
                      <p className="text-xs text-slate-400">
                        Historical incident count across Chicago
                      </p>
                    </div>
                    <span className="pill-sky">Offense Types</span>
                  </div>

                  <div className="space-y-3 pt-1">
                    {Object.entries(eda.categories.top_10)
                      .slice(0, 5)
                      .map(([name, count]: any) => {
                        const pct = ((count / kpis.total_records_analyzed) * 100).toFixed(1);
                        return (
                          <div key={name} className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="font-medium text-slate-200">{name}</span>
                              <span className="font-mono text-slate-400">
                                {count.toLocaleString()} ({pct}%)
                              </span>
                            </div>
                            <div className="w-full bg-[#09111e] h-2 rounded-full overflow-hidden border border-[#1b2b45]">
                              <div
                                className="bg-sky-500 h-full rounded-full transition-all"
                                style={{ width: `${Math.min(100, parseFloat(pct) * 3)}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>

                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Top Community Areas by Volume
                      </h3>
                      <p className="text-xs text-slate-400">
                        Official Chicago neighborhood districts
                      </p>
                    </div>
                    <span className="pill-amber">District Volume</span>
                  </div>

                  <div className="space-y-3 pt-1">
                    {Object.entries(eda.community_areas.top_10)
                      .slice(0, 5)
                      .map(([area, count]: any) => {
                        const areaNames: Record<string, string> = {
                          "8.0": "Near North Side (#8)",
                          "25.0": "Austin (#25)",
                          "28.0": "Near West Side (#28)",
                          "43.0": "South Shore (#43)",
                          "32.0": "The Loop (#32)",
                        };
                        const label = areaNames[area] || `Community Area ${area}`;
                        const pct = ((count / kpis.total_records_analyzed) * 100).toFixed(1);
                        return (
                          <div key={area} className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="font-medium text-slate-200">{label}</span>
                              <span className="font-mono text-slate-400">
                                {count.toLocaleString()} ({pct}%)
                              </span>
                            </div>
                            <div className="w-full bg-[#09111e] h-2 rounded-full overflow-hidden border border-[#1b2b45]">
                              <div
                                className="bg-amber-500 h-full rounded-full transition-all"
                                style={{ width: `${Math.min(100, parseFloat(pct) * 12)}%` }}
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

          {/* ════════════════ TAB 2: CRIME MAP ════════════════ */}
          {activeTab === "map" && (
            <div className="max-w-7xl mx-auto space-y-4 h-full flex flex-col">
              {/* Clean Filter Controls Bar */}
              <div className="glass-panel p-4 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="p-2 rounded-xl bg-sky-500/10 text-sky-400">
                    <MapPin className="w-4 h-4" />
                  </span>
                  <div>
                    <h3 className="text-sm font-bold text-white">
                      Geographic Spatial Studio
                    </h3>
                    <p className="text-xs text-slate-400">
                      Explore 29 DBSCAN clusters, verified crime incidents, and bounding density envelopes
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2.5">
                  <span className="text-xs text-slate-400 flex items-center gap-1 font-medium mr-1">
                    <Filter className="w-3.5 h-3.5" /> Filters:
                  </span>

                  <select
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="app-select"
                  >
                    <option value="">All Crime Categories</option>
                    {Object.keys(eda.categories.top_10).map((cat: string) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>

                  <select
                    value={timeWindowFilter}
                    onChange={(e) => setTimeWindowFilter(e.target.value)}
                    className="app-select"
                  >
                    <option value="">All Time Windows</option>
                    <option value="Morning">Morning (06:00–11:59)</option>
                    <option value="Afternoon">Afternoon (12:00–17:59)</option>
                    <option value="Evening">Evening (18:00–23:59)</option>
                    <option value="Night">Night (00:00–05:59)</option>
                  </select>

                  <select
                    value={clusterFilter ?? ""}
                    onChange={(e) =>
                      setClusterFilter(
                        e.target.value === "" ? null : parseInt(e.target.value)
                      )
                    }
                    className="app-select"
                  >
                    <option value="">All 29 Clusters</option>
                    {clusters.map((c: any) => (
                      <option key={c.cluster_id} value={c.cluster_id}>
                        Cluster #{c.cluster_id} (Area {c.dominant_community_area})
                      </option>
                    ))}
                  </select>

                  {(clusterFilter !== null || categoryFilter || timeWindowFilter) && (
                    <button
                      onClick={() => {
                        setClusterFilter(null);
                        setCategoryFilter("");
                        setTimeWindowFilter("");
                      }}
                      className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-sky-400 font-medium transition"
                    >
                      Reset
                    </button>
                  )}

                  {loadingMapIncidents && (
                    <span className="text-xs text-amber-400 flex items-center gap-1.5 font-medium animate-pulse ml-2">
                      <RefreshCw className="w-3 h-3 animate-spin" /> Fetching incidents…
                    </span>
                  )}
                </div>
              </div>

              {/* Map Canvas */}
              <div className="glass-panel p-2 flex-1 min-h-[580px] rounded-2xl overflow-hidden relative">
                <CrimeMap
                  clusters={clusters}
                  incidents={mapIncidents}
                  filterClusterId={clusterFilter}
                  onSelectCluster={setSelectedCluster}
                />
              </div>

              {/* Selected Cluster Info Card */}
              {selectedCluster && (
                <div className="glass-panel p-4 flex items-center justify-between border-sky-500/40 bg-sky-950/20">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-sky-300">
                        Cluster #{selectedCluster.cluster_id} Selected
                      </span>
                      <span className="pill-sky">Area {selectedCluster.dominant_community_area}</span>
                    </div>
                    <p className="text-xs text-slate-300">
                      Centroid: {selectedCluster.centroid_latitude.toFixed(4)}°N, {selectedCluster.centroid_longitude.toFixed(4)}°W &middot; Incidents: {selectedCluster.incident_count.toLocaleString()} ({selectedCluster.percentage_of_total}%) &middot; Dominant: {selectedCluster.primary_crime_type}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedCluster(null)}
                    className="px-3 py-1.5 bg-[#09111e] hover:bg-[#121f35] text-slate-300 rounded-lg text-xs font-medium border border-[#1b2b45] transition"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ════════════════ TAB 3: HOTSPOT ANALYSIS ════════════════ */}
          {activeTab === "hotspots" && (
            <div className="max-w-7xl mx-auto space-y-6">
              {/* Segmented Sub-Navigation */}
              <div className="flex items-center justify-between border-b border-[#1a2942] pb-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setHotspotSubTab("clusters")}
                    className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "clusters"
                        ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                        : "bg-[#0c1524] text-slate-400 hover:text-white"
                    }`}
                  >
                    Cluster Directory (29 Clusters)
                  </button>
                  <button
                    onClick={() => setHotspotSubTab("experiments")}
                    className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "experiments"
                        ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                        : "bg-[#0c1524] text-slate-400 hover:text-white"
                    }`}
                  >
                    Sensitivity Experiments (25 Tests)
                  </button>
                  <button
                    onClick={() => setHotspotSubTab("params")}
                    className={`px-4 py-2 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "params"
                        ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                        : "bg-[#0c1524] text-slate-400 hover:text-white"
                    }`}
                  >
                    DBSCAN Parameters & Math
                  </button>
                </div>

                {hotspotSubTab === "clusters" && (
                  <div className="relative w-64">
                    <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3" />
                    <input
                      type="text"
                      placeholder="Search cluster, crime, area…"
                      value={clusterSearchQuery}
                      onChange={(e) => setClusterSearchQuery(e.target.value)}
                      className="app-input pl-8 py-1.5 text-xs w-full"
                    />
                  </div>
                )}
              </div>

              {/* Sub-Tab 1: Clusters Catalog */}
              {hotspotSubTab === "clusters" && (
                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Identified Hotspot Clusters ({filteredClusters.length})
                      </h3>
                      <p className="text-xs text-slate-400">
                        Dominant crimes, geographic coordinates, and historical volume
                      </p>
                    </div>
                    <span className="pill-sky">ε=500m &middot; min_samples=50</span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-[#1b2b45]">
                    <table className="pro-table">
                      <thead>
                        <tr>
                          <th>Cluster ID</th>
                          <th>Incident Count</th>
                          <th>Share %</th>
                          <th>Centroid (Lat, Lon)</th>
                          <th>Dominant Crime</th>
                          <th>Community Area</th>
                          <th>Bounding Coordinate Extents</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {filteredClusters.map((c: any) => (
                          <tr key={c.cluster_id}>
                            <td className="font-bold text-red-400">#{c.cluster_id}</td>
                            <td className="font-bold text-white">{c.incident_count.toLocaleString()}</td>
                            <td className="text-sky-400">{c.percentage_of_total}%</td>
                            <td>
                              {c.centroid_latitude.toFixed(4)}°N, {c.centroid_longitude.toFixed(4)}°W
                            </td>
                            <td className="font-sans font-medium text-slate-200">
                              {c.primary_crime_type}
                            </td>
                            <td className="font-sans text-slate-300">
                              Area {c.dominant_community_area}
                            </td>
                            <td className="text-[11px] text-slate-400">
                              [{c.min_latitude.toFixed(3)}, {c.min_longitude.toFixed(3)}] to [{c.max_latitude.toFixed(3)}, {c.max_longitude.toFixed(3)}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Sub-Tab 2: Sensitivity Grid */}
              {hotspotSubTab === "experiments" && (
                <div className="glass-panel p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Hyperparameter Sensitivity Experiments
                      </h3>
                      <p className="text-xs text-slate-400">
                        Tested combinations evaluating radius (&epsilon;) vs density threshold (min_samples)
                      </p>
                    </div>
                    <span className="pill-cloudy">25 Total Runs</span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-[#1b2b45]">
                    <table className="pro-table">
                      <thead>
                        <tr>
                          <th>Epsilon (m)</th>
                          <th>Min Samples</th>
                          <th>Clusters</th>
                          <th>Clustered Pts</th>
                          <th>Clustered %</th>
                          <th>Noise %</th>
                          <th>Mean Cluster Size</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {experiments.slice(0, 16).map((exp: any, i: number) => {
                          const isSelected =
                            exp.eps_meters === params.eps_meters &&
                            exp.min_samples === params.min_samples;
                          return (
                            <tr
                              key={i}
                              className={isSelected ? "bg-sky-500/10 font-bold" : ""}
                            >
                              <td>{exp.eps_meters}m</td>
                              <td>{exp.min_samples}</td>
                              <td className="text-sky-400 font-bold">{exp.num_clusters}</td>
                              <td>{exp.clustered_points.toLocaleString()}</td>
                              <td>{exp.clustered_percentage}%</td>
                              <td className="text-amber-400">{exp.noise_percentage}%</td>
                              <td>{exp.mean_cluster_size}</td>
                              <td>
                                {isSelected ? (
                                  <span className="pill-sky font-bold">SELECTED</span>
                                ) : (
                                  <span className="text-slate-500 text-[11px]">Evaluation</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Sub-Tab 3: Mathematical Parameters */}
              {hotspotSubTab === "params" && (
                <div className="glass-panel p-6 space-y-6">
                  <div>
                    <h3 className="text-base font-bold text-white">
                      DBSCAN Algorithmic Architecture
                    </h3>
                    <p className="text-xs text-slate-400 mt-1">
                      Mathematical foundation of density-based clustering using spherical radians
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-4 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Neighborhood Radius (&epsilon;)</span>
                      <div className="text-lg font-bold text-red-400 font-mono">
                        {params.eps_meters} meters
                      </div>
                      <span className="text-[11px] text-slate-500 block">&epsilon;_rad &approx; 7.85 &times; 10&minus;5</span>
                    </div>

                    <div className="p-4 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Min Samples (min_samples)</span>
                      <div className="text-lg font-bold text-red-400 font-mono">
                        {params.min_samples} incidents
                      </div>
                      <span className="text-[11px] text-slate-500 block">Density threshold</span>
                    </div>

                    <div className="p-4 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Earth Model</span>
                      <div className="text-lg font-bold text-sky-400 font-mono">
                        6,371.0088 km
                      </div>
                      <span className="text-[11px] text-slate-500 block">IUGG Volumetric Mean</span>
                    </div>

                    <div className="p-4 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Distance Metric</span>
                      <div className="text-lg font-bold text-sky-400 font-mono">
                        Haversine
                      </div>
                      <span className="text-[11px] text-slate-500 block">Spherical Radians</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-[#09111e] border border-[#1b2b45] text-xs text-slate-300 leading-relaxed space-y-2">
                    <strong className="text-white block">Why Euclidean Distance Fails in Chicago:</strong>
                    <p>
                      At 41.88°N latitude, 1 degree of latitude measures ~111.0 km, while 1 degree of longitude measures only ~82.7 km. A planar Euclidean distance metric distorts Chicago distances by over <strong>34%</strong>. The Haversine metric corrects this by calculating true great-circle angular distances on the sphere.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ════════════════ TAB 4: ANALYTICS ════════════════ */}
          {activeTab === "analytics" && (
            <div className="max-w-7xl mx-auto space-y-7">
              {/* Summary KPIs Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="glass-panel p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-400">
                    <Compass className="w-4 h-4 text-sky-400" /> Geographic Extent
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-300 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Latitude Span:</span>
                      <span>{eda.geographic_extent.min_latitude.toFixed(3)}°N – {eda.geographic_extent.max_latitude.toFixed(3)}°N</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Longitude Span:</span>
                      <span>{eda.geographic_extent.min_longitude.toFixed(3)}°W – {eda.geographic_extent.max_longitude.toFixed(3)}°W</span>
                    </div>
                    <div className="flex justify-between text-sky-400">
                      <span className="text-slate-400 font-sans">Mean Centroid:</span>
                      <span>{eda.geographic_extent.mean_latitude.toFixed(4)}°N, {eda.geographic_extent.mean_longitude.toFixed(4)}°W</span>
                    </div>
                  </div>
                </div>

                <div className="glass-panel p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-400">
                    <Clock className="w-4 h-4 text-amber-400" /> Temporal Coverage
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-300 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Period:</span>
                      <span>2020 through 2025</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Diurnal Trough:</span>
                      <span className="text-amber-400">5:00 AM (Lowest)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Peak Hours:</span>
                      <span>12:00 PM – 10:00 PM</span>
                    </div>
                  </div>
                </div>

                <div className="glass-panel p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-semibold uppercase text-slate-400">
                    <TrendingUp className="w-4 h-4 text-red-400" /> Cluster Concentration
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-300 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Largest Hotspot:</span>
                      <span className="text-red-400 font-bold">{kpis.largest_cluster_size.toLocaleString()} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Smallest Hotspot:</span>
                      <span>{kpis.smallest_cluster_size} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400 font-sans">Median Size:</span>
                      <span>{kpis.median_cluster_size} incidents</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* 4 Clean Recharts Visualizations */}
              {spatial_temporal && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Diurnal Wave */}
                  <div className="glass-panel p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        24-Hour Diurnal Crime Wave
                      </h3>
                      <p className="text-xs text-slate-400">
                        Hourly incident frequency across Chicago
                      </p>
                    </div>
                    <div className="h-[250px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={spatial_temporal.hourly_distribution}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1b2b45" />
                          <XAxis dataKey="label" stroke="#64748b" fontSize={10} interval={3} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0c1524",
                              borderColor: "#1b2b45",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#f8fafc",
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="count"
                            name="Incidents"
                            stroke="#0ea5e9"
                            fill="#0ea5e9"
                            fillOpacity={0.2}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Daytime vs Nighttime Shifts */}
                  <div className="glass-panel p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Daytime vs. Nighttime Hotspot Shifts
                      </h3>
                      <p className="text-xs text-slate-400">
                        Comparing Daytime (06:00–17:59) vs Nighttime (18:00–05:59)
                      </p>
                    </div>
                    <div className="h-[250px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.cluster_day_night_shifts.slice(0, 8)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1b2b45" />
                          <XAxis dataKey="cluster_label" stroke="#64748b" fontSize={10} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0c1524",
                              borderColor: "#1b2b45",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#f8fafc",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: "11px" }} />
                          <Bar dataKey="daytime_count" name="Daytime" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="nighttime_count" name="Nighttime" fill="#ef4444" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Time Window Breakdown */}
                  <div className="glass-panel p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        6-Hour Activity Windows
                      </h3>
                      <p className="text-xs text-slate-400">
                        Incident volume across 4 standardized operational windows
                      </p>
                    </div>
                    <div className="h-[240px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.time_windows}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1b2b45" />
                          <XAxis dataKey="window" stroke="#64748b" fontSize={10} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0c1524",
                              borderColor: "#1b2b45",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#f8fafc",
                            }}
                          />
                          <Bar dataKey="count" name="Incidents" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Category Hotspot Concentration */}
                  <div className="glass-panel p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Category Hotspot Concentration
                      </h3>
                      <p className="text-xs text-slate-400">
                        Proportion inside clusters vs dispersed noise per offense
                      </p>
                    </div>
                    <div className="h-[240px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.category_hotspot_profiles.slice(0, 7)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1b2b45" />
                          <XAxis dataKey="category" stroke="#64748b" fontSize={9} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0c1524",
                              borderColor: "#1b2b45",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#f8fafc",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: "11px" }} />
                          <Bar dataKey="clustered" stackId="a" name="In Hotspots" fill="#0ea5e9" />
                          <Bar dataKey="noise" stackId="a" name="Dispersed Noise" fill="#334155" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ════════════════ TAB 5: RISK PREDICTION ════════════════ */}
          {activeTab === "prediction" && (
            <div className="max-w-4xl mx-auto space-y-6">
              {/* Transparent ML Status Notice */}
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="text-xs space-y-1">
                  <h4 className="font-bold text-amber-200">
                    Provisional Proximity Calculation Active
                  </h4>
                  <p className="text-amber-300/80 leading-relaxed">
                    Final supervised classification models from Member 1 (Lina) remain pending final artifact serialization. This tool calculates <strong>geometric proximity</strong> to historical DBSCAN clusters and does not fabricate ML predictions.
                  </p>
                </div>
              </div>

              {/* Prediction Form & Presets */}
              <div className="glass-panel p-6 space-y-5">
                <div>
                  <h3 className="text-base font-bold text-white">
                    Evaluate Spatial Proximity Exposure
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Select a Chicago benchmark location or enter custom latitude/longitude coordinates
                  </p>
                </div>

                {/* Quick Presets */}
                <div className="space-y-2">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Quick Geographic Presets:
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { name: "Downtown / Chicago Loop", lat: 41.8781, lon: -87.6298 },
                      { name: "Near North Side (Area 8)", lat: 41.8996, lon: -87.6334 },
                      { name: "Austin (Area 25)", lat: 41.8900, lon: -87.7600 },
                      { name: "South Shore (Area 43)", lat: 41.7600, lon: -87.5700 },
                      { name: "Far South / Low Density", lat: 41.6600, lon: -87.5600 },
                    ].map((p) => (
                      <button
                        key={p.name}
                        type="button"
                        onClick={() => {
                          setPredLat(p.lat);
                          setPredLon(p.lon);
                        }}
                        className="px-3 py-1.5 rounded-lg bg-[#09111e] hover:bg-sky-500/15 hover:text-sky-300 text-xs font-medium border border-[#1b2b45] transition"
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>

                <form onSubmit={handlePredict} className="space-y-4 pt-2">
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
                        className="app-input w-full font-mono"
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
                        className="app-input w-full font-mono"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Evaluation Date
                      </label>
                      <input
                        type="date"
                        value={predDate}
                        onChange={(e) => setPredDate(e.target.value)}
                        className="app-input w-full font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Evaluation Time
                      </label>
                      <input
                        type="time"
                        value={predTime}
                        onChange={(e) => setPredTime(e.target.value)}
                        className="app-input w-full font-mono"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={isPredicting}
                    className="w-full py-3 px-4 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs tracking-wide shadow-lg shadow-sky-600/30 transition flex items-center justify-center gap-2"
                  >
                    {isPredicting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" /> Evaluating Proximity…
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4" /> Calculate Spatial Risk Exposure
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* Assessment Result Card */}
              {predResult && (
                <div className="glass-panel p-6 space-y-4 border-sky-500/40">
                  <div className="flex items-center justify-between pb-3 border-b border-[#1b2b45]">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-sky-400" />
                      <h4 className="font-bold text-white text-sm">
                        Spatial Analysis Exposure Result
                      </h4>
                    </div>
                    <span className="pill-sky font-mono font-medium">
                      Status: {predResult.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Spatial Risk Tier</span>
                      <div className="font-bold text-red-400 text-sm">
                        {predResult.provisional_spatial_risk_tier}
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Distance to Hotspot</span>
                      <div className="font-bold text-white font-mono text-sm">
                        {predResult.distance_to_hotspot_km} km
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-[#09111e] border border-[#1b2b45] space-y-1">
                      <span className="text-xs text-slate-400">Nearest Hotspot Cluster</span>
                      <div className="font-bold text-sky-400 font-mono text-sm">
                        Cluster #{predResult.nearest_hotspot.cluster_id} ({predResult.nearest_hotspot.incident_count} pts)
                      </div>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-[#09111e] text-xs text-slate-400 leading-relaxed border border-[#1b2b45]">
                    <strong className="text-slate-300">Methodological Note: </strong>
                    {predResult.disclaimer}
                  </div>
                </div>
              )}

              {/* Persisted History Table */}
              <div className="glass-panel p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <History className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-bold text-white">
                      Persisted Spatial Query History
                    </h3>
                  </div>
                  <span className="pill-sky">DB: {historyBackend}</span>
                </div>

                {predHistory.length === 0 ? (
                  <p className="text-xs text-slate-400">
                    No spatial queries recorded yet. Submit a coordinate query above to persist it into the database.
                  </p>
                ) : (
                  <div className="overflow-x-auto rounded-xl border border-[#1b2b45]">
                    <table className="pro-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Coordinates</th>
                          <th>Nearest Cluster</th>
                          <th>Distance</th>
                          <th>Risk Tier</th>
                          <th>Target Time</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono text-xs">
                        {predHistory.map((h: any) => (
                          <tr key={h.id}>
                            <td className="text-slate-400">#{h.id}</td>
                            <td>
                              {h.latitude.toFixed(4)}°N, {h.longitude.toFixed(4)}°W
                            </td>
                            <td className="text-red-400">
                              {h.nearest_cluster_id !== null ? `Cluster #${h.nearest_cluster_id}` : "N/A"}
                            </td>
                            <td>{h.distance_to_hotspot_km !== null ? `${h.distance_to_hotspot_km} km` : "N/A"}</td>
                            <td className="font-sans text-slate-200">{h.risk_tier}</td>
                            <td className="text-slate-400">{h.target_timestamp}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ════════════════ TAB 6: METHODOLOGY ════════════════ */}
          {activeTab === "methodology" && (
            <div className="max-w-4xl mx-auto space-y-6">
              <div className="glass-panel p-6 space-y-6">
                <div className="flex items-center gap-3 pb-4 border-b border-[#1b2b45]">
                  <span className="p-2.5 rounded-xl bg-sky-500/10 text-sky-400">
                    <BookOpen className="w-5 h-5" />
                  </span>
                  <div>
                    <h3 className="text-base font-bold text-white">
                      Geospatial Methodology & Ethical Safeguards
                    </h3>
                    <p className="text-xs text-slate-400">
                      Mathematical formulations, hyperparameter selection criteria, and fairness guarantees
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                    1. Spherical Haversine Metric
                  </h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    Treating geographic degrees as Cartesian coordinates in Chicago introduces an error exceeding <strong>34%</strong>. To eliminate distortion, coordinates are converted to radians and distances computed using the great-circle Haversine formula:
                  </p>
                  <div className="p-3.5 rounded-xl bg-[#09111e] border border-[#1b2b45] font-mono text-xs text-sky-300 overflow-x-auto">
                    d = 2R &middot; arcsin(&radic;(sin&sup2;(&Delta;&phi;/2) + cos(&phi;&sub1;)cos(&phi;&sub2;)sin&sup2;(&Delta;&lambda;/2)))
                  </div>
                  <p className="text-xs text-slate-400">
                    Where R = 6,371.0088 km (standard IUGG mean Earth radius).
                  </p>
                </div>

                <div className="space-y-3 pt-4 border-t border-[#1b2b45]">
                  <h4 className="text-xs font-bold text-sky-400 uppercase tracking-wider">
                    2. Retrospective Hotspots vs. Predictive Models
                  </h4>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    DBSCAN clusters reflect <em>where crime historically concentrated</em>. They do not constitute automated predictive forecasts. Full predictive modeling involves supervised regression models (Member 1, Lina) integrating weather, temporal lags, and calendar dynamics.
                  </p>
                </div>

                <div className="space-y-3 pt-4 border-t border-[#1b2b45]">
                  <h4 className="text-xs font-bold text-red-400 uppercase tracking-wider">
                    3. Ethical Safeguards & Fairness Controls
                  </h4>
                  <ul className="space-y-2 text-xs text-slate-300">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span><strong>No Demographic Attributes:</strong> Race, ethnicity, income, and personal identifiers are strictly excluded from all models.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span><strong>Leakage Prevention:</strong> Post-arrest fields (`Arrest`, `IUCR`, `FBI Code`) are quarantined and barred from feature pipelines.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                      <span><strong>Resource Allocation Context:</strong> Predictions serve as decision support for municipal resource allocation, never automated punitive actions.</span>
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
