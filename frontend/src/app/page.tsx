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

// Dynamically import CrimeMap — client-only
const CrimeMap = dynamic(() => import("../components/CrimeMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center text-sm"
      style={{ color: "var(--text-muted)" }}>
      Initialising Spatial Map Engine…
    </div>
  ),
});

/* ─── Shared chart theme ─────────────────────────── */
const CHART_GRID  = "#1e3a52";
const CHART_AXIS  = "#64748b";
const TOOLTIP_STYLE = {
  backgroundColor: "#0c1a2e",
  border: "1px solid #1e3a52",
  borderRadius: "8px",
  fontSize: "12px",
  color: "#f0f9ff",
};

/* ─── Stat card ──────────────────────────────────── */
interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ReactNode;
  accent?: string;
}
function StatCard({ label, value, sub, icon, accent = "var(--sky-400)" }: StatCardProps) {
  return (
    <div className="card flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
          {label}
        </span>
        <span style={{ color: accent }}>{icon}</span>
      </div>
      <div className="stat-num" style={{ color: "var(--text-primary)" }}>{value}</div>
      {sub && <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{sub}</div>}
    </div>
  );
}

/* ─── Section header ─────────────────────────────── */
function SectionHeader({ title, sub }: { title: string; sub?: string }) {
  return (
    <div className="mb-4">
      <h3 className="section-title">{title}</h3>
      {sub && <p className="section-sub">{sub}</p>}
    </div>
  );
}

/* ─── Progress bar row ───────────────────────────── */
function ProgressRow({ label, value, pct, color }: { label: string; value: string; pct: number; color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
        <span className="font-mono" style={{ color: "var(--text-muted)" }}>{value}</span>
      </div>
      <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-base)" }}>
        <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(100, pct)}%`, background: color }} />
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════
   MAIN PAGE COMPONENT
   ═══════════════════════════════════════════════════ */
export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "map" | "hotspots" | "analytics" | "prediction" | "methodology"
  >("dashboard");

  const [selectedCluster, setSelectedCluster] = useState<any>(null);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [timeWindowFilter, setTimeWindowFilter] = useState<string>("");
  const [mapIncidents, setMapIncidents] = useState<any[]>([]);
  const [loadingMapIncidents, setLoadingMapIncidents] = useState<boolean>(false);

  // Prediction
  const [predLat, setPredLat] = useState<number>(41.8781);
  const [predLon, setPredLon] = useState<number>(-87.6298);
  const [predDate, setPredDate] = useState<string>("2026-11-25");
  const [predTime, setPredTime] = useState<string>("14:30");
  const [predResult, setPredResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [predHistory, setPredHistory] = useState<any[]>([]);
  const [historyBackend, setHistoryBackend] = useState<string>("SQLite / Local");

  const { summary, clusters, experiments, eda, spatial_temporal } = analysisData as any;
  const kpis   = summary.results_summary;
  const params  = summary.selected_hyperparameters;

  /* ─── Fetch helpers ──────────────── */
  const fetchPredictionHistory = () => {
    fetch("http://localhost:8000/api/prediction-history?limit=10")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.history) {
          setPredHistory(d.history);
          setHistoryBackend(d.database_backend || "SQLite");
        }
      })
      .catch(() => {});
  };

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((r) => { if (r.ok) { setBackendOnline(true); fetchPredictionHistory(); } })
      .catch(() => setBackendOnline(false));
  }, []);

  useEffect(() => {
    if (activeTab !== "map" || !backendOnline) return;
    setLoadingMapIncidents(true);
    const qs = new URLSearchParams({ limit: "600" });
    if (clusterFilter !== null) qs.set("cluster_id", String(clusterFilter));
    if (categoryFilter)   qs.set("category",    categoryFilter);
    if (timeWindowFilter) qs.set("time_window",  timeWindowFilter);
    fetch(`http://localhost:8000/api/historical-incidents?${qs.toString()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.incidents) setMapIncidents(d.incidents); })
      .catch(() => setMapIncidents([]))
      .finally(() => setLoadingMapIncidents(false));
  }, [activeTab, backendOnline, clusterFilter, categoryFilter, timeWindowFilter]);

  /* ─── Risk prediction ────────────── */
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    try {
      const res = await fetch("http://localhost:8000/api/predict-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ latitude: predLat, longitude: predLon, date: predDate, time: predTime }),
      });
      if (res.ok) {
        const data = await res.json();
        setPredResult(data);
        fetchPredictionHistory();
        setIsPredicting(false);
        return;
      }
    } catch { /* fall through to local calc */ }

    // Client-side fallback
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
      if (dist < minD) { minD = dist; nearestC = c; }
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
        "Notice: This is a geometric proximity calculation relative to historical DBSCAN clusters. Final supervised classification and count forecasting are pending Member 1's (Lina) model testing and artifact serialization. This does NOT represent a confirmed future crime forecast.",
    });
    setIsPredicting(false);
  };

  const setPresetLocation = (_: string, lat: number, lon: number) => {
    setPredLat(lat);
    setPredLon(lon);
  };

  /* ─── Tab definitions ────────────── */
  const NAV = [
    { id: "dashboard",    label: "Dashboard",       icon: LayoutDashboard },
    { id: "map",          label: "Crime Map",        icon: MapPin          },
    { id: "hotspots",     label: "Hotspot Analysis", icon: Flame           },
    { id: "analytics",    label: "Analytics",        icon: BarChart3       },
    { id: "prediction",   label: "Risk Prediction",  icon: ShieldAlert     },
    { id: "methodology",  label: "Methodology",      icon: BookOpen        },
  ];

  const TAB_TITLES: Record<string, string> = {
    dashboard:   "Executive Dashboard",
    map:         "Historical Crime Map",
    hotspots:    "DBSCAN Hotspot Analysis",
    analytics:   "Statistical Analytics",
    prediction:  "Spatial Risk Prediction",
    methodology: "Methodology & Safeguards",
  };

  /* ══════════════════════════════════
     RENDER
     ══════════════════════════════════ */
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg-base)", color: "var(--text-primary)" }}>

      {/* ── SIDEBAR ──────────────────────────────── */}
      <aside className="w-60 flex flex-col justify-between flex-shrink-0 py-5 px-3"
        style={{ background: "var(--bg-panel)", borderRight: "1px solid var(--border)" }}>

        {/* Logo */}
        <div>
          <div className="flex items-center gap-3 px-2 py-2 mb-7">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0"
              style={{ background: "linear-gradient(135deg,#0ea5e9 0%,#0369a1 100%)" }}>
              <ShieldAlert className="w-5 h-5 text-white" />
            </div>
            <div>
              <p className="text-sm font-bold leading-tight" style={{ color: "var(--text-primary)" }}>
                Urban Crime AI
              </p>
              <p className="text-[11px] leading-tight" style={{ color: "var(--text-muted)" }}>
                Hotspot Intelligence
              </p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-1">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`nav-item w-full text-left${activeTab === id ? " active" : ""}`}
              >
                <Icon className="w-4 h-4 flex-shrink-0" />
                <span>{label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Footer info */}
        <div className="px-2 space-y-3">
          {/* Backend status */}
          <div className="flex items-center justify-between text-xs">
            <span style={{ color: "var(--text-muted)" }}>API Status</span>
            <span className="flex items-center gap-1.5 font-semibold"
              style={{ color: backendOnline ? "#34d399" : "var(--amber)" }}>
              <span className={`w-2 h-2 rounded-full${backendOnline ? " animate-pulse" : ""}`}
                style={{ background: backendOnline ? "#34d399" : "var(--amber)" }} />
              {backendOnline ? "Live" : "Offline"}
            </span>
          </div>

          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "0.75rem" }}>
            <div className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>Contributor</div>
            <div className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
              Aditi Verma · Member 2
            </div>
            <div className="text-[11px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              Spatial Analytics & Backend
            </div>
          </div>
          <span className="badge-green">Milestone 3 ✓</span>
        </div>
      </aside>

      {/* ── MAIN ─────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-hidden">

        {/* Top Bar */}
        <header className="flex-shrink-0 h-14 px-7 flex items-center justify-between"
          style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)" }}>
          <div>
            <h2 className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>
              {TAB_TITLES[activeTab]}
            </h2>
            <p className="text-[11px]" style={{ color: "var(--text-muted)" }}>
              Chicago Crime Dataset · 2020–2025
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              <span className="font-mono font-bold" style={{ color: "var(--sky-400)" }}>
                {kpis.total_records_analyzed.toLocaleString()}
              </span>{" "}
              records
            </span>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              <span className="font-mono font-bold" style={{ color: "var(--red)" }}>
                {kpis.num_clusters_identified}
              </span>{" "}
              hotspot clusters
            </span>
          </div>
        </header>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-6">

          {/* ══════════════ TAB 1: DASHBOARD ══════════════ */}
          {activeTab === "dashboard" && (
            <div className="space-y-6">

              {/* KPI Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard
                  label="Valid Incidents"
                  value={kpis.total_records_analyzed.toLocaleString()}
                  sub="99.84% geographic completeness"
                  icon={<Database className="w-4 h-4" />}
                  accent="var(--sky-400)"
                />
                <StatCard
                  label="Hotspot Clusters"
                  value={kpis.num_clusters_identified}
                  sub={`ε=${params.eps_meters}m · min_samples=${params.min_samples}`}
                  icon={<Flame className="w-4 h-4" />}
                  accent="var(--red)"
                />
                <StatCard
                  label="Clustered Points"
                  value={`${kpis.clustered_percentage}%`}
                  sub={`${kpis.clustered_incidents.toLocaleString()} assigned to hotspots`}
                  icon={<Layers className="w-4 h-4" />}
                  accent="var(--sky-500)"
                />
                <StatCard
                  label="Noise / Dispersed"
                  value={`${kpis.noise_percentage}%`}
                  sub={`${kpis.noise_incidents.toLocaleString()} isolated incidents`}
                  icon={<Compass className="w-4 h-4" />}
                  accent="var(--amber)"
                />
              </div>

              {/* Map + Top Clusters */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                {/* Map */}
                <div className="lg:col-span-2 card" style={{ padding: "1.25rem" }}>
                  <div className="flex items-center justify-between mb-4">
                    <SectionHeader
                      title="Chicago Spatial Hotspot Map"
                      sub="29 DBSCAN clusters (ε=500m, Haversine metric)"
                    />
                    <button
                      onClick={() => setActiveTab("map")}
                      className="text-xs flex items-center gap-1 font-medium"
                      style={{ color: "var(--sky-400)" }}
                    >
                      Explore full map <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <div className="rounded-xl overflow-hidden" style={{ height: 380 }}>
                    <CrimeMap clusters={clusters} filterClusterId={clusterFilter} onSelectCluster={setSelectedCluster} />
                  </div>
                </div>

                {/* Top clusters list */}
                <div className="card flex flex-col">
                  <SectionHeader title="Highest-Density Clusters" sub="Ranked by historical incident count" />
                  <div className="space-y-2 overflow-y-auto flex-1" style={{ maxHeight: 400 }}>
                    {clusters.slice(0, 7).map((c: any) => (
                      <div
                        key={c.cluster_id}
                        onClick={() => {
                          setClusterFilter(clusterFilter === c.cluster_id ? null : c.cluster_id);
                          setSelectedCluster(c);
                        }}
                        className="card-hover cursor-pointer rounded-xl p-3 text-xs transition-all"
                        style={{
                          background: clusterFilter === c.cluster_id ? "rgba(14,165,233,0.1)" : "var(--bg-base)",
                          border: `1px solid ${clusterFilter === c.cluster_id ? "var(--sky-500)" : "var(--border-soft)"}`,
                        }}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
                            Cluster #{c.cluster_id} · Area {c.dominant_community_area}
                          </span>
                          <span className="badge-red">{c.incident_count.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between" style={{ color: "var(--text-muted)" }}>
                          <span>{c.primary_crime_type}</span>
                          <span>{c.percentage_of_total}% share</span>
                        </div>
                      </div>
                    ))}
                  </div>
                  <p className="text-[11px] mt-3 pt-3" style={{ color: "var(--text-muted)", borderTop: "1px solid var(--border)" }}>
                    Click a cluster to highlight it on the map.
                  </p>
                </div>
              </div>

              {/* Crime type + area breakdown */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Crime types */}
                <div className="card">
                  <SectionHeader title="Top Crime Classifications" sub="Historical incident counts across study area" />
                  <div className="space-y-3">
                    {Object.entries(eda.categories.top_10).slice(0, 6).map(([name, count]: any) => {
                      const pct = ((count / kpis.total_records_analyzed) * 100).toFixed(1);
                      return (
                        <ProgressRow
                          key={name}
                          label={name}
                          value={`${count.toLocaleString()} (${pct}%)`}
                          pct={parseFloat(pct) * 3}
                          color="var(--sky-500)"
                        />
                      );
                    })}
                  </div>
                </div>

                {/* Community areas */}
                <div className="card">
                  <SectionHeader title="Top Community Areas by Volume" sub="Official Chicago community area codes" />
                  <div className="space-y-3">
                    {Object.entries(eda.community_areas.top_10).slice(0, 6).map(([area, count]: any) => {
                      const areaNames: Record<string, string> = {
                        "8.0": "Near North Side (#8)",
                        "25.0": "Austin (#25)",
                        "28.0": "Near West Side (#28)",
                        "43.0": "South Shore (#43)",
                        "32.0": "The Loop (#32)",
                        "24.0": "West Town (#24)",
                      };
                      const label = areaNames[area] || `Community Area ${area}`;
                      const pct = ((count / kpis.total_records_analyzed) * 100).toFixed(1);
                      return (
                        <ProgressRow
                          key={area}
                          label={label}
                          value={`${count.toLocaleString()} (${pct}%)`}
                          pct={parseFloat(pct) * 15}
                          color="var(--amber)"
                        />
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════ TAB 2: MAP ══════════════ */}
          {activeTab === "map" && (
            <div className="space-y-4">
              {/* Filter bar */}
              <div className="card flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <MapPin className="w-4 h-4" style={{ color: "var(--sky-400)" }} />
                  <div>
                    <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                      Chicago Crime Map & DBSCAN Clusters
                    </p>
                    <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                      Interactive spatial view with incident overlay and filter controls
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 ml-auto">
                  <Filter className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />

                  <select
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="form-select"
                  >
                    <option value="">All Categories</option>
                    {Object.keys(eda.categories.top_10).map((cat: string) => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>

                  <select
                    value={timeWindowFilter}
                    onChange={(e) => setTimeWindowFilter(e.target.value)}
                    className="form-select"
                  >
                    <option value="">All Time Windows</option>
                    <option value="Morning">Morning (06–11)</option>
                    <option value="Afternoon">Afternoon (12–17)</option>
                    <option value="Evening">Evening (18–23)</option>
                    <option value="Night">Night (00–05)</option>
                  </select>

                  <select
                    value={clusterFilter ?? ""}
                    onChange={(e) => setClusterFilter(e.target.value === "" ? null : parseInt(e.target.value))}
                    className="form-select"
                  >
                    <option value="">All {clusters.length} Clusters</option>
                    {clusters.map((c: any) => (
                      <option key={c.cluster_id} value={c.cluster_id}>
                        Cluster #{c.cluster_id} — {c.incident_count} pts
                      </option>
                    ))}
                  </select>

                  {(clusterFilter !== null || categoryFilter || timeWindowFilter) && (
                    <button
                      onClick={() => { setClusterFilter(null); setCategoryFilter(""); setTimeWindowFilter(""); }}
                      className="text-xs underline"
                      style={{ color: "var(--sky-400)" }}
                    >
                      Reset
                    </button>
                  )}

                  {loadingMapIncidents && (
                    <span className="text-xs flex items-center gap-1 animate-pulse" style={{ color: "var(--amber)" }}>
                      <RefreshCw className="w-3 h-3 animate-spin" /> Updating…
                    </span>
                  )}
                </div>
              </div>

              {/* Map */}
              <div className="card" style={{ padding: 0, overflow: "hidden", height: 600 }}>
                <CrimeMap
                  clusters={clusters}
                  incidents={mapIncidents}
                  filterClusterId={clusterFilter}
                  onSelectCluster={setSelectedCluster}
                />
              </div>

              {/* Selected cluster detail */}
              {selectedCluster && (
                <div className="card flex items-center justify-between gap-4"
                  style={{ borderColor: "var(--sky-500)" }}>
                  <div>
                    <span className="font-bold text-sm" style={{ color: "var(--sky-400)" }}>
                      Selected: Cluster #{selectedCluster.cluster_id}
                    </span>
                    <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
                      Centroid {selectedCluster.centroid_latitude.toFixed(4)}°N,{" "}
                      {selectedCluster.centroid_longitude.toFixed(4)}°W ·{" "}
                      {selectedCluster.incident_count.toLocaleString()} incidents ({selectedCluster.percentage_of_total}%) ·{" "}
                      Dominant: {selectedCluster.primary_crime_type}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedCluster(null)}
                    className="text-xs px-3 py-1.5 rounded-lg"
                    style={{ background: "var(--bg-base)", color: "var(--text-muted)", border: "1px solid var(--border)" }}
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ══════════════ TAB 3: HOTSPOTS ══════════════ */}
          {activeTab === "hotspots" && (
            <div className="space-y-5">
              {/* Algorithm banner */}
              <div className="card">
                <div className="flex items-center gap-2 mb-3">
                  <Flame className="w-4 h-4" style={{ color: "var(--red)" }} />
                  <h3 className="section-title">DBSCAN Configuration & Sensitivity</h3>
                </div>
                <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                  DBSCAN identifies clusters where geographic density exceeds a critical threshold without enforcing
                  spherical geometry. Distance is computed using the <strong>Haversine</strong> formula on radians
                  (ε = 500 m → ε_rad ≈ 7.85 × 10⁻⁵). Points without {params.min_samples} neighbours are labelled
                  as <strong>noise (−1)</strong>.
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                  {[
                    { label: "Neighbourhood Radius (ε)", value: `${params.eps_meters} m`, accent: "var(--red)" },
                    { label: "Min Samples", value: `${params.min_samples} incidents`, accent: "var(--red)" },
                    { label: "Earth Radius Model", value: "6,371.0088 km", accent: "var(--sky-400)" },
                    { label: "Distance Metric", value: "Haversine (Radians)", accent: "var(--sky-400)" },
                  ].map((item) => (
                    <div key={item.label} className="rounded-xl p-3 text-xs" style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}>
                      <span className="block mb-1" style={{ color: "var(--text-muted)" }}>{item.label}</span>
                      <strong className="font-mono text-sm" style={{ color: item.accent }}>{item.value}</strong>
                    </div>
                  ))}
                </div>
              </div>

              {/* Experiment table */}
              <div className="card">
                <SectionHeader
                  title="Hyperparameter Sensitivity Experiments"
                  sub="25 tested combinations across ε ∈ [200m, 1000m] and min_samples ∈ [15, 100]"
                />
                <div className="overflow-x-auto">
                  <table className="data-table w-full text-left">
                    <thead>
                      <tr>
                        <th>ε (m)</th>
                        <th>min_samples</th>
                        <th>Clusters</th>
                        <th>Clustered Pts</th>
                        <th>Clustered %</th>
                        <th>Noise %</th>
                        <th>Mean Size</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {experiments.slice(0, 15).map((exp: any, i: number) => {
                        const isSel = exp.eps_meters === params.eps_meters && exp.min_samples === params.min_samples;
                        return (
                          <tr key={i} style={{ color: isSel ? "var(--sky-400)" : "var(--text-secondary)" }}>
                            <td>{exp.eps_meters}</td>
                            <td>{exp.min_samples}</td>
                            <td>{exp.num_clusters}</td>
                            <td>{exp.clustered_points.toLocaleString()}</td>
                            <td>{exp.clustered_percentage}%</td>
                            <td>{exp.noise_percentage}%</td>
                            <td>{exp.mean_cluster_size}</td>
                            <td>
                              {isSel
                                ? <span className="badge-green">SELECTED</span>
                                : <span style={{ color: "var(--text-muted)", fontSize: 11 }}>Eval</span>}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Cluster summary table */}
              <div className="card">
                <SectionHeader
                  title="Detected Hotspot Clusters"
                  sub="Centroids, bounding coordinates, and dominant crime classifications"
                />
                <div className="overflow-x-auto" style={{ maxHeight: 380 }}>
                  <table className="data-table w-full text-left">
                    <thead className="sticky top-0" style={{ background: "var(--bg-card)" }}>
                      <tr>
                        <th>ID</th>
                        <th>Count</th>
                        <th>Share</th>
                        <th>Centroid</th>
                        <th>Dominant Crime</th>
                        <th>Area</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {clusters.map((c: any) => (
                        <tr key={c.cluster_id}>
                          <td><span className="font-bold" style={{ color: "var(--red)" }}>#{c.cluster_id}</span></td>
                          <td>{c.incident_count.toLocaleString()}</td>
                          <td>{c.percentage_of_total}%</td>
                          <td>{c.centroid_latitude.toFixed(4)}°N, {c.centroid_longitude.toFixed(4)}°W</td>
                          <td className="font-sans">{c.primary_crime_type}</td>
                          <td>Area {c.dominant_community_area}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════ TAB 4: ANALYTICS ══════════════ */}
          {activeTab === "analytics" && (
            <div className="space-y-5">
              {/* Summary metric cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Geographic extent */}
                <div className="card">
                  <div className="flex items-center gap-2 mb-3">
                    <Compass className="w-4 h-4" style={{ color: "var(--sky-400)" }} />
                    <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                      Spatial Coverage
                    </span>
                  </div>
                  <div className="space-y-2 text-xs font-mono">
                    {[
                      ["Lat Range", `${eda.geographic_extent.min_latitude.toFixed(3)}°N – ${eda.geographic_extent.max_latitude.toFixed(3)}°N`],
                      ["Lon Range", `${eda.geographic_extent.min_longitude.toFixed(3)}°W – ${eda.geographic_extent.max_longitude.toFixed(3)}°W`],
                      ["Mean Center", `${eda.geographic_extent.mean_latitude.toFixed(4)}°N, ${eda.geographic_extent.mean_longitude.toFixed(4)}°W`],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        <span style={{ color: "var(--text-muted)", fontFamily: "inherit" }}>{k}</span>
                        <span style={{ color: "var(--sky-400)" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Temporal */}
                <div className="card">
                  <div className="flex items-center gap-2 mb-3">
                    <Clock className="w-4 h-4" style={{ color: "var(--amber)" }} />
                    <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                      Temporal Patterns
                    </span>
                  </div>
                  <div className="space-y-2 text-xs font-mono">
                    {[
                      ["Date Range", "2020 – 2025"],
                      ["Diurnal Trough", "5:00 AM (lowest)"],
                      ["Diurnal Peak", "12 PM – 10 PM"],
                      ["Seasonality", "Summer +30% vs Feb"],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        <span style={{ color: "var(--text-muted)", fontFamily: "inherit" }}>{k}</span>
                        <span style={{ color: "var(--text-secondary)" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cluster concentration */}
                <div className="card">
                  <div className="flex items-center gap-2 mb-3">
                    <TrendingUp className="w-4 h-4" style={{ color: "var(--red)" }} />
                    <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--text-muted)" }}>
                      Cluster Metrics
                    </span>
                  </div>
                  <div className="space-y-2 text-xs font-mono">
                    {[
                      ["Largest Hotspot", `${kpis.largest_cluster_size.toLocaleString()} incidents`],
                      ["Smallest Hotspot", `${kpis.smallest_cluster_size} incidents`],
                      ["Median Size", `${kpis.median_cluster_size} incidents`],
                      ["Noise Fraction", `${kpis.noise_percentage}%`],
                    ].map(([k, v]) => (
                      <div key={k} className="flex justify-between py-1" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        <span style={{ color: "var(--text-muted)", fontFamily: "inherit" }}>{k}</span>
                        <span style={{ color: "var(--red)" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Interactive charts */}
              {spatial_temporal && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                  {/* Diurnal area chart */}
                  <div className="card">
                    <SectionHeader
                      title="24-Hour Diurnal Distribution"
                      sub={`Hourly incidents across ${spatial_temporal.total_records_with_valid_timestamps.toLocaleString()} timestamped records`}
                    />
                    <div style={{ height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={spatial_temporal.hourly_distribution}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                          <XAxis dataKey="label" stroke={CHART_AXIS} fontSize={10} interval={3} />
                          <YAxis stroke={CHART_AXIS} fontSize={10} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} />
                          <Area type="monotone" dataKey="count" name="Incidents"
                            stroke="var(--sky-500)" fill="var(--sky-500)" fillOpacity={0.2} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Day/Night bar chart */}
                  <div className="card">
                    <SectionHeader
                      title="Daytime vs. Nighttime by Cluster"
                      sub="Top 8 clusters — 06:00–17:59 (day) vs. 18:00–05:59 (night)"
                    />
                    <div style={{ height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.cluster_day_night_shifts.slice(0, 8)}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                          <XAxis dataKey="cluster_label" stroke={CHART_AXIS} fontSize={10} />
                          <YAxis stroke={CHART_AXIS} fontSize={10} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="daytime_count"   name="Daytime"   fill="var(--sky-400)"  radius={[4,4,0,0]} />
                          <Bar dataKey="nighttime_count" name="Nighttime" fill="var(--red)"      radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Time-window bar */}
                  <div className="card">
                    <SectionHeader
                      title="6-Hour Activity Windows"
                      sub="Incident volume across morning, afternoon, evening, night"
                    />
                    <div style={{ height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.time_windows}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                          <XAxis dataKey="window" stroke={CHART_AXIS} fontSize={10} />
                          <YAxis stroke={CHART_AXIS} fontSize={10} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} />
                          <Bar dataKey="count" name="Incidents" fill="var(--amber)" radius={[4,4,0,0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Category hotspot stacked bar */}
                  <div className="card">
                    <SectionHeader
                      title="Category Hotspot Concentration"
                      sub="Proportion inside DBSCAN clusters vs. dispersed noise per offense type"
                    />
                    <div style={{ height: 220 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.category_hotspot_profiles.slice(0, 7)}>
                          <CartesianGrid strokeDasharray="3 3" stroke={CHART_GRID} />
                          <XAxis dataKey="category" stroke={CHART_AXIS} fontSize={9} />
                          <YAxis stroke={CHART_AXIS} fontSize={10} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="clustered" stackId="a" name="In Hotspots" fill="var(--sky-500)" />
                          <Bar dataKey="noise"     stackId="a" name="Noise (−1)"  fill="#334155" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}

              {/* Visualisation gallery */}
              <div className="card">
                <SectionHeader
                  title="Generated Analytical Visualisations"
                  sub="Exported to outputs/figures/ from real Chicago crime data"
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    { title: "A — Incident Scatter", desc: "Individual offenses plotted by coordinate. Heavy clustering in Loop/Western corridors.", file: "vis_a_historical_locations.png" },
                    { title: "B — Density Heatmap (Hexbin)", desc: "Log-scaled frequency per spatial bin highlighting chronic density corridors.", file: "vis_b_crime_density_heatmap.png" },
                    { title: "C — Crime Categories", desc: "THEFT (21.8%) and BATTERY (18.0%) comprise nearly 40% of all incidents.", file: "vis_c_crime_category_distribution.png" },
                    { title: "D — Community Area Volume", desc: "Austin (#25), Near North (#8), Near West (#28), and Loop (#32) top the volume ranking.", file: "vis_d_area_level_distribution.png" },
                  ].map((v) => (
                    <div key={v.title} className="rounded-xl p-4 text-xs"
                      style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}>
                      <h4 className="font-semibold mb-1" style={{ color: "var(--text-primary)" }}>{v.title}</h4>
                      <p className="mb-2 leading-relaxed" style={{ color: "var(--text-muted)" }}>{v.desc}</p>
                      <span className="font-mono text-[10px]" style={{ color: "var(--sky-500)" }}>
                        outputs/figures/{v.file}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ══════════════ TAB 5: PREDICTION ══════════════ */}
          {activeTab === "prediction" && (
            <div className="max-w-3xl mx-auto space-y-5">
              {/* Notice */}
              <div className="rounded-xl p-5 flex gap-3"
                style={{ background: "rgba(245,158,11,0.08)", border: "1px solid rgba(245,158,11,0.3)" }}>
                <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: "var(--amber)" }} />
                <div className="text-xs space-y-1.5">
                  <h3 className="font-bold text-sm" style={{ color: "var(--amber-light)" }}>
                    Supervised ML Integration: Pending
                  </h3>
                  <p style={{ color: "rgba(253,230,138,0.85)" }}>
                    Lina (Member 1) has completed data preprocessing and core model training. Final validation on the
                    2025 held-out split is currently in progress.
                  </p>
                  <p className="font-medium" style={{ color: "var(--amber)" }}>
                    The form below computes a <strong>provisional spatial proximity score</strong> (Haversine distance
                    to the nearest historical DBSCAN hotspot). It is NOT a confirmed future crime forecast.
                  </p>
                </div>
              </div>

              {/* Input form */}
              <div className="card">
                <SectionHeader
                  title="Query Spatial Risk Exposure"
                  sub="Enter coordinates or choose a preset to evaluate proximity to historical hotspot centroids."
                />

                {/* Presets */}
                <div className="mb-5">
                  <p className="text-xs mb-2 font-medium" style={{ color: "var(--text-muted)" }}>Quick Presets:</p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { name: "Downtown / Loop",        lat: 41.8781, lon: -87.6298 },
                      { name: "Near North Side (#8)",   lat: 41.8996, lon: -87.6334 },
                      { name: "Austin (#25)",            lat: 41.8900, lon: -87.7600 },
                      { name: "South Shore (#43)",       lat: 41.7600, lon: -87.5700 },
                      { name: "Far South (Low Density)", lat: 41.6600, lon: -87.5600 },
                    ].map((p) => (
                      <button
                        key={p.name}
                        type="button"
                        onClick={() => setPresetLocation(p.name, p.lat, p.lon)}
                        className="text-xs px-3 py-1.5 rounded-lg transition"
                        style={{
                          background: "var(--bg-base)",
                          border: "1px solid var(--border)",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>

                <form onSubmit={handlePredict} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { label: "Latitude (°N)",   val: predLat, set: setPredLat, type: "number", step: "0.0001" },
                      { label: "Longitude (°W)",  val: predLon, set: setPredLon, type: "number", step: "0.0001" },
                      { label: "Target Date",     val: predDate, set: setPredDate, type: "date",   step: undefined },
                      { label: "Target Time",     val: predTime, set: setPredTime, type: "time",   step: undefined },
                    ].map(({ label, val, set, type, step }) => (
                      <div key={label}>
                        <label className="block text-xs font-medium mb-1" style={{ color: "var(--text-secondary)" }}>
                          {label}
                        </label>
                        <input
                          type={type}
                          step={step}
                          value={val as any}
                          onChange={(e) => set(type === "number" ? parseFloat(e.target.value) : e.target.value as any)}
                          className="form-input w-full font-mono"
                          required
                        />
                      </div>
                    ))}
                  </div>

                  <button
                    type="submit"
                    disabled={isPredicting}
                    className="w-full py-2.5 px-4 rounded-xl text-sm font-semibold text-white flex items-center justify-center gap-2 transition"
                    style={{ background: isPredicting ? "var(--sky-700)" : "var(--sky-600)" }}
                  >
                    {isPredicting
                      ? <><RefreshCw className="w-4 h-4 animate-spin" /> Calculating…</>
                      : <><Activity className="w-4 h-4" /> Evaluate Spatial Risk</>}
                  </button>
                </form>
              </div>

              {/* Result */}
              {predResult && (
                <div className="card space-y-4" style={{ borderColor: "var(--sky-500)" }}>
                  <div className="flex items-center justify-between pb-3" style={{ borderBottom: "1px solid var(--border)" }}>
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5" style={{ color: "var(--sky-400)" }} />
                      <h4 className="font-bold" style={{ color: "var(--text-primary)" }}>
                        Provisional Spatial Analysis Result
                      </h4>
                    </div>
                    <span className="badge-sky font-mono">{predResult.status}</span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                    <div className="rounded-xl p-3" style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}>
                      <span className="block mb-1" style={{ color: "var(--text-muted)" }}>Spatial Risk Tier</span>
                      <strong className="font-semibold" style={{ color: "var(--red)" }}>
                        {predResult.provisional_spatial_risk_tier}
                      </strong>
                    </div>
                    <div className="rounded-xl p-3" style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}>
                      <span className="block mb-1" style={{ color: "var(--text-muted)" }}>Distance to Nearest Hotspot</span>
                      <strong className="font-mono" style={{ color: "var(--text-primary)" }}>
                        {predResult.distance_to_hotspot_km} km
                      </strong>
                    </div>
                    <div className="rounded-xl p-3" style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}>
                      <span className="block mb-1" style={{ color: "var(--text-muted)" }}>Nearest Cluster</span>
                      <strong className="font-mono" style={{ color: "var(--text-primary)" }}>
                        #{predResult.nearest_hotspot.cluster_id} ({predResult.nearest_hotspot.incident_count} pts)
                      </strong>
                    </div>
                  </div>

                  <div className="rounded-xl p-3 text-xs leading-relaxed"
                    style={{ background: "var(--bg-base)", border: "1px solid var(--border)", color: "var(--text-muted)" }}>
                    <strong style={{ color: "var(--text-secondary)" }}>Methodological Note: </strong>
                    {predResult.disclaimer}
                  </div>
                </div>
              )}

              {/* History */}
              <div className="card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <History className="w-4 h-4" style={{ color: "#34d399" }} />
                    <h3 className="section-title">Recent Persisted Queries</h3>
                  </div>
                  <span className="badge-green">DB: {historyBackend}</span>
                </div>

                {predHistory.length === 0 ? (
                  <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                    {backendOnline
                      ? "No queries yet. Submit a coordinate above to log it."
                      : "Start the FastAPI backend on :8000 to view query history."}
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="data-table w-full text-left">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Coordinates</th>
                          <th>Nearest Cluster</th>
                          <th>Distance</th>
                          <th>Risk Tier</th>
                          <th>Time</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {predHistory.map((h: any) => (
                          <tr key={h.id}>
                            <td style={{ color: "var(--text-muted)" }}>#{h.id}</td>
                            <td>{h.latitude.toFixed(4)}°N, {h.longitude.toFixed(4)}°W</td>
                            <td style={{ color: "var(--red)" }}>
                              {h.nearest_cluster_id !== null ? `Cluster #${h.nearest_cluster_id}` : "N/A"}
                            </td>
                            <td>{h.distance_to_hotspot_km !== null ? `${h.distance_to_hotspot_km} km` : "N/A"}</td>
                            <td className="font-sans">{h.risk_tier}</td>
                            <td style={{ color: "var(--text-muted)" }}>{h.target_timestamp}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══════════════ TAB 6: METHODOLOGY ══════════════ */}
          {activeTab === "methodology" && (
            <div className="max-w-3xl mx-auto space-y-5">
              <div className="card space-y-5">
                <div className="flex items-center gap-2 pb-4" style={{ borderBottom: "1px solid var(--border)" }}>
                  <BookOpen className="w-5 h-5" style={{ color: "var(--sky-400)" }} />
                  <h3 className="section-title">Spatial Analysis Methodology & Algorithmic Rationale</h3>
                </div>

                {/* Section 1 */}
                <div className="space-y-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: "var(--sky-400)" }}>
                    1. The Spherical Haversine Distance Metric
                  </h4>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    At Chicago&apos;s latitude (~41.88°N), one degree of latitude ≈ 111.0 km while one degree of longitude ≈
                    82.7 km. Treating geographic degrees as planar Euclidean coordinates introduces a systematic distortion
                    of over <strong>34%</strong>. We circumvent this by converting to radians and applying the{" "}
                    <strong>Haversine formula</strong>:
                  </p>
                  <div className="rounded-xl p-3 font-mono text-xs overflow-x-auto"
                    style={{ background: "var(--bg-base)", border: "1px solid var(--border)", color: "var(--sky-400)" }}>
                    d = 2R · arcsin(√(sin²(Δφ/2) + cos(φ₁)cos(φ₂)sin²(Δλ/2)))
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    R = 6,371.0088 km (IUGG volumetric mean). The DBSCAN ε is passed as a radian:
                    ε_rad = ε_km / R.
                  </p>
                </div>

                {/* Section 2 */}
                <div className="space-y-3 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
                  <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: "var(--sky-400)" }}>
                    2. DBSCAN Clustering & Parameter Selection
                  </h4>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    DBSCAN discovers clusters of arbitrary non-convex shapes based on local spatial density. We evaluated
                    25 hyperparameter combinations across ε ∈ [200m, 1000m] and min_samples ∈ [15, 100].{" "}
                    <strong>Selected: ε = 500 m, min_samples = 50</strong> — balancing street-block granularity without
                    spatial fragmentation or collapsing into an uninformative mega-cluster.
                  </p>
                </div>

                {/* Section 3 */}
                <div className="space-y-3 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
                  <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: "var(--sky-400)" }}>
                    3. Historical Hotspots vs. Future Risk Prediction
                  </h4>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                    A DBSCAN cluster is a <strong>retrospective summary</strong> of where recorded incidents historically
                    concentrated. Hotspot clustering alone cannot predict when, where, or why future offences will occur.
                    Prospective forecasting requires supervised regression and classification models integrating calendar
                    cycles, weather, land use, and temporal lags — Member 1&apos;s (Lina&apos;s) ongoing contribution.
                  </p>
                </div>

                {/* Section 4 */}
                <div className="space-y-3 pt-4" style={{ borderTop: "1px solid var(--border)" }}>
                  <h4 className="text-xs font-bold uppercase tracking-wider" style={{ color: "var(--red)" }}>
                    4. Security, Fairness & Ethical Safeguards
                  </h4>
                  <ul className="text-xs leading-relaxed space-y-2" style={{ color: "var(--text-secondary)" }}>
                    {[
                      ["No Sensitive Personal Attributes", "Race, ethnicity, religion, and personal identifiers are strictly excluded from the analytical pipeline."],
                      ["Reporting & Enforcement Bias Awareness", "Police records reflect reported incidents and active enforcement patterns, which can over-represent proactive policing zones. Outputs must never be interpreted as inherent community criminality."],
                      ["Strict Data Leakage Prevention", "Post-investigation fields (Arrest, IUCR, FBI Code, Description) are quarantined and prohibited from prospective feature pipelines."],
                      ["Decision Support Context", "System outputs are designed strictly for resource allocation and public safety support, not automated punitive actions."],
                    ].map(([title, body]) => (
                      <li key={title} className="flex gap-2">
                        <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "#34d399" }} />
                        <span><strong style={{ color: "var(--text-primary)" }}>{title}: </strong>{body}</span>
                      </li>
                    ))}
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
