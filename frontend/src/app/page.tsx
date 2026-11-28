"use client";

import React, { useState, useEffect, useCallback } from "react";
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
  ArrowRight,
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

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const RISK_TIER_STYLES: Record<
  string,
  { badge: string; border: string; bg: string; text: string }
> = {
  Low: {
    badge: "bg-emerald-100 text-emerald-800 border-emerald-300",
    border: "border-emerald-300",
    bg: "bg-emerald-50/70",
    text: "text-emerald-700",
  },
  Moderate: {
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    border: "border-amber-300",
    bg: "bg-amber-50/70",
    text: "text-amber-700",
  },
  High: {
    badge: "bg-orange-100 text-orange-800 border-orange-300",
    border: "border-orange-300",
    bg: "bg-orange-50/70",
    text: "text-orange-700",
  },
  "Very High": {
    badge: "bg-red-100 text-red-800 border-red-300",
    border: "border-red-300",
    bg: "bg-red-50/70",
    text: "text-red-700",
  },
};

// Dynamically import CrimeMap with fallback
const CrimeMap = dynamic(() => import("../components/CrimeMap"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[460px] flex flex-col items-center justify-center gap-3 bg-sky-50/50 rounded-2xl border border-sky-200">
      <RefreshCw className="w-6 h-6 animate-spin text-sky-600" />
      <span className="text-xs font-semibold text-slate-600">
        Loading Interactive Chicago Map…
      </span>
    </div>
  ),
});

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<
    "dashboard" | "map" | "hotspots" | "analytics" | "prediction" | "methodology"
  >("dashboard");

  // Map & filter states
  const [selectedCluster, setSelectedCluster] = useState<any>(null);
  const [clusterFilter, setClusterFilter] = useState<number | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>("");
  const [timeWindowFilter, setTimeWindowFilter] = useState<string>("");
  const [mapIncidents, setMapIncidents] = useState<any[]>([]);
  const [loadingMapIncidents, setLoadingMapIncidents] = useState<boolean>(false);

  // Interactive Map: ML Hotspots (617-Cluster System), KDE Surface & Layer Toggles
  const [mapAreaId, setMapAreaId] = useState<number>(8);
  const [mlMapHotspots, setMlMapHotspots] = useState<any[]>([]);
  const [mlMapMeta, setMlMapMeta] = useState<{
    total_matching_clusters: number;
    dominant_clusters_count: number;
    cross_boundary_clusters_count: number;
  } | null>(null);
  const [loadingMlHotspots, setLoadingMlHotspots] = useState<boolean>(false);
  const [mlMapError, setMlMapError] = useState<string | null>(null);
  const [showIncidentsLayer, setShowIncidentsLayer] = useState<boolean>(true);
  const [showSpatialClustersLayer, setShowSpatialClustersLayer] = useState<boolean>(true);
  const [showMlHotspotsLayer, setShowMlHotspotsLayer] = useState<boolean>(true);
  const [showKdeLayer, setShowKdeLayer] = useState<boolean>(false);

  // Hotspot sub-tab
  const [hotspotSubTab, setHotspotSubTab] = useState<"clusters" | "experiments" | "params">("clusters");
  const [clusterSearchQuery, setClusterSearchQuery] = useState<string>("");

  // ML Community Area risk prediction form (POST /api/predict)
  const [mlAreaId, setMlAreaId] = useState<number>(8);
  const [mlDate, setMlDate] = useState<string>("2024-06-15");
  const [mlIncludeHotspots, setMlIncludeHotspots] = useState<boolean>(true);
  const [mlPredResult, setMlPredResult] = useState<any>(null);
  const [isMlPredicting, setIsMlPredicting] = useState<boolean>(false);
  const [mlPredError, setMlPredError] = useState<string | null>(null);
  const [mlServiceReady, setMlServiceReady] = useState<boolean>(false);
  const [availableAreas, setAvailableAreas] = useState<number[]>(
    Array.from({ length: 77 }, (_, idx) => idx + 1)
  );
  const [modelInfo, setModelInfo] = useState<any>(null);

  // Spatial proximity risk prediction form (POST /api/predict-risk)
  const [predLat, setPredLat] = useState<number>(41.8781);
  const [predLon, setPredLon] = useState<number>(-87.6298);
  const [predDate, setPredDate] = useState<string>("2026-11-25");
  const [predTime, setPredTime] = useState<string>("14:30");
  const [predResult, setPredResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);
  const [predHistory, setPredHistory] = useState<any[]>([]);
  const [historyBackend, setHistoryBackend] = useState<string>("SQLite");
  const [historyTypeFilter, setHistoryTypeFilter] = useState<
    "all" | "ml_forecast" | "spatial_proximity"
  >("all");

  const fallbackData = analysisData as any;
  const [summary, setSummary] = useState<any>(fallbackData.summary);
  const [clusters, setClusters] = useState<any[]>(fallbackData.clusters || []);
  const [experiments, setExperiments] = useState<any[]>(fallbackData.experiments || []);
  const [spatial_temporal, setSpatialTemporal] = useState<any>(
    fallbackData.spatial_temporal
  );
  const [kdeSurface, setKdeSurface] = useState<any>(null);
  const [kdeMeta, setKdeMeta] = useState<{
    grid_size: number;
    bandwidth_meters: number;
    target_year: number;
  } | null>(null);
  const [analyticsDataSource, setAnalyticsDataSource] = useState<
    "loading" | "live" | "fallback"
  >("loading");
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const eda = fallbackData.eda;

  const kpis = summary?.results_summary || {};
  const params = summary?.selected_hyperparameters || {};

  // Derive temporal summary statistics dynamically from spatial_temporal
  const hourlyDist: any[] = spatial_temporal?.hourly_distribution || [];
  const diurnalTrough =
    hourlyDist.length > 0
      ? hourlyDist.reduce((min, cur) => (cur.count < min.count ? cur : min), hourlyDist[0])
      : null;
  const diurnalPeak =
    hourlyDist.length > 0
      ? hourlyDist.reduce((max, cur) => (cur.count > max.count ? cur : max), hourlyDist[0])
      : null;

  // Derive spatial cluster envelope dynamically from clusters
  const clusterEnvelope =
    clusters.length > 0
      ? {
          minLat: Math.min(...clusters.map((c: any) => c.min_latitude)),
          maxLat: Math.max(...clusters.map((c: any) => c.max_latitude)),
        }
      : null;

  // Derive map category filter options from live spatial_temporal profiles (with eda fallback)
  const mapCategories: string[] =
    Array.isArray(spatial_temporal?.category_hotspot_profiles) &&
    spatial_temporal.category_hotspot_profiles.length > 0
      ? spatial_temporal.category_hotspot_profiles.map((c: any) => c.category)
      : Object.keys(eda?.categories?.top_10 || {});

  // Fetch prediction query history
  const fetchPredictionHistory = useCallback(
    (typeFilter: "all" | "ml_forecast" | "spatial_proximity" = historyTypeFilter) => {
      const qs = new URLSearchParams({ limit: "12" });
      if (typeFilter !== "all") {
        qs.set("prediction_type", typeFilter);
      }
      fetch(`${API_BASE_URL}/api/prediction-history?${qs.toString()}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d?.history) {
            setPredHistory(d.history);
            setHistoryBackend(d.database_backend || "SQLite");
          }
        })
        .catch(() => {});
    },
    [historyTypeFilter]
  );

  useEffect(() => {
    let mounted = true;

    fetch(`${API_BASE_URL}/health`)
      .then((r) => {
        if (r.ok && mounted) {
          setBackendOnline(true);
          fetchPredictionHistory();
        }
      })
      .catch(() => {
        if (mounted) setBackendOnline(false);
      });

    fetch(`${API_BASE_URL}/health/ready`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (mounted && d?.status === "ready") {
          setMlServiceReady(true);
        }
      })
      .catch(() => {
        if (mounted) setMlServiceReady(false);
      });

    fetch(`${API_BASE_URL}/api/areas`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (mounted && Array.isArray(d?.areas) && d.areas.length > 0) {
          setAvailableAreas(d.areas.map((a: any) => a.community_area_id));
        }
      })
      .catch(() => {});

    fetch(`${API_BASE_URL}/api/model-info`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (mounted && d) setModelInfo(d);
      })
      .catch(() => {});

    // Priority 1: Load live spatial analysis data from backend APIs (with offline fallback)
    Promise.allSettled([
      fetch(`${API_BASE_URL}/api/summary`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`${API_BASE_URL}/api/hotspots`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`${API_BASE_URL}/api/experiments`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`${API_BASE_URL}/api/spatial-temporal`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
      fetch(`${API_BASE_URL}/api/kde`).then((r) => (r.ok ? r.json() : Promise.reject(r.status))),
    ]).then(([sumRes, hotRes, expRes, stRes, kdeRes]) => {
      if (!mounted) return;
      let anyLive = false;
      const errors: string[] = [];

      if (sumRes.status === "fulfilled" && sumRes.value?.results_summary) {
        setSummary(sumRes.value);
        anyLive = true;
      } else {
        errors.push("/api/summary");
      }

      if (hotRes.status === "fulfilled" && Array.isArray(hotRes.value?.clusters)) {
        setClusters(hotRes.value.clusters);
        anyLive = true;
      } else {
        errors.push("/api/hotspots");
      }

      if (expRes.status === "fulfilled" && Array.isArray(expRes.value)) {
        setExperiments(expRes.value);
        anyLive = true;
      } else {
        errors.push("/api/experiments");
      }

      if (stRes.status === "fulfilled" && stRes.value?.hourly_distribution) {
        setSpatialTemporal(stRes.value);
        anyLive = true;
      } else {
        errors.push("/api/spatial-temporal");
      }

      if (kdeRes.status === "fulfilled" && kdeRes.value?.grid_size) {
        setKdeSurface(kdeRes.value);
        setKdeMeta({
          grid_size: kdeRes.value.grid_size,
          bandwidth_meters: kdeRes.value.bandwidth_meters,
          target_year: kdeRes.value.target_year,
        });
      }

      setAnalyticsDataSource(anyLive ? "live" : "fallback");
      setAnalyticsError(
        errors.length > 0 && anyLive
          ? `Partial live load (fallback used for: ${errors.join(", ")})`
          : errors.length > 0
          ? "Backend unavailable — displaying offline precomputed analysisData.json fallback."
          : null
      );
    });

    return () => {
      mounted = false;
    };
  }, [fetchPredictionHistory]);

  // Priority 3: Fetch ML cross-boundary DBSCAN hotspots for selected Community Area (GET /api/hotspots/{area_id})
  useEffect(() => {
    if (activeTab !== "map" || !mlServiceReady) return;
    let mounted = true;
    setLoadingMlHotspots(true);
    setMlMapError(null);

    fetch(`${API_BASE_URL}/api/hotspots/${mapAreaId}?limit=50`)
      .then(async (r) => {
        const d = await r.json().catch(() => null);
        if (!r.ok) {
          throw new Error(d?.detail || `Failed with status ${r.status}`);
        }
        return d;
      })
      .then((d) => {
        if (!mounted) return;
        setMlMapHotspots(Array.isArray(d?.clusters) ? d.clusters : []);
        setMlMapMeta({
          total_matching_clusters: d?.total_matching_clusters ?? 0,
          dominant_clusters_count: d?.dominant_clusters_count ?? 0,
          cross_boundary_clusters_count: d?.cross_boundary_clusters_count ?? 0,
        });
      })
      .catch((err: any) => {
        if (!mounted) return;
        setMlMapHotspots([]);
        setMlMapMeta(null);
        setMlMapError(err?.message || "Unable to load ML Community Area hotspots.");
      })
      .finally(() => {
        if (mounted) setLoadingMlHotspots(false);
      });

    return () => {
      mounted = false;
    };
  }, [activeTab, mlServiceReady, mapAreaId]);

  // Fetch filtered incidents when map filters change
  useEffect(() => {
    if (activeTab !== "map" || !backendOnline) return;
    setLoadingMapIncidents(true);
    const qs = new URLSearchParams({ limit: "600" });
    if (clusterFilter !== null) qs.set("cluster_id", String(clusterFilter));
    if (categoryFilter) qs.set("category", categoryFilter);
    if (timeWindowFilter) qs.set("time_window", timeWindowFilter);

    fetch(`${API_BASE_URL}/api/historical-incidents?${qs.toString()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d?.incidents) setMapIncidents(d.incidents);
      })
      .catch(() => setMapIncidents([]))
      .finally(() => setLoadingMapIncidents(false));
  }, [activeTab, backendOnline, clusterFilter, categoryFilter, timeWindowFilter]);

  const handleMlPredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsMlPredicting(true);
    setMlPredError(null);
    setMlPredResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          community_area: mlAreaId,
          date: mlDate,
          include_hotspots: mlIncludeHotspots,
          hotspot_limit: 5,
        }),
      });

      const data = await res.json().catch(() => null);
      if (!res.ok) {
        const detail =
          typeof data?.detail === "string"
            ? data.detail
            : Array.isArray(data?.detail)
            ? data.detail.map((item: any) => item.msg || JSON.stringify(item)).join("; ")
            : `Prediction request failed with status ${res.status}.`;
        setMlPredError(detail);
        return;
      }

      setMlPredResult(data);
      fetchPredictionHistory();
    } catch {
      setMlPredError(
        `Unable to reach the prediction backend at ${API_BASE_URL}. Please ensure the FastAPI service is running.`
      );
    } finally {
      setIsMlPredicting(false);
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict-risk`, {
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
      // Local fallback calculation if backend disconnected
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
      status: "SPATIAL_PROXIMITY_ONLY",
      query_coordinates: { latitude: predLat, longitude: predLon },
      nearest_hotspot: nearestC,
      distance_to_hotspot_km: parseFloat(minD.toFixed(3)),
      provisional_spatial_risk_tier: tier,
      model_integration_status: "OFFLINE_SPATIAL_FALLBACK",
      disclaimer:
        "Notice: This is a geometric Haversine proximity calculation relative to historical DBSCAN clusters. Supervised ML crime-count forecasting requires the FastAPI backend (POST /api/predict).",
    });
    setIsPredicting(false);
  };

  const navItems = [
    { id: "dashboard", label: "Dashboard Overview", icon: LayoutDashboard },
    { id: "map", label: "Interactive Crime Map", icon: MapPin },
    { id: "hotspots", label: "Hotspot Analysis", icon: Flame },
    { id: "analytics", label: "Analytics & Trends", icon: BarChart3 },
    { id: "prediction", label: "Risk Assessment", icon: ShieldAlert },
    { id: "methodology", label: "Methodology", icon: BookOpen },
  ];

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
    <div className="flex h-screen w-screen overflow-hidden bg-[#f4f7fa] text-[#0f172a] font-sans">
      {/* ─── SIDEBAR: CLEAN CLOUDY WHITE ─────────────────────── */}
      <aside className="w-64 flex-shrink-0 flex flex-col justify-between p-6 bg-white border-r border-slate-200 z-20 shadow-sm">
        <div>
          {/* Logo & Platform Title */}
          <div className="flex items-center gap-3 pb-6 mb-6 border-b border-slate-100">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-sky-600 flex items-center justify-center shadow-md shadow-sky-500/20 text-white">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-extrabold text-sm tracking-tight text-slate-900">
                Urban Crime AI
              </h1>
              <p className="text-[11px] text-sky-600 font-bold uppercase tracking-wider">
                Spatial Intelligence
              </p>
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
                  className={`w-full flex items-center gap-3 px-3.5 py-3 rounded-xl text-xs font-bold transition-all ${
                    active
                      ? "bg-sky-50 text-sky-700 border border-sky-200 shadow-sm"
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  <Icon
                    className={`w-4 h-4 transition-colors ${
                      active ? "text-sky-600" : "text-slate-400"
                    }`}
                  />
                  <span>{label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer Meta / Project Badge */}
        <div className="pt-4 border-t border-slate-100 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-500 font-medium">Backend Engine</span>
            <span
              className={`flex items-center gap-1.5 font-bold ${
                backendOnline ? "text-emerald-600" : "text-amber-600"
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

          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-[11px] space-y-1">
            <div className="flex justify-between">
              <span className="text-slate-500">Author:</span>
              <span className="font-bold text-slate-800">Aditi Verma</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Deliverable:</span>
              <span className="text-sky-600 font-bold">Milestone 3</span>
            </div>
          </div>
        </div>
      </aside>

      {/* ─── MAIN CONTENT VIEWPORT ───────────────────────────── */}
      <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#f4f7fa]">
        {/* Modern Clean Header */}
        <header className="h-16 flex-shrink-0 px-8 flex items-center justify-between border-b border-slate-200 bg-white shadow-sm z-10">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-extrabold tracking-tight text-slate-900 capitalize">
              {activeTab === "dashboard" && "Executive Spatial Intelligence"}
              {activeTab === "map" && "Interactive Chicago Crime Map"}
              {activeTab === "hotspots" && "DBSCAN Crime Hotspot Analysis"}
              {activeTab === "analytics" && "Urban Crime Statistical & Temporal Trends"}
              {activeTab === "prediction" && "Location Spatial Risk Assessment"}
              {activeTab === "methodology" && "Methodology & Ethical Safeguards"}
            </h2>
            <span className="pill-sky hidden sm:inline-flex">
              Chicago 2020–2025
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="pill-cloudy">
              <Database className="w-3.5 h-3.5 text-sky-600" />
              <span className="font-bold">{kpis.total_records_analyzed.toLocaleString()} Incidents</span>
            </span>
            <span className="pill-red">
              <Flame className="w-3.5 h-3.5 text-red-600" />
              <span className="font-bold">{kpis.num_clusters_identified} Hotspot Zones</span>
            </span>
          </div>
        </header>

        {/* Scrollable Content Body */}
        <div className="flex-1 overflow-y-auto p-8">
          {/* ════════════════ TAB 1: DASHBOARD (UNCROWDED & SPACIOUS) ════════════════ */}
          {activeTab === "dashboard" && (
            <div className="max-w-6xl mx-auto space-y-8 pb-12">
              {/* Executive Welcome & Key Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
                <div className="cloud-card p-6 space-y-2">
                  <div className="flex items-center justify-between text-slate-500">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Total Analyzed Crimes
                    </span>
                    <span className="p-2.5 rounded-xl bg-sky-100 text-sky-600">
                      <Database className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-slate-900 tracking-tight">
                    {kpis.total_records_analyzed.toLocaleString()}
                  </div>
                  <p className="text-xs text-emerald-600 font-bold flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" />{" "}
                    {analyticsDataSource === "live"
                      ? "Live /api/summary Data"
                      : analyticsDataSource === "loading"
                      ? "Syncing with Backend…"
                      : "Offline Fallback Data"}
                  </p>
                </div>

                <div className="cloud-card p-6 space-y-2">
                  <div className="flex items-center justify-between text-slate-500">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Identified Hotspots
                    </span>
                    <span className="p-2.5 rounded-xl bg-red-100 text-red-600">
                      <Flame className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-red-600 tracking-tight">
                    {kpis.num_clusters_identified}
                  </div>
                  <p className="text-xs text-slate-500 font-medium">
                    Radius &epsilon;={params.eps_meters}m &middot; min_pts={params.min_samples}
                  </p>
                </div>

                <div className="cloud-card p-6 space-y-2">
                  <div className="flex items-center justify-between text-slate-500">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Cluster Crime Share
                    </span>
                    <span className="p-2.5 rounded-xl bg-sky-100 text-sky-600">
                      <Layers className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-sky-600 tracking-tight">
                    {kpis.clustered_percentage}%
                  </div>
                  <p className="text-xs text-slate-500 font-medium">
                    {kpis.clustered_incidents.toLocaleString()} crimes inside hotspots
                  </p>
                </div>

                <div className="cloud-card p-6 space-y-2">
                  <div className="flex items-center justify-between text-slate-500">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                      Dispersed Incidents
                    </span>
                    <span className="p-2.5 rounded-xl bg-amber-100 text-amber-600">
                      <Compass className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-3xl font-extrabold text-amber-600 tracking-tight">
                    {kpis.noise_percentage}%
                  </div>
                  <p className="text-xs text-slate-500 font-medium">
                    {kpis.noise_incidents.toLocaleString()} isolated baseline points
                  </p>
                </div>
              </div>

              {/* The Hero Centerpiece: Crime Hotspot Map */}
              <div className="cloud-card p-6 space-y-4">
                <div className="flex items-center justify-between flex-wrap gap-4 pb-2 border-b border-slate-100">
                  <div>
                    <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                      <MapPin className="w-5 h-5 text-sky-600" />
                      Chicago Crime Hotspot Distribution
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      High-density geospatial clusters identified via DBSCAN Haversine distance metric
                    </p>
                  </div>

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setActiveTab("map")}
                      className="px-4 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold flex items-center gap-2 shadow-sm transition"
                    >
                      Open Full Screen Map Studio <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Map Display */}
                <div className="rounded-2xl overflow-hidden border border-slate-200 h-[480px] w-full shadow-inner">
                  <CrimeMap
                    clusters={clusters}
                    filterClusterId={clusterFilter}
                    onSelectCluster={setSelectedCluster}
                  />
                </div>

                <div className="flex items-center justify-between text-xs text-slate-500 pt-1">
                  <span>* Switch between <strong>Dark Map</strong> and <strong>Light Map</strong> directly on the top-right map button.</span>
                  <span className="text-sky-700 font-bold font-mono">
                    {clusterEnvelope
                      ? `Cluster Envelope: ${clusterEnvelope.minLat.toFixed(2)}°N - ${clusterEnvelope.maxLat.toFixed(2)}°N`
                      : "Chicago Spatial Envelope"}
                  </span>
                </div>
                {analyticsError && (
                  <p className="text-[11px] text-amber-700 font-medium">{analyticsError}</p>
                )}
              </div>

              {/* Quick Navigation Cards: 3 Simple Steps */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div
                  onClick={() => setActiveTab("hotspots")}
                  className="cloud-card p-6 space-y-3 cursor-pointer hover:border-sky-300 transition group"
                >
                  <div className="w-10 h-10 rounded-xl bg-red-100 text-red-600 flex items-center justify-center font-bold">
                    <Flame className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-slate-900 group-hover:text-sky-600 transition">
                      1. Hotspot Directory
                    </h4>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      Explore detailed tables for all {clusters.length} discovered clusters and {experiments.length} hyperparameter evaluations.
                    </p>
                  </div>
                  <span className="text-xs font-bold text-sky-600 flex items-center gap-1 pt-1">
                    View Clusters <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>

                <div
                  onClick={() => setActiveTab("analytics")}
                  className="cloud-card p-6 space-y-3 cursor-pointer hover:border-sky-300 transition group"
                >
                  <div className="w-10 h-10 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center font-bold">
                    <BarChart3 className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-slate-900 group-hover:text-sky-600 transition">
                      2. Temporal Analytics
                    </h4>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      Analyze 24-hour crime diurnal waves, daytime vs nighttime shifts, and category breakdowns.
                    </p>
                  </div>
                  <span className="text-xs font-bold text-sky-600 flex items-center gap-1 pt-1">
                    View Charts <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>

                <div
                  onClick={() => setActiveTab("prediction")}
                  className="cloud-card p-6 space-y-3 cursor-pointer hover:border-sky-300 transition group"
                >
                  <div className="w-10 h-10 rounded-xl bg-amber-100 text-amber-700 flex items-center justify-center font-bold">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-slate-900 group-hover:text-sky-600 transition">
                      3. Spatial Risk Assessment
                    </h4>
                    <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                      Test any custom Chicago latitude/longitude to compute proximity to historical crime clusters.
                    </p>
                  </div>
                  <span className="text-xs font-bold text-sky-600 flex items-center gap-1 pt-1">
                    Run Assessment <ChevronRight className="w-3.5 h-3.5" />
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* ════════════════ TAB 2: CRIME MAP (DEDICATED FULL STUDIO) ════════════════ */}
          {activeTab === "map" && (
            <div className="max-w-6xl mx-auto space-y-5 h-full flex flex-col pb-8">
              {/* Clean Filter Controls Bar */}
              <div className="cloud-card p-5 flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <span className="p-2 rounded-xl bg-sky-100 text-sky-600">
                    <MapPin className="w-5 h-5" />
                  </span>
                  <div>
                    <h3 className="text-sm font-bold text-slate-900">
                      Geographic Spatial Studio
                    </h3>
                    <p className="text-xs text-slate-500">
                      Filter {clusters.length} DBSCAN clusters and verified crime incidents across Chicago
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-xs text-slate-500 flex items-center gap-1 font-bold">
                    <Filter className="w-3.5 h-3.5" /> Filters:
                  </span>

                  <select
                    value={categoryFilter}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="cloud-select"
                  >
                    <option value="">All Crime Categories</option>
                    {mapCategories.map((cat: string) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>

                  <select
                    value={timeWindowFilter}
                    onChange={(e) => setTimeWindowFilter(e.target.value)}
                    className="cloud-select"
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
                    className="cloud-select"
                  >
                    <option value="">All {clusters.length} Clusters</option>
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
                      className="px-3 py-1.5 rounded-lg bg-slate-200 hover:bg-slate-300 text-xs text-slate-700 font-bold transition"
                    >
                      Reset Filters
                    </button>
                  )}

                  {loadingMapIncidents && (
                    <span className="text-xs text-amber-600 flex items-center gap-1.5 font-bold animate-pulse ml-2">
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Fetching incidents…
                    </span>
                  )}
                </div>

                {/* Layer Toggles & Community Area Selector for 617-Cluster ML Hotspots & KDE */}
                <div className="w-full pt-3 mt-1 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex flex-wrap items-center gap-3 text-xs">
                    <span className="text-slate-500 font-bold flex items-center gap-1">
                      <Layers className="w-3.5 h-3.5" /> Map Layers:
                    </span>
                    <label className="flex items-center gap-1.5 cursor-pointer text-slate-700 font-medium">
                      <input
                        type="checkbox"
                        checked={showIncidentsLayer}
                        onChange={(e) => setShowIncidentsLayer(e.target.checked)}
                        className="rounded border-slate-300 text-sky-600"
                      />
                      Historical Incidents
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer text-slate-700 font-medium">
                      <input
                        type="checkbox"
                        checked={showSpatialClustersLayer}
                        onChange={(e) => setShowSpatialClustersLayer(e.target.checked)}
                        className="rounded border-slate-300 text-rose-600"
                      />
                      29 Spatial Clusters (&epsilon;=500m)
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer text-purple-800 font-semibold">
                      <input
                        type="checkbox"
                        checked={showMlHotspotsLayer}
                        onChange={(e) => setShowMlHotspotsLayer(e.target.checked)}
                        className="rounded border-slate-300 text-purple-600"
                      />
                      617-Cluster ML Hotspots (&epsilon;=100m)
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer text-amber-800 font-semibold">
                      <input
                        type="checkbox"
                        checked={showKdeLayer}
                        onChange={(e) => setShowKdeLayer(e.target.checked)}
                        className="rounded border-slate-300 text-amber-600"
                      />
                      KDE Density Grid {kdeMeta ? `(${kdeMeta.grid_size}×${kdeMeta.grid_size})` : ""}
                    </label>
                  </div>

                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-slate-500 font-bold">ML Area:</span>
                    <select
                      value={mapAreaId}
                      onChange={(e) => setMapAreaId(parseInt(e.target.value, 10))}
                      className="cloud-select font-mono"
                    >
                      {availableAreas.map((a) => (
                        <option key={a} value={a}>
                          Area {a}
                        </option>
                      ))}
                    </select>
                    {loadingMlHotspots && (
                      <span className="text-purple-700 font-bold animate-pulse flex items-center gap-1">
                        <RefreshCw className="w-3 h-3 animate-spin" /> Loading ML Hotspots…
                      </span>
                    )}
                    {!loadingMlHotspots && mlMapMeta && (
                      <span className="pill-sky font-mono">
                        Area {mapAreaId}: {mlMapMeta.total_matching_clusters} ML hotspots (
                        {mlMapMeta.cross_boundary_clusters_count} cross-boundary)
                      </span>
                    )}
                    {mlMapError && (
                      <span className="text-red-600 font-medium">{mlMapError}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Map Canvas */}
              <div className="cloud-card p-2 flex-1 min-h-[580px] rounded-2xl overflow-hidden relative shadow-sm">
                <CrimeMap
                  clusters={clusters}
                  mlClusters={mlMapHotspots}
                  kdeSurface={kdeSurface}
                  incidents={mapIncidents}
                  showIncidents={showIncidentsLayer}
                  showSpatialClusters={showSpatialClustersLayer}
                  showMlHotspots={showMlHotspotsLayer}
                  showKdeSurface={showKdeLayer}
                  selectedAreaId={mapAreaId}
                  filterClusterId={clusterFilter}
                  onSelectCluster={setSelectedCluster}
                />
              </div>

              {/* Selected Cluster Info Card */}
              {selectedCluster && (
                <div className="cloud-card p-4 flex items-center justify-between border-sky-300 bg-sky-50">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-sky-900">
                        Cluster #{selectedCluster.cluster_id} Selected
                      </span>
                      <span className="pill-sky font-bold">Community Area {selectedCluster.dominant_community_area}</span>
                    </div>
                    <p className="text-xs text-slate-700">
                      Centroid: {selectedCluster.centroid_latitude.toFixed(4)}°N, {selectedCluster.centroid_longitude.toFixed(4)}°W &middot; Incidents: {selectedCluster.incident_count.toLocaleString()} ({selectedCluster.percentage_of_total}%) &middot; Dominant Crime: {selectedCluster.primary_crime_type}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedCluster(null)}
                    className="px-3.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded-lg text-xs font-bold border border-slate-300 transition"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ════════════════ TAB 3: HOTSPOT ANALYSIS (ORGANIZED SUB-TABS) ════════════════ */}
          {activeTab === "hotspots" && (
            <div className="max-w-6xl mx-auto space-y-6 pb-12">
              {/* Segmented Sub-Navigation */}
              <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setHotspotSubTab("clusters")}
                    className={`px-4 py-2.5 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "clusters"
                        ? "bg-sky-600 text-white shadow-sm"
                        : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    Cluster Catalog ({clusters.length} Hotspots)
                  </button>
                  <button
                    onClick={() => setHotspotSubTab("experiments")}
                    className={`px-4 py-2.5 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "experiments"
                        ? "bg-sky-600 text-white shadow-sm"
                        : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    Parameter Sensitivity ({experiments.length} Tests)
                  </button>
                  <button
                    onClick={() => setHotspotSubTab("params")}
                    className={`px-4 py-2.5 rounded-xl text-xs font-bold transition-all ${
                      hotspotSubTab === "params"
                        ? "bg-sky-600 text-white shadow-sm"
                        : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    DBSCAN Math & Architecture
                  </button>
                </div>

                {hotspotSubTab === "clusters" && (
                  <div className="relative w-64">
                    <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3.5" />
                    <input
                      type="text"
                      placeholder="Search cluster, crime, area…"
                      value={clusterSearchQuery}
                      onChange={(e) => setClusterSearchQuery(e.target.value)}
                      className="cloud-input pl-8 py-2 text-xs w-full"
                    />
                  </div>
                )}
              </div>

              {/* Sub-Tab 1: Clusters Catalog */}
              {hotspotSubTab === "clusters" && (
                <div className="cloud-card p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-slate-900">
                        Identified Hotspot Clusters ({filteredClusters.length})
                      </h3>
                      <p className="text-xs text-slate-500">
                        Dominant crimes, geographic coordinates, and historical volume
                      </p>
                    </div>
                    <span className="pill-sky font-bold">
                      &epsilon;={params.eps_meters}m &middot; min_samples={params.min_samples}
                    </span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-200">
                    <table className="cloud-table">
                      <thead>
                        <tr>
                          <th>Cluster ID</th>
                          <th>Incident Count</th>
                          <th>Share %</th>
                          <th>Centroid (Lat, Lon)</th>
                          <th>Dominant Crime</th>
                          <th>Community Area</th>
                          <th>Bounding Coordinates</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono">
                        {filteredClusters.map((c: any) => (
                          <tr key={c.cluster_id}>
                            <td className="font-bold text-red-600">#{c.cluster_id}</td>
                            <td className="font-bold text-slate-900">{c.incident_count.toLocaleString()}</td>
                            <td className="text-sky-700 font-bold">{c.percentage_of_total}%</td>
                            <td>
                              {c.centroid_latitude.toFixed(4)}°N, {c.centroid_longitude.toFixed(4)}°W
                            </td>
                            <td className="font-sans font-bold text-slate-800">
                              {c.primary_crime_type}
                            </td>
                            <td className="font-sans text-slate-700">
                              Area {c.dominant_community_area}
                            </td>
                            <td className="text-[11px] text-slate-500">
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
                <div className="cloud-card p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-slate-900">
                        Hyperparameter Sensitivity Experiments
                      </h3>
                      <p className="text-xs text-slate-500">
                        Tested combinations evaluating radius (&epsilon;) vs density threshold (min_samples)
                      </p>
                    </div>
                    <span className="pill-cloudy font-bold">{experiments.length} Total Runs</span>
                  </div>

                  <div className="overflow-x-auto rounded-xl border border-slate-200">
                    <table className="cloud-table">
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
                              className={isSelected ? "bg-sky-50 font-bold" : ""}
                            >
                              <td>{exp.eps_meters}m</td>
                              <td>{exp.min_samples}</td>
                              <td className="text-sky-700 font-bold">{exp.num_clusters}</td>
                              <td>{exp.clustered_points.toLocaleString()}</td>
                              <td>{exp.clustered_percentage}%</td>
                              <td className="text-amber-700">{exp.noise_percentage}%</td>
                              <td>{exp.mean_cluster_size}</td>
                              <td>
                                {isSelected ? (
                                  <span className="pill-sky font-bold">SELECTED</span>
                                ) : (
                                  <span className="text-slate-400 text-[11px]">Evaluation</span>
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
                <div className="cloud-card p-6 space-y-6">
                  <div>
                    <h3 className="text-base font-bold text-slate-900">
                      DBSCAN Algorithmic Architecture
                    </h3>
                    <p className="text-xs text-slate-500 mt-1">
                      Mathematical foundation of density-based clustering using spherical radians
                    </p>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Neighborhood Radius (&epsilon;)</span>
                      <div className="text-lg font-bold text-red-600 font-mono">
                        {params.eps_meters} meters
                      </div>
                      <span className="text-[11px] text-slate-400 block">&epsilon;_rad &approx; 7.85 &times; 10&minus;5</span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Min Samples (min_samples)</span>
                      <div className="text-lg font-bold text-red-600 font-mono">
                        {params.min_samples} incidents
                      </div>
                      <span className="text-[11px] text-slate-400 block">Density threshold</span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Earth Model</span>
                      <div className="text-lg font-bold text-sky-700 font-mono">
                        6,371.0088 km
                      </div>
                      <span className="text-[11px] text-slate-400 block">IUGG Volumetric Mean</span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Distance Metric</span>
                      <div className="text-lg font-bold text-sky-700 font-mono">
                        Haversine
                      </div>
                      <span className="text-[11px] text-slate-400 block">Spherical Radians</span>
                    </div>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed space-y-2">
                    <strong className="text-slate-900 block font-bold">Why Euclidean Distance Fails in Chicago:</strong>
                    <p>
                      At 41.88°N latitude, 1 degree of latitude measures ~111.0 km, while 1 degree of longitude measures only ~82.7 km. A planar Euclidean distance metric distorts Chicago distances by over <strong>34%</strong>. The Haversine metric corrects this by calculating true great-circle angular distances on the sphere.
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ════════════════ TAB 4: ANALYTICS (CLEAN 2x2 CHARTS) ════════════════ */}
          {activeTab === "analytics" && (
            <div className="max-w-6xl mx-auto space-y-8 pb-12">
              {/* Summary KPIs Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="cloud-card p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase text-slate-600">
                    <Compass className="w-4 h-4 text-sky-600" /> Geographic Extent
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-700 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Latitude Span:</span>
                      <span>{eda.geographic_extent.min_latitude.toFixed(3)}°N – {eda.geographic_extent.max_latitude.toFixed(3)}°N</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Longitude Span:</span>
                      <span>{eda.geographic_extent.min_longitude.toFixed(3)}°W – {eda.geographic_extent.max_longitude.toFixed(3)}°W</span>
                    </div>
                    <div className="flex justify-between text-sky-700 font-bold">
                      <span className="text-slate-500 font-sans">Mean Centroid:</span>
                      <span>{eda.geographic_extent.mean_latitude.toFixed(4)}°N, {eda.geographic_extent.mean_longitude.toFixed(4)}°W</span>
                    </div>
                  </div>
                </div>

                <div className="cloud-card p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase text-slate-600">
                    <Clock className="w-4 h-4 text-amber-600" /> Temporal Coverage
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-700 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Valid Timestamps:</span>
                      <span>
                        {spatial_temporal?.total_records_with_valid_timestamps?.toLocaleString() || "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Diurnal Trough:</span>
                      <span className="text-amber-700 font-bold">
                        {diurnalTrough
                          ? `${diurnalTrough.label} (${diurnalTrough.count.toLocaleString()})`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Peak Hour:</span>
                      <span>
                        {diurnalPeak
                          ? `${diurnalPeak.label} (${diurnalPeak.count.toLocaleString()})`
                          : "N/A"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="cloud-card p-5 space-y-2">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase text-slate-600">
                    <TrendingUp className="w-4 h-4 text-red-600" /> Cluster Concentration
                  </div>
                  <div className="text-xs space-y-1.5 font-mono text-slate-700 pt-1">
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Largest Hotspot:</span>
                      <span className="text-red-600 font-bold">{kpis.largest_cluster_size.toLocaleString()} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Smallest Hotspot:</span>
                      <span>{kpis.smallest_cluster_size} incidents</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500 font-sans">Median Size:</span>
                      <span>{kpis.median_cluster_size} incidents</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Macroscopic ML KDE Surface Metadata (GET /api/kde — separate from 29 spatial clusters) */}
              {kdeMeta && (
                <div className="cloud-card p-4 flex flex-wrap items-center justify-between gap-3 bg-sky-50/40 border-sky-200">
                  <div className="text-xs text-slate-700">
                    <strong className="text-slate-900">
                      Macroscopic ML KDE Density Surface (<span className="font-mono">GET /api/kde</span>):
                    </strong>{" "}
                    {kdeMeta.grid_size}&times;{kdeMeta.grid_size} continuous grid &middot;{" "}
                    {kdeMeta.bandwidth_meters}m bandwidth &middot; Target Year {kdeMeta.target_year}
                  </div>
                  <span className="pill-sky font-mono">
                    Data Source: {analyticsDataSource === "live" ? "Live API" : "Fallback"}
                  </span>
                </div>
              )}

              {/* 4 Clean Recharts Visualizations */}
              {spatial_temporal && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Diurnal Wave */}
                  <div className="cloud-card p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        24-Hour Diurnal Crime Wave
                      </h3>
                      <p className="text-xs text-slate-500">
                        Hourly incident frequency across Chicago
                      </p>
                    </div>
                    <div className="h-[260px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={spatial_temporal.hourly_distribution}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="label" stroke="#64748b" fontSize={10} interval={3} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              borderColor: "#e2e8f0",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#0f172a",
                              boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="count"
                            name="Incidents"
                            stroke="#0284c7"
                            fill="#bae6fd"
                            fillOpacity={0.4}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Daytime vs Nighttime Shifts */}
                  <div className="cloud-card p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        Daytime vs. Nighttime Hotspot Shifts
                      </h3>
                      <p className="text-xs text-slate-500">
                        Comparing Daytime (06:00–17:59) vs Nighttime (18:00–05:59)
                      </p>
                    </div>
                    <div className="h-[260px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.cluster_day_night_shifts.slice(0, 8)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="cluster_label" stroke="#64748b" fontSize={10} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              borderColor: "#e2e8f0",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#0f172a",
                              boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: "11px" }} />
                          <Bar dataKey="daytime_count" name="Daytime" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="nighttime_count" name="Nighttime" fill="#ef4444" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Time Window Breakdown */}
                  <div className="cloud-card p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        6-Hour Activity Windows
                      </h3>
                      <p className="text-xs text-slate-500">
                        Incident volume across 4 standardized operational windows
                      </p>
                    </div>
                    <div className="h-[250px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.time_windows}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="window" stroke="#64748b" fontSize={10} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              borderColor: "#e2e8f0",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#0f172a",
                              boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
                            }}
                          />
                          <Bar dataKey="count" name="Incidents" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Category Hotspot Concentration */}
                  <div className="cloud-card p-6 space-y-3">
                    <div>
                      <h3 className="text-sm font-bold text-slate-900">
                        Category Hotspot Concentration
                      </h3>
                      <p className="text-xs text-slate-500">
                        Proportion inside clusters vs dispersed noise per offense
                      </p>
                    </div>
                    <div className="h-[250px] w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={spatial_temporal.category_hotspot_profiles.slice(0, 7)}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                          <XAxis dataKey="category" stroke="#64748b" fontSize={9} />
                          <YAxis stroke="#64748b" fontSize={10} />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#ffffff",
                              borderColor: "#e2e8f0",
                              borderRadius: "10px",
                              fontSize: "12px",
                              color: "#0f172a",
                              boxShadow: "0 8px 24px rgba(0,0,0,0.08)",
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: "11px" }} />
                          <Bar dataKey="clustered" stackId="a" name="In Hotspots" fill="#0284c7" />
                          <Bar dataKey="noise" stackId="a" name="Dispersed Noise" fill="#94a3b8" />
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
            <div className="max-w-4xl mx-auto space-y-6 pb-12">
              {/* Service Readiness Banner */}
              <div
                className={`p-4 rounded-xl border flex items-start gap-3 ${
                  mlServiceReady
                    ? "bg-emerald-50 border-emerald-200"
                    : "bg-amber-50 border-amber-200"
                }`}
              >
                {mlServiceReady ? (
                  <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                )}
                <div className="text-xs space-y-1">
                  <h4
                    className={`font-bold ${
                      mlServiceReady ? "text-emerald-900" : "text-amber-900"
                    }`}
                  >
                    {mlServiceReady
                      ? "Unified ML & Spatial Prediction Service Active"
                      : "ML Prediction Service Unavailable / Offline"}
                  </h4>
                  <p
                    className={`leading-relaxed ${
                      mlServiceReady ? "text-emerald-800" : "text-amber-800"
                    }`}
                  >
                    {mlServiceReady ? (
                      <>
                        Connected to{" "}
                        <strong>
                          {modelInfo?.model_name || "Random Forest Regressor (Champion)"}
                        </strong>{" "}
                        ({modelInfo?.feature_count || 34} features) covering verified historical
                        records from{" "}
                        <span className="font-mono">
                          {modelInfo?.historical_data_range?.start_date || "2020-01-31"}
                        </span>{" "}
                        to{" "}
                        <span className="font-mono">
                          {modelInfo?.historical_data_range?.end_date || "2025-12-31"}
                        </span>
                        . Future dates require 30 continuous days of observed history.
                      </>
                    ) : (
                      <>
                        Unable to verify <span className="font-mono">/health/ready</span> on{" "}
                        <span className="font-mono">{API_BASE_URL}</span>. ML forecasting requires
                        the running FastAPI backend; spatial proximity fallback remains available
                        below.
                      </>
                    )}
                  </p>
                </div>
              </div>

              {/* ── SECTION A: SUPERVISED ML FORECAST (POST /api/predict) ── */}
              <div className="cloud-card p-6 space-y-5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-base font-bold text-slate-900">
                      Community Area Crime-Count &amp; Four-Tier Risk Forecast
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Powered by serialized Random Forest model (<span className="font-mono">POST /api/predict</span>)
                    </p>
                  </div>
                  <span className="pill-sky font-mono">
                    4-Tier Scale: Low &middot; Moderate &middot; High &middot; Very High
                  </span>
                </div>

                {/* Quick Community Area Benchmarks */}
                <div className="space-y-2">
                  <span className="text-xs font-bold text-slate-600 uppercase tracking-wider">
                    Quick Community Area Benchmarks:
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { label: "Area 8 (High Volume)", id: 8 },
                      { label: "Area 32 (Cross-Boundary Loop)", id: 32 },
                      { label: "Area 25 (West Side)", id: 25 },
                      { label: "Area 43 (South Shore)", id: 43 },
                      { label: "Area 9 (Low Volume)", id: 9 },
                    ].map((b) => (
                      <button
                        key={b.id}
                        type="button"
                        onClick={() => setMlAreaId(b.id)}
                        className={`px-3.5 py-1.5 rounded-lg text-xs font-bold border transition ${
                          mlAreaId === b.id
                            ? "bg-sky-600 text-white border-sky-600"
                            : "bg-slate-100 hover:bg-sky-100 hover:text-sky-800 text-slate-700 border-slate-200"
                        }`}
                      >
                        {b.label}
                      </button>
                    ))}
                  </div>
                </div>

                <form onSubmit={handleMlPredict} className="space-y-4 pt-1">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Community Area (1–77)
                      </label>
                      <select
                        value={mlAreaId}
                        onChange={(e) => setMlAreaId(parseInt(e.target.value, 10))}
                        className="cloud-input w-full font-mono"
                      >
                        {availableAreas.map((areaNum) => (
                          <option key={areaNum} value={areaNum}>
                            Community Area {areaNum}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Evaluation Date (Historical: 2020-01-31 to 2025-12-31)
                      </label>
                      <input
                        type="date"
                        value={mlDate}
                        onChange={(e) => setMlDate(e.target.value)}
                        className="cloud-input w-full font-mono"
                        required
                      />
                    </div>
                  </div>

                  <label className="flex items-center gap-2 text-xs text-slate-700 font-medium cursor-pointer">
                    <input
                      type="checkbox"
                      checked={mlIncludeHotspots}
                      onChange={(e) => setMlIncludeHotspots(e.target.checked)}
                      className="rounded border-slate-300 text-sky-600"
                    />
                    Include localized cross-boundary DBSCAN hotspot clusters for this area
                  </label>

                  <button
                    type="submit"
                    disabled={isMlPredicting}
                    className="w-full py-3.5 px-4 rounded-xl bg-sky-600 hover:bg-sky-500 disabled:opacity-60 text-white font-bold text-xs tracking-wide shadow-md shadow-sky-600/20 transition flex items-center justify-center gap-2"
                  >
                    {isMlPredicting ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" /> Running ML Prediction…
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" /> Predict Community Area Crime Risk
                      </>
                    )}
                  </button>
                </form>

                {/* Error Feedback */}
                {mlPredError && (
                  <div className="p-4 rounded-xl bg-red-50 border border-red-200 flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                    <div className="text-xs space-y-1">
                      <h4 className="font-bold text-red-900">Prediction Request Rejected</h4>
                      <p className="text-red-800 leading-relaxed">{mlPredError}</p>
                    </div>
                  </div>
                )}

                {/* ML Prediction Result Card */}
                {mlPredResult && (
                  <div
                    className={`p-5 rounded-2xl border space-y-4 ${
                      RISK_TIER_STYLES[mlPredResult.prediction.risk_tier]?.bg || "bg-sky-50/50"
                    } ${
                      RISK_TIER_STYLES[mlPredResult.prediction.risk_tier]?.border || "border-sky-200"
                    }`}
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200/80">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-sky-600" />
                        <div>
                          <h4 className="font-bold text-slate-900 text-sm">
                            Community Area {mlPredResult.query.community_area} &middot;{" "}
                            <span className="font-mono">{mlPredResult.query.date}</span>
                          </h4>
                          <span className="text-[11px] text-slate-500">
                            {mlPredResult.model_info?.model_name} (
                            {mlPredResult.query.is_historical
                              ? "Verified Historical Feature Store"
                              : "Dynamic 30-Day History Forecast"}
                            )
                          </span>
                        </div>
                      </div>
                      <span
                        className={`px-3 py-1 rounded-full text-xs font-extrabold border ${
                          RISK_TIER_STYLES[mlPredResult.prediction.risk_tier]?.badge ||
                          "bg-slate-100 text-slate-800 border-slate-300"
                        }`}
                      >
                        Risk Tier: {mlPredResult.prediction.risk_tier}
                      </span>
                    </div>

                    {/* Core Metrics Grid */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                        <span className="text-xs text-slate-500 font-medium">
                          Predicted Crime Count
                        </span>
                        <div className="font-extrabold text-slate-900 font-mono text-lg">
                          {mlPredResult.prediction.predicted_crime_count.toFixed(2)} incidents
                        </div>
                      </div>

                      <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                        <span className="text-xs text-slate-500 font-medium">
                          Historical Actual Count
                        </span>
                        <div className="font-extrabold text-slate-900 font-mono text-lg">
                          {mlPredResult.context.historical_actual_count !== null
                            ? `${mlPredResult.context.historical_actual_count} incidents`
                            : "N/A (Future)"}
                        </div>
                      </div>

                      <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                        <span className="text-xs text-slate-500 font-medium">
                          Prediction Error (Pred - Actual)
                        </span>
                        <div className="font-extrabold text-sky-700 font-mono text-lg">
                          {mlPredResult.context.prediction_error !== null
                            ? `${mlPredResult.context.prediction_error >= 0 ? "+" : ""}${mlPredResult.context.prediction_error.toFixed(2)}`
                            : "N/A"}
                        </div>
                      </div>
                    </div>

                    {/* 4-Tier Classification Scale */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {[
                        {
                          name: "Low",
                          rule: `≤ ${mlPredResult.prediction.thresholds_used.low}`,
                        },
                        {
                          name: "Moderate",
                          rule: `${mlPredResult.prediction.thresholds_used.low} – ${mlPredResult.prediction.thresholds_used.moderate}`,
                        },
                        {
                          name: "High",
                          rule: `${mlPredResult.prediction.thresholds_used.moderate} – ${mlPredResult.prediction.thresholds_used.high}`,
                        },
                        {
                          name: "Very High",
                          rule: `> ${mlPredResult.prediction.thresholds_used.high}`,
                        },
                      ].map((tier) => {
                        const isActive = mlPredResult.prediction.risk_tier === tier.name;
                        return (
                          <div
                            key={tier.name}
                            className={`p-2.5 rounded-xl border text-center transition ${
                              isActive
                                ? RISK_TIER_STYLES[tier.name]?.badge + " ring-2 ring-offset-1 ring-sky-400"
                                : "bg-white/80 border-slate-200 text-slate-500"
                            }`}
                          >
                            <div className="text-xs font-extrabold">{tier.name}</div>
                            <div className="text-[11px] font-mono">{tier.rule}</div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Temporal Context */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                        <span className="text-slate-500 block">Prev Day (lag_1)</span>
                        <span className="font-bold font-mono text-slate-800">
                          {mlPredResult.context.baseline_prev_day}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                        <span className="text-slate-500 block">Prev Week (lag_7)</span>
                        <span className="font-bold font-mono text-slate-800">
                          {mlPredResult.context.baseline_prev_week}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                        <span className="text-slate-500 block">7-Day Mean</span>
                        <span className="font-bold font-mono text-slate-800">
                          {mlPredResult.context.recent_7d_mean.toFixed(2)}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                        <span className="text-slate-500 block">Citywide Lag-1</span>
                        <span className="font-bold font-mono text-slate-800">
                          {mlPredResult.context.citywide_trend_lag1}
                        </span>
                      </div>
                    </div>

                    {/* Localized Cross-Boundary Hotspots */}
                    {mlPredResult.hotspots?.clusters && (
                      <div className="p-4 rounded-xl bg-white border border-slate-200 space-y-2">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <span className="text-xs font-bold text-slate-800">
                            Localized DBSCAN Hotspots in Area {mlPredResult.query.community_area}
                          </span>
                          <span className="text-[11px] font-mono text-slate-600">
                            Matching: {mlPredResult.hotspots.total_matching_clusters} &middot;
                            Dominant: {mlPredResult.hotspots.dominant_clusters_count} &middot;
                            Cross-Boundary: {mlPredResult.hotspots.cross_boundary_clusters_count}
                          </span>
                        </div>
                        {mlPredResult.hotspots.clusters.length === 0 ? (
                          <p className="text-xs text-slate-500">
                            No dense DBSCAN hotspot clusters intersect this Community Area.
                          </p>
                        ) : (
                          <div className="space-y-1.5">
                            {mlPredResult.hotspots.clusters.map((c: any) => (
                              <div
                                key={c.hotspot_id}
                                className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-lg bg-slate-50 border border-slate-100 text-xs"
                              >
                                <div>
                                  <span className="font-bold text-sky-700 font-mono">
                                    Hotspot #{c.hotspot_id} (Rank {c.hotspot_rank})
                                  </span>{" "}
                                  <span className="text-slate-600">
                                    &middot; {c.top_crime_types}
                                  </span>
                                </div>
                                <div className="font-mono text-slate-700">
                                  <strong>{c.incident_count_in_area.toLocaleString()}</strong> in area
                                  {" / "}
                                  {c.incident_count_total.toLocaleString()} total (
                                  {c.radius_meters.toFixed(0)}m)
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── SECTION B: SPATIAL PROXIMITY & DB HISTORY (POST /api/predict-risk) ── */}
              <div className="cloud-card p-6 space-y-5">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Evaluate Spatial Proximity Exposure
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Select a Chicago benchmark location or enter custom latitude/longitude coordinates
                    (<span className="font-mono">POST /api/predict-risk</span>)
                  </p>
                </div>

                {/* Quick Presets */}
                <div className="space-y-2">
                  <span className="text-xs font-bold text-slate-600 uppercase tracking-wider">
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
                        className="px-3.5 py-1.5 rounded-lg bg-slate-100 hover:bg-sky-100 hover:text-sky-800 text-xs font-bold text-slate-700 border border-slate-200 transition"
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>

                <form onSubmit={handlePredict} className="space-y-4 pt-2">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Latitude (°N)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={predLat}
                        onChange={(e) => setPredLat(parseFloat(e.target.value))}
                        className="cloud-input w-full font-mono"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Longitude (°W)
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={predLon}
                        onChange={(e) => setPredLon(parseFloat(e.target.value))}
                        className="cloud-input w-full font-mono"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Evaluation Date
                      </label>
                      <input
                        type="date"
                        value={predDate}
                        onChange={(e) => setPredDate(e.target.value)}
                        className="cloud-input w-full font-mono"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-bold text-slate-700 mb-1">
                        Evaluation Time
                      </label>
                      <input
                        type="time"
                        value={predTime}
                        onChange={(e) => setPredTime(e.target.value)}
                        className="cloud-input w-full font-mono"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={isPredicting}
                    className="w-full py-3.5 px-4 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs tracking-wide shadow-md shadow-sky-600/20 transition flex items-center justify-center gap-2"
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
                <div className="cloud-card p-6 space-y-4 border-sky-300 bg-sky-50/50">
                  <div className="flex items-center justify-between pb-3 border-b border-sky-200">
                    <div className="flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-sky-600" />
                      <h4 className="font-bold text-slate-900 text-sm">
                        Spatial Analysis Exposure Result
                      </h4>
                    </div>
                    <span className="pill-sky font-mono font-bold">
                      Status: {predResult.status}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Spatial Risk Tier</span>
                      <div className="font-extrabold text-red-600 text-sm">
                        {predResult.provisional_spatial_risk_tier}
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Distance to Hotspot</span>
                      <div className="font-extrabold text-slate-900 font-mono text-sm">
                        {predResult.distance_to_hotspot_km} km
                      </div>
                    </div>

                    <div className="p-3.5 rounded-xl bg-white border border-slate-200 space-y-1">
                      <span className="text-xs text-slate-500 font-medium">Nearest Hotspot Cluster</span>
                      <div className="font-bold text-sky-700 font-mono text-sm">
                        Cluster #{predResult.nearest_hotspot.cluster_id} ({predResult.nearest_hotspot.incident_count} pts)
                      </div>
                    </div>
                  </div>

                  <div className="p-3.5 rounded-xl bg-white text-xs text-slate-600 leading-relaxed border border-slate-200">
                    <strong className="text-slate-800">Methodological Note: </strong>
                    {predResult.disclaimer}
                  </div>
                </div>
              )}

              {/* Persisted History Table */}
              <div className="cloud-card p-6 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <History className="w-4 h-4 text-emerald-600" />
                    <h3 className="text-sm font-bold text-slate-900">
                      Persisted Prediction &amp; Spatial Query History
                    </h3>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {(
                      [
                        { id: "all", label: "All" },
                        { id: "ml_forecast", label: "ML Forecasts" },
                        { id: "spatial_proximity", label: "Spatial Proximity" },
                      ] as const
                    ).map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => {
                          setHistoryTypeFilter(opt.id);
                          fetchPredictionHistory(opt.id);
                        }}
                        className={`px-2.5 py-1 rounded-lg text-[11px] font-bold border transition ${
                          historyTypeFilter === opt.id
                            ? "bg-sky-600 text-white border-sky-600"
                            : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                    <span className="pill-sky">DB: {historyBackend}</span>
                  </div>
                </div>

                {predHistory.length === 0 ? (
                  <p className="text-xs text-slate-500">
                    No queries recorded yet for this filter. Submit an ML forecast or coordinate query above to persist it into the database.
                  </p>
                ) : (
                  <div className="overflow-x-auto rounded-xl border border-slate-200">
                    <table className="cloud-table">
                      <thead>
                        <tr>
                          <th>ID</th>
                          <th>Method</th>
                          <th>Target Scope</th>
                          <th>Forecast / Proximity</th>
                          <th>Risk Tier</th>
                          <th>Target Date / Time</th>
                        </tr>
                      </thead>
                      <tbody className="font-mono text-xs">
                        {predHistory.map((h: any) => {
                          const isMl = h.prediction_type === "ml_forecast";
                          return (
                            <tr key={h.id}>
                              <td className="text-slate-500">#{h.id}</td>
                              <td className="font-sans">
                                {isMl ? (
                                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-sky-100 text-sky-800 border border-sky-200">
                                    ML Forecast
                                  </span>
                                ) : (
                                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                                    Spatial Proximity
                                  </span>
                                )}
                              </td>
                              <td>
                                {isMl
                                  ? `Community Area ${h.community_area}`
                                  : h.latitude !== null && h.longitude !== null
                                  ? `${h.latitude.toFixed(4)}°N, ${h.longitude.toFixed(4)}°W`
                                  : "N/A"}
                              </td>
                              <td className="font-bold">
                                {isMl ? (
                                  <span className="text-sky-700">
                                    {h.predicted_crime_count !== null
                                      ? `${h.predicted_crime_count.toFixed(2)} crimes`
                                      : "N/A"}
                                    {h.historical_actual_count !== null && (
                                      <span className="text-slate-500 font-normal">
                                        {" "}
                                        (actual: {h.historical_actual_count})
                                      </span>
                                    )}
                                  </span>
                                ) : (
                                  <span className="text-red-600">
                                    {h.nearest_cluster_id !== null
                                      ? `Cluster #${h.nearest_cluster_id}`
                                      : "N/A"}
                                    {h.distance_to_hotspot_km !== null && (
                                      <span className="text-slate-600 font-normal">
                                        {" "}
                                        ({h.distance_to_hotspot_km} km)
                                      </span>
                                    )}
                                  </span>
                                )}
                              </td>
                              <td className="font-sans text-slate-800 font-semibold">
                                {h.risk_tier}
                              </td>
                              <td className="text-slate-500">{h.target_timestamp}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ════════════════ TAB 6: METHODOLOGY ════════════════ */}
          {activeTab === "methodology" && (
            <div className="max-w-4xl mx-auto space-y-6 pb-12">
              <div className="cloud-card p-6 space-y-6">
                <div className="flex items-center gap-3 pb-4 border-b border-slate-200">
                  <span className="p-2.5 rounded-xl bg-sky-100 text-sky-600">
                    <BookOpen className="w-5 h-5" />
                  </span>
                  <div>
                    <h3 className="text-base font-bold text-slate-900">
                      Geospatial Methodology & Ethical Safeguards
                    </h3>
                    <p className="text-xs text-slate-500">
                      Mathematical formulations, hyperparameter selection criteria, and fairness guarantees
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <h4 className="text-xs font-bold text-sky-700 uppercase tracking-wider">
                    1. Spherical Haversine Metric
                  </h4>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    Treating geographic degrees as Cartesian coordinates in Chicago introduces an error exceeding <strong>34%</strong>. To eliminate distortion, coordinates are converted to radians and distances computed using the great-circle Haversine formula:
                  </p>
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 font-mono text-xs text-sky-800 overflow-x-auto">
                    d = 2R &middot; arcsin(&radic;(sin&sup2;(&Delta;&phi;/2) + cos(&phi;&sub1;)cos(&phi;&sub2;)sin&sup2;(&Delta;&lambda;/2)))
                  </div>
                  <p className="text-xs text-slate-500">
                    Where R = 6,371.0088 km (standard IUGG mean Earth radius).
                  </p>
                </div>

                <div className="space-y-3 pt-4 border-t border-slate-200">
                  <h4 className="text-xs font-bold text-sky-700 uppercase tracking-wider">
                    2. Retrospective Hotspots vs. Predictive Models
                  </h4>
                  <p className="text-xs text-slate-700 leading-relaxed">
                    DBSCAN clusters reflect <em>where crime historically concentrated</em>. They do not constitute automated predictive forecasts. Full predictive modeling involves supervised regression models (Member 1, Lina) integrating weather, temporal lags, and calendar dynamics.
                  </p>
                </div>

                <div className="space-y-3 pt-4 border-t border-slate-200">
                  <h4 className="text-xs font-bold text-red-600 uppercase tracking-wider">
                    3. Ethical Safeguards & Fairness Controls
                  </h4>
                  <ul className="space-y-2 text-xs text-slate-700">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                      <span><strong>No Demographic Attributes:</strong> Race, ethnicity, income, and personal identifiers are strictly excluded from all models.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
                      <span><strong>Leakage Prevention:</strong> Post-arrest fields (`Arrest`, `IUCR`, `FBI Code`) are quarantined and barred from feature pipelines.</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
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
