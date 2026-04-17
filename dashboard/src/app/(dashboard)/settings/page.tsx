"use client";

import {
  useState, useCallback, useEffect, useRef, Fragment,
} from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import {
  Activity, AlertTriangle, Check, ChevronDown, ChevronRight,
  Cpu, Database, Eye, EyeOff, FileText, Globe, Loader2,
  Lock, Mail, Plus, RefreshCw, Save, Settings2,
  Shield, Trash2, User, UserPlus, Users, Webhook,
  WifiOff, Zap, Server,
} from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";
import { cn, fmtDatetime } from "@/lib/utils";
import { Button }   from "@/components/ui/button";
import { Input }    from "@/components/ui/input";
import { Badge }    from "@/components/ui/badge";

// ── Types ─────────────────────────────────────────────────────────────────────

interface KafkaConfig {
  broker_url:           string;
  schema_registry_url:  string;
  sasl_username?:       string;
  sasl_password?:       string;
  security_protocol:    string;
  local_broker_url?:    string;
}

interface TopicInfo {
  name:        string;
  partitions:  number;
  retention_h: number;
  lag:         number;
  healthy:     boolean;
}

interface Thresholds {
  min_presence_pct:   number;
  grace_minutes:      number;
  early_exit_minutes: number;
  min_detections:     number;
  dept_overrides:     DeptOverride[];
}

interface DeptOverride {
  department:         string;
  min_presence_pct?:  number | null;
  grace_minutes?:     number | null;
  early_exit_minutes?:number | null;
  min_detections?:    number | null;
}

interface AIPTZSettings {
  confidence_floor:       number;
  liveness_threshold:     number;
  template_alpha:         number;
  drift_limit:            number;
  max_hunts_per_cell:     number;
  zoom_budget_s:          number;
  dbscan_eps_deg:         number;
  path_replan_s:          number;
  settle_ms:              number;
  target_inter_ocular_px: number;
}

interface ModelVersion {
  id:           string;
  name:         string;
  type:         string;
  version:      string;
  trt_engine:   boolean;
  last_updated: string;
  status:       "LOADED" | "LOADING" | "ERROR";
}

interface ERPAdapter {
  id:          string;
  name:        string;
  type:        string;
  enabled:     boolean;
  status:      "CONNECTED" | "DISCONNECTED" | "ERROR" | "UNCONFIGURED";
  last_sync?:  string;
  error_msg?:  string;
  config:      Record<string, string>;
}

interface Viewer {
  user_id:    string;
  name:       string;
  email:      string;
  status:     "ACTIVE" | "SUSPENDED" | "INACTIVE";
  last_login: string | null;
  created_at: string;
}

// ── Shared helpers ────────────────────────────────────────────────────────────

function Section({
  title, icon: Icon, children, action,
}: {
  title:     string;
  icon?:     React.ComponentType<{ className?: string }>;
  children:  React.ReactNode;
  action?:   React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-zinc-800 bg-zinc-900/40">
        {Icon && <Icon className="h-4 w-4 text-zinc-500" />}
        <h3 className="text-sm font-semibold text-zinc-300">{title}</h3>
        {action && <div className="ml-auto">{action}</div>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function SliderField({
  label, value, min, max, step = 1, unit = "", readOnly = false,
  onChange, hint,
}: {
  label:     string;
  value:     number;
  min:       number;
  max:       number;
  step?:     number;
  unit?:     string;
  readOnly?: boolean;
  onChange:  (v: number) => void;
  hint?:     string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-zinc-300">{label}</label>
        <span className="text-sm font-bold text-blue-400 tabular-nums min-w-[52px] text-right">
          {step < 1 ? value.toFixed(2) : value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={readOnly}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-blue-600 disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          background: `linear-gradient(to right, #2563eb ${((value - min) / (max - min)) * 100}%, #3f3f46 ${((value - min) / (max - min)) * 100}%)`,
        }}
      />
      <div className="flex justify-between text-[9px] text-zinc-600 font-mono">
        <span>{min}{unit}</span>
        {hint && <span className="text-zinc-500">{hint}</span>}
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

function SaveBar({
  dirty, saving, onSave, onReset,
}: {
  dirty:   boolean;
  saving:  boolean;
  onSave:  () => void;
  onReset: () => void;
}) {
  if (!dirty) return null;
  return (
    <div className="sticky bottom-0 z-10 flex items-center justify-between px-4 py-3 bg-zinc-900 border-t border-zinc-700 rounded-b-xl mt-4">
      <p className="text-xs text-amber-400 flex items-center gap-1.5">
        <AlertTriangle className="h-3.5 w-3.5" /> Unsaved changes
      </p>
      <div className="flex gap-2">
        <Button variant="ghost" size="sm" onClick={onReset} className="h-8 text-xs">Discard</Button>
        <Button size="sm" onClick={onSave} disabled={saving} className="h-8 text-xs gap-1.5 bg-blue-600 hover:bg-blue-700">
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
          Save Changes
        </Button>
      </div>
    </div>
  );
}

// ── Kafka tab ─────────────────────────────────────────────────────────────────

function KafkaTab({ readOnly }: { readOnly: boolean }) {
  const [cfg,     setCfg]     = useState<KafkaConfig | null>(null);
  const [dirty,   setDirty]   = useState(false);
  const [saving,  setSaving]  = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [showPw,  setShowPw]  = useState(false);

  const { data: remote } = useQuery({
    queryKey: ["kafka-config"],
    queryFn:  () => api.get<KafkaConfig>("/api/kafka/config").then(r => r.data),
    staleTime: 60_000,
  });

  useEffect(() => {
    if (remote && !dirty) setCfg({ ...remote, security_protocol: remote.security_protocol ?? "PLAINTEXT" });
  }, [remote, dirty]);

  const { data: topics } = useQuery({
    queryKey:        ["kafka-topics"],
    queryFn:         () => api.get<{ topics: TopicInfo[] }>("/api/kafka/topics").then(r => r.data.topics ?? []),
    refetchInterval: 30_000,
    staleTime:       10_000,
  });

  const update = useCallback((key: keyof KafkaConfig, val: string) => {
    setCfg(prev => prev ? { ...prev, [key]: val } : prev);
    setDirty(true);
    setTestResult(null);
  }, []);

  const handleSave = async () => {
    if (!cfg) return;
    setSaving(true);
    try {
      await api.put("/api/kafka/config", cfg);
      setDirty(false);
      toast.success("Kafka config saved");
    } catch { toast.error("Failed to save"); }
    finally { setSaving(false); }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await api.post<{ ok: boolean; message: string }>("/api/kafka/test");
      setTestResult({ ok: res.data.ok, msg: res.data.message });
    } catch { setTestResult({ ok: false, msg: "Connection failed" }); }
    finally { setTesting(false); }
  };

  if (!cfg) return <div className="flex items-center justify-center h-40 text-zinc-600 text-sm"><Loader2 className="h-4 w-4 animate-spin mr-2" />Loading…</div>;

  const Field = ({ label, k, type = "text" }: { label: string; k: keyof KafkaConfig; type?: string }) => (
    <div>
      <label className="text-xs font-medium text-zinc-400 block mb-1">{label}</label>
      <Input
        type={type === "password" && !showPw ? "password" : "text"}
        value={String(cfg[k] ?? "")}
        onChange={e => update(k, e.target.value)}
        disabled={readOnly}
        className="bg-zinc-900 border-zinc-700 text-sm h-9"
        placeholder={readOnly ? "••••••••" : undefined}
      />
    </div>
  );

  return (
    <div className="space-y-4">
      <Section title="Connection" icon={Database}
        action={
          <Button size="sm" variant="outline" onClick={handleTest} disabled={testing} className="h-7 text-xs gap-1.5">
            {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
            Test Connection
          </Button>
        }
      >
        {testResult && (
          <div className={cn("flex items-center gap-2 p-2.5 rounded-lg text-xs mb-3 border",
            testResult.ok ? "bg-green-900/20 border-green-800 text-green-400" : "bg-red-900/20 border-red-800 text-red-400"
          )}>
            {testResult.ok ? <Check className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
            {testResult.msg}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Field label="Broker URL" k="broker_url" />
          <Field label="Schema Registry URL" k="schema_registry_url" />
          <Field label="Local Broker URL (optional)" k="local_broker_url" />
          <div>
            <label className="text-xs font-medium text-zinc-400 block mb-1">Security Protocol</label>
            <select
              value={cfg.security_protocol}
              onChange={e => update("security_protocol", e.target.value)}
              disabled={readOnly}
              className="w-full h-9 px-3 rounded-md border border-zinc-700 bg-zinc-900 text-sm text-zinc-200"
            >
              {["PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"].map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          {(cfg.security_protocol ?? "").startsWith("SASL") && (
            <>
              <Field label="SASL Username" k="sasl_username" />
              <div>
                <label className="text-xs font-medium text-zinc-400 block mb-1">SASL Password</label>
                <div className="relative">
                  <Input
                    type={showPw ? "text" : "password"}
                    value={cfg.sasl_password ?? ""}
                    onChange={e => update("sasl_password", e.target.value)}
                    disabled={readOnly}
                    className="bg-zinc-900 border-zinc-700 text-sm h-9 pr-9"
                  />
                  <button className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300" onClick={() => setShowPw(p => !p)}>
                    {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
        {readOnly && <p className="text-xs text-zinc-600 mt-3 flex items-center gap-1"><Lock className="h-3 w-3" /> Read-only — contact Super Admin to modify Kafka settings</p>}
      </Section>

      <Section title="Topics" icon={Activity}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800">
                {["Topic", "Partitions", "Retention", "Lag", "Health"].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-zinc-500 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(topics ?? []).map(t => (
                <tr key={t.name} className="border-b border-zinc-800/50 hover:bg-zinc-900/30">
                  <td className="px-3 py-2.5 font-mono text-zinc-300 text-[10px]">{t.name}</td>
                  <td className="px-3 py-2.5 text-zinc-400">{t.partitions}</td>
                  <td className="px-3 py-2.5 text-zinc-400">{t.retention_h < 24 ? `${t.retention_h}h` : `${t.retention_h / 24}d`}</td>
                  <td className={cn("px-3 py-2.5 font-mono font-bold tabular-nums",
                    t.lag === 0 ? "text-zinc-500" : t.lag < 100 ? "text-amber-400" : "text-red-400"
                  )}>
                    {t.lag}
                  </td>
                  <td className="px-3 py-2.5">
                    <span className={cn("inline-flex items-center gap-1 text-[10px] font-semibold px-1.5 py-0.5 rounded",
                      t.healthy ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"
                    )}>
                      <span className={cn("h-1.5 w-1.5 rounded-full", t.healthy ? "bg-green-400" : "bg-red-400")} />
                      {t.healthy ? "OK" : "ERR"}
                    </span>
                  </td>
                </tr>
              ))}
              {!topics?.length && (
                <tr><td colSpan={5} className="px-3 py-6 text-center text-zinc-600">No topics found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Section>

      {!readOnly && <SaveBar dirty={dirty} saving={saving} onSave={handleSave} onReset={() => { setCfg(remote!); setDirty(false); }} />}
    </div>
  );
}

// ── Thresholds tab ────────────────────────────────────────────────────────────

const THRESHOLD_DEFAULTS: Thresholds = {
  min_presence_pct:   75,
  grace_minutes:      10,
  early_exit_minutes: 15,
  min_detections:     3,
  dept_overrides:     [],
};

function ThresholdsTab({ readOnly }: { readOnly: boolean }) {
  const [cfg,    setCfg]   = useState<Thresholds>(THRESHOLD_DEFAULTS);
  const [dirty,  setDirty] = useState(false);
  const [saving, setSaving]= useState(false);
  const [editDept, setEditDept] = useState<DeptOverride | null>(null);
  const [newDept,  setNewDept]  = useState("");

  const { data: remote } = useQuery({
    queryKey: ["settings-thresholds"],
    queryFn:  () => api.get<Thresholds>("/api/settings/thresholds").then(r => r.data),
    staleTime: 60_000,
  });
  useEffect(() => { if (remote && !dirty) setCfg(remote); }, [remote, dirty]);

  const set = (key: keyof Thresholds, val: number) => {
    setCfg(p => ({ ...p, [key]: val })); setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try { await api.put("/api/settings/thresholds", cfg); setDirty(false); toast.success("Thresholds saved"); }
    catch { toast.error("Save failed"); }
    finally { setSaving(false); }
  };

  const addOverride = () => {
    if (!newDept.trim()) return;
    setCfg(p => ({ ...p, dept_overrides: [...p.dept_overrides, { department: newDept.trim() }] }));
    setNewDept(""); setDirty(true);
  };

  const removeOverride = (dept: string) => {
    setCfg(p => ({ ...p, dept_overrides: p.dept_overrides.filter(d => d.department !== dept) }));
    setDirty(true);
  };

  const updateOverrideField = (dept: string, key: keyof DeptOverride, val: string) => {
    setCfg(p => ({
      ...p,
      dept_overrides: p.dept_overrides.map(d =>
        d.department === dept ? { ...d, [key]: val === "" ? null : Number(val) } : d
      ),
    }));
    setDirty(true);
  };

  return (
    <div className="space-y-4">
      <Section title="Global Thresholds" icon={Settings2}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
          <SliderField label="Minimum Presence" value={cfg.min_presence_pct}    min={50}  max={100} unit="%" readOnly={readOnly} onChange={v => set("min_presence_pct", v)}   hint="default 75%" />
          <SliderField label="Grace Period"      value={cfg.grace_minutes}        min={0}   max={30}  unit=" min" readOnly={readOnly} onChange={v => set("grace_minutes", v)}       hint="default 10 min" />
          <SliderField label="Early Exit Cutoff" value={cfg.early_exit_minutes}   min={5}   max={30}  unit=" min" readOnly={readOnly} onChange={v => set("early_exit_minutes", v)}  hint="default 15 min" />
          <SliderField label="Min Detections"    value={cfg.min_detections}       min={1}   max={10}             readOnly={readOnly} onChange={v => set("min_detections", v)}       hint="default 3" />
        </div>
      </Section>

      <Section title="Per-Department Overrides" icon={Users}
        action={!readOnly && (
          <div className="flex gap-2">
            <Input value={newDept} onChange={e => setNewDept(e.target.value)} placeholder="Department name…" className="h-7 text-xs w-40 bg-zinc-900 border-zinc-700" />
            <Button size="sm" variant="outline" onClick={addOverride} className="h-7 text-xs gap-1"><Plus className="h-3 w-3" />Add</Button>
          </div>
        )}
      >
        {cfg.dept_overrides.length === 0 ? (
          <p className="text-xs text-zinc-600 text-center py-4">No overrides — all departments use global defaults</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className="px-3 py-2 text-left text-zinc-500 font-semibold">Department</th>
                  <th className="px-3 py-2 text-left text-zinc-500 font-semibold">Min Presence %</th>
                  <th className="px-3 py-2 text-left text-zinc-500 font-semibold">Grace (min)</th>
                  <th className="px-3 py-2 text-left text-zinc-500 font-semibold">Early Exit (min)</th>
                  <th className="px-3 py-2 text-left text-zinc-500 font-semibold">Min Detections</th>
                  {!readOnly && <th className="px-3 py-2" />}
                </tr>
              </thead>
              <tbody>
                {cfg.dept_overrides.map(d => (
                  <tr key={d.department} className="border-b border-zinc-800/50">
                    <td className="px-3 py-2 font-semibold text-zinc-300">{d.department}</td>
                    {(["min_presence_pct", "grace_minutes", "early_exit_minutes", "min_detections"] as const).map(k => (
                      <td key={k} className="px-3 py-2">
                        {readOnly
                          ? <span className="text-zinc-400">{d[k] ?? <span className="text-zinc-600">default</span>}</span>
                          : <Input
                              type="number"
                              value={d[k] ?? ""}
                              onChange={e => updateOverrideField(d.department, k, e.target.value)}
                              placeholder="default"
                              className="h-7 text-xs w-24 bg-zinc-900 border-zinc-700"
                            />
                        }
                      </td>
                    ))}
                    {!readOnly && (
                      <td className="px-3 py-2">
                        <button onClick={() => removeOverride(d.department)} className="text-zinc-600 hover:text-red-400 transition-colors">
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {!readOnly && <SaveBar dirty={dirty} saving={saving} onSave={handleSave} onReset={() => { setCfg(remote ?? THRESHOLD_DEFAULTS); setDirty(false); }} />}
    </div>
  );
}

// ── AI & PTZ tab ──────────────────────────────────────────────────────────────

const AI_DEFAULTS: AIPTZSettings = {
  confidence_floor:       72,
  liveness_threshold:     65,
  template_alpha:         0.05,
  drift_limit:            0.15,
  max_hunts_per_cell:     3,
  zoom_budget_s:          15,
  dbscan_eps_deg:         8,
  path_replan_s:          90,
  settle_ms:              350,
  target_inter_ocular_px: 100,
};

function AIPTZTab({ readOnly }: { readOnly: boolean }) {
  const [cfg,    setCfg]   = useState<AIPTZSettings>(AI_DEFAULTS);
  const [dirty,  setDirty] = useState(false);
  const [saving, setSaving]= useState(false);

  const { data: remote } = useQuery({
    queryKey: ["settings-ai-ptz"],
    queryFn:  () => api.get<AIPTZSettings>("/api/settings/ai-ptz").then(r => r.data),
    staleTime: 60_000,
  });
  useEffect(() => { if (remote && !dirty) setCfg(remote); }, [remote, dirty]);

  const { data: nodeInfo } = useQuery({
    queryKey: ["node-info-models"],
    queryFn:  () => api.get<{ models?: ModelVersion[] }>("/api/node/info").then(r => r.data.models ?? []),
    staleTime: 120_000,
  });
  const models: ModelVersion[] = nodeInfo ?? [];

  const set = (key: keyof AIPTZSettings, val: number) => {
    setCfg(p => ({ ...p, [key]: val })); setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try { await api.put("/api/settings/ai-ptz", cfg); setDirty(false); toast.success("AI & PTZ settings saved"); }
    catch { toast.error("Save failed"); }
    finally { setSaving(false); }
  };

  const STATUS_STYLE = {
    LOADED:  "text-green-400 bg-green-900/30",
    LOADING: "text-amber-400 bg-amber-900/30",
    ERROR:   "text-red-400 bg-red-900/30",
  };

  return (
    <div className="space-y-4">
      <Section title="Recognition & Identity" icon={Shield}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
          <SliderField label="Confidence Floor"   value={cfg.confidence_floor}   min={50} max={95} unit="%" readOnly={readOnly} onChange={v => set("confidence_floor", v)}   hint="default 72%" />
          <SliderField label="Liveness Threshold" value={cfg.liveness_threshold} min={30} max={90} unit="%" readOnly={readOnly} onChange={v => set("liveness_threshold", v)} hint="default 65%" />
          <SliderField label="Template Alpha (EMA)" value={cfg.template_alpha}   min={0.01} max={0.10} step={0.01} readOnly={readOnly} onChange={v => set("template_alpha", v)} hint="default 0.05" />
          <SliderField label="Drift Guard Limit"  value={cfg.drift_limit}        min={0.05} max={0.25} step={0.01} readOnly={readOnly} onChange={v => set("drift_limit", v)}    hint="default 0.15" />
        </div>
      </Section>

      <Section title="PTZ Scan Control" icon={Cpu}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-5">
          <SliderField label="Max Face Hunts / Cell"     value={cfg.max_hunts_per_cell}      min={1}   max={8}   readOnly={readOnly} onChange={v => set("max_hunts_per_cell", v)}      hint="default 3" />
          <SliderField label="Zoom Budget"               value={cfg.zoom_budget_s}           min={5}   max={30}  unit="s"   readOnly={readOnly} onChange={v => set("zoom_budget_s", v)}           hint="default 15s" />
          <SliderField label="DBSCAN ε (cluster radius)" value={cfg.dbscan_eps_deg}          min={3}   max={20}  unit="°"   readOnly={readOnly} onChange={v => set("dbscan_eps_deg", v)}          hint="default 8°" />
          <SliderField label="Path Replan Interval"      value={cfg.path_replan_s}           min={30}  max={300} unit="s"   readOnly={readOnly} onChange={v => set("path_replan_s", v)}           hint="default 90s" />
          <SliderField label="PTZ Settle Time"           value={cfg.settle_ms}               min={100} max={800} unit=" ms" readOnly={readOnly} onChange={v => set("settle_ms", v)}               hint="default 350ms" />
          <SliderField label="Target Inter-Ocular (px)"  value={cfg.target_inter_ocular_px}  min={60}  max={150}            readOnly={readOnly} onChange={v => set("target_inter_ocular_px", v)}  hint="default 100px" />
        </div>
      </Section>

      <Section title="Model Versions" icon={Database}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800">
                {["Model", "Type", "Version", "TRT Engine", "Last Updated", "Status"].map(h => (
                  <th key={h} className="px-3 py-2 text-left text-zinc-500 font-semibold whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map(m => (
                <tr key={m.id} className="border-b border-zinc-800/50 hover:bg-zinc-900/30">
                  <td className="px-3 py-2.5 font-semibold text-zinc-200">{m.name}</td>
                  <td className="px-3 py-2.5 text-zinc-500">{m.type}</td>
                  <td className="px-3 py-2.5 font-mono text-zinc-400">{m.version}</td>
                  <td className="px-3 py-2.5">
                    {m.trt_engine
                      ? <span className="text-green-400 font-bold text-[10px]">✓ FP16</span>
                      : <span className="text-zinc-600 text-[10px]">ONNX</span>
                    }
                  </td>
                  <td className="px-3 py-2.5 text-zinc-500">{fmtDatetime(m.last_updated)}</td>
                  <td className="px-3 py-2.5">
                    <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", STATUS_STYLE[m.status])}>
                      {m.status}
                    </span>
                  </td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr><td colSpan={6} className="px-3 py-6 text-center text-zinc-600">No model data — ensure node is online</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Section>

      {!readOnly && <SaveBar dirty={dirty} saving={saving} onSave={handleSave} onReset={() => { setCfg(remote ?? AI_DEFAULTS); setDirty(false); }} />}
    </div>
  );
}

// ── ERP Adapters tab ──────────────────────────────────────────────────────────

const ERP_META = [
  {
    id: "sap", name: "SAP SuccessFactors", abbr: "SAP", color: "#0066b3",
    desc: "Sync via OData API to HR modules",
    fields: [
      { key: "api_url",      label: "API URL",      type: "url"      },
      { key: "client_id",    label: "Client ID",    type: "text"     },
      { key: "client_secret",label: "Client Secret",type: "password" },
      { key: "company_id",   label: "Company ID",   type: "text"     },
    ],
  },
  {
    id: "oracle", name: "Oracle PeopleSoft", abbr: "ORC", color: "#f80000",
    desc: "Integration broker via OData/REST",
    fields: [
      { key: "gateway_url",    label: "Gateway URL",      type: "url"      },
      { key: "namespace",      label: "Namespace URI",    type: "text"     },
      { key: "operation_name", label: "Service Operation",type: "text"     },
      { key: "api_key",        label: "API Key",          type: "password" },
    ],
  },
  {
    id: "rest", name: "Generic REST API", abbr: "REST", color: "#22c55e",
    desc: "POST attendance records to any REST endpoint",
    fields: [
      { key: "endpoint_url", label: "Endpoint URL",  type: "url"      },
      { key: "auth_header",  label: "Auth Header",   type: "text"     },
      { key: "api_key",      label: "API Key/Token", type: "password" },
      { key: "batch_size",   label: "Batch Size",    type: "number"   },
    ],
  },
  {
    id: "webhook", name: "Webhook", abbr: "WH", color: "#a855f7",
    desc: "Fire-and-forget POST on each attendance event",
    fields: [
      { key: "url",          label: "Webhook URL",  type: "url"  },
      { key: "secret",       label: "HMAC Secret",  type: "password" },
      { key: "retry_count",  label: "Retry Count",  type: "number"   },
    ],
  },
  {
    id: "graphql", name: "GraphQL", abbr: "GQL", color: "#e535ab",
    desc: "Mutation-based sync to GraphQL endpoint",
    fields: [
      { key: "endpoint_url", label: "GraphQL URL",  type: "url"      },
      { key: "auth_token",   label: "Auth Token",   type: "password" },
      { key: "mutation_name",label: "Mutation Name",type: "text"     },
    ],
  },
  {
    id: "sftp", name: "SFTP File Export", abbr: "FTP", color: "#f59e0b",
    desc: "Daily CSV export to SFTP server",
    fields: [
      { key: "host",     label: "Host",     type: "text"     },
      { key: "port",     label: "Port",     type: "number"   },
      { key: "username", label: "Username", type: "text"     },
      { key: "password", label: "Password", type: "password" },
      { key: "path",     label: "Remote Path", type: "text"  },
    ],
  },
] as const;

function ERPAdapterCard({
  meta,
  adapter,
  onSave,
}: {
  meta:    (typeof ERP_META)[number];
  adapter: ERPAdapter | undefined;
  onSave:  (id: string, config: Record<string, string>, enabled: boolean) => Promise<void>;
}) {
  const [expanded,  setExpanded]  = useState(false);
  const [testing,   setTesting]   = useState(false);
  const [testRes,   setTestRes]   = useState<{ ok: boolean; msg: string } | null>(null);
  const [config,    setConfig]    = useState<Record<string, string>>(adapter?.config ?? {});
  const [enabled,   setEnabled]   = useState(adapter?.enabled ?? false);
  const [saving,    setSaving]    = useState(false);
  const [showPw,    setShowPw]    = useState<Record<string, boolean>>({});
  const [logs,      setLogs]      = useState<string[]>([]);
  const [showLogs,  setShowLogs]  = useState(false);

  const status = adapter?.status ?? "UNCONFIGURED";
  const STATUS_STYLE = {
    CONNECTED:    "text-green-400 bg-green-900/30 border-green-800",
    DISCONNECTED: "text-zinc-500 bg-zinc-800 border-zinc-700",
    ERROR:        "text-red-400 bg-red-900/30 border-red-800",
    UNCONFIGURED: "text-zinc-600 bg-zinc-900 border-zinc-800",
  };

  const handleTest = async () => {
    setTesting(true); setTestRes(null);
    try {
      const res = await api.post<{ ok: boolean; message: string }>(`/api/erp/${meta.id}/test`);
      setTestRes({ ok: res.data.ok, msg: res.data.message });
    } catch { setTestRes({ ok: false, msg: "Test failed" }); }
    finally { setTesting(false); }
  };

  const handleSave = async () => {
    setSaving(true);
    try { await onSave(meta.id, config, enabled); toast.success(`${meta.name} saved`); }
    catch { toast.error("Save failed"); }
    finally { setSaving(false); }
  };

  const fetchLogs = async () => {
    try {
      const res = await api.get<{ logs: string[] }>(`/api/erp/${meta.id}/logs`);
      setLogs(res.data.logs ?? []);
    } catch { setLogs(["Failed to load logs"]); }
    setShowLogs(true);
  };

  return (
    <div className={cn("rounded-xl border transition-colors", expanded ? "border-zinc-700 bg-zinc-900" : "border-zinc-800 bg-zinc-950")}>
      {/* Header */}
      <div className="flex items-center gap-3 p-4">
        {/* Abbr badge */}
        <div className="h-10 w-10 rounded-lg flex items-center justify-center text-[10px] font-black shrink-0 text-white"
          style={{ background: meta.color }}>
          {meta.abbr}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-zinc-200">{meta.name}</p>
          <p className="text-[10px] text-zinc-500">{meta.desc}</p>
        </div>
        <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded border whitespace-nowrap", STATUS_STYLE[status])}>
          {status}
        </span>
        {/* Enable toggle */}
        <button
          onClick={() => setEnabled(p => !p)}
          className={cn("h-5 w-9 rounded-full transition-colors relative shrink-0",
            enabled ? "bg-blue-600" : "bg-zinc-700"
          )}
        >
          <span className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform",
            enabled ? "translate-x-4" : "translate-x-0.5"
          )} />
        </button>
        <button onClick={() => setExpanded(p => !p)} className="text-zinc-500 hover:text-zinc-300">
          {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
      </div>

      {/* Expanded config */}
      {expanded && (
        <div className="border-t border-zinc-800 p-4 space-y-4">
          {testRes && (
            <div className={cn("flex items-center gap-2 p-2.5 rounded-lg text-xs border",
              testRes.ok ? "bg-green-900/20 border-green-800 text-green-400" : "bg-red-900/20 border-red-800 text-red-400"
            )}>
              {testRes.ok ? <Check className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
              {testRes.msg}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {meta.fields.map(f => {
              const isPassword = f.type === "password";
              const visible    = showPw[f.key];
              return (
                <div key={f.key}>
                  <label className="text-xs font-medium text-zinc-400 block mb-1">{f.label}</label>
                  <div className="relative">
                    <Input
                      type={isPassword && !visible ? "password" : "text"}
                      value={config[f.key] ?? ""}
                      onChange={e => setConfig(p => ({ ...p, [f.key]: e.target.value }))}
                      className="bg-zinc-900 border-zinc-700 text-sm h-8 pr-8"
                    />
                    {isPassword && (
                      <button className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
                        onClick={() => setShowPw(p => ({ ...p, [f.key]: !p[f.key] }))}>
                        {visible ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-2 pt-1">
            <Button size="sm" onClick={handleSave} disabled={saving} className="h-7 text-xs gap-1 bg-blue-600 hover:bg-blue-700">
              {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
              Save
            </Button>
            <Button size="sm" variant="outline" onClick={handleTest} disabled={testing} className="h-7 text-xs gap-1">
              {testing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
              Test
            </Button>
            <Button size="sm" variant="ghost" onClick={fetchLogs} className="h-7 text-xs gap-1 text-zinc-400">
              <FileText className="h-3 w-3" /> Logs
            </Button>
            {adapter?.last_sync && (
              <span className="ml-auto text-[10px] text-zinc-600">Last sync: {fmtDatetime(adapter.last_sync)}</span>
            )}
          </div>

          {showLogs && logs.length > 0 && (
            <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-3 font-mono text-[10px] text-zinc-400 max-h-40 overflow-y-auto space-y-0.5">
              {logs.map((l, i) => <p key={i} className="leading-relaxed">{l}</p>)}
            </div>
          )}
          {showLogs && logs.length === 0 && (
            <p className="text-xs text-zinc-600 text-center py-2">No recent logs</p>
          )}
        </div>
      )}
    </div>
  );
}

function ERPTab() {
  const qc = useQueryClient();
  const { data: adapters } = useQuery({
    queryKey: ["erp-adapters"],
    queryFn:  () => api.get<ERPAdapter[]>("/api/settings/erp").then(r => r.data),
    staleTime: 120_000,
  });

  const handleSave = useCallback(async (id: string, config: Record<string, string>, enabled: boolean) => {
    await api.put(`/api/settings/erp/${id}`, { config, enabled });
    qc.invalidateQueries({ queryKey: ["erp-adapters"] });
  }, [qc]);

  return (
    <div className="space-y-3">
      {ERP_META.map(meta => (
        <ERPAdapterCard
          key={meta.id}
          meta={meta}
          adapter={adapters?.find(a => a.id === meta.id)}
          onSave={handleSave}
        />
      ))}
    </div>
  );
}

// ── Viewers tab ───────────────────────────────────────────────────────────────

function ViewersTab() {
  const qc = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [newName,    setNewName]    = useState("");
  const [newEmail,   setNewEmail]   = useState("");
  const [creating,   setCreating]   = useState(false);
  const [createdPw,  setCreatedPw]  = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["viewers"],
    queryFn:  () => api.get<{ viewers: Viewer[] }>("/api/users/viewers").then(r => r.data.viewers ?? r.data as unknown as Viewer[]),
    staleTime: 30_000,
  });
  const viewers: Viewer[] = Array.isArray(data) ? data : (data as unknown as Viewer[]) ?? [];

  const handleCreate = async () => {
    if (!newName || !newEmail) return;
    setCreating(true);
    try {
      const res = await api.post<{ user: Viewer; temp_password: string }>("/api/users/viewers", {
        name: newName, email: newEmail,
      });
      setCreatedPw(res.data.temp_password);
      setNewName(""); setNewEmail("");
      setShowCreate(false);
      qc.invalidateQueries({ queryKey: ["viewers"] });
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Create failed";
      toast.error(msg);
    }
    finally { setCreating(false); }
  };

  const toggleStatus = async (viewer: Viewer) => {
    const newStatus = viewer.status === "ACTIVE" ? "SUSPENDED" : "ACTIVE";
    try {
      await api.put(`/api/users/viewers/${viewer.user_id}/status`, { status: newStatus });
      qc.invalidateQueries({ queryKey: ["viewers"] });
      toast.success(`Viewer ${newStatus === "ACTIVE" ? "activated" : "suspended"}`);
    } catch { toast.error("Status change failed"); }
  };

  const deleteViewer = async (viewer: Viewer) => {
    if (!confirm(`Delete viewer ${viewer.name}?`)) return;
    try {
      await api.delete(`/api/users/viewers/${viewer.user_id}`);
      qc.invalidateQueries({ queryKey: ["viewers"] });
      toast.success("Viewer deleted");
    } catch { toast.error("Delete failed"); }
  };

  const STATUS_STYLE = {
    ACTIVE:    "text-green-400 bg-green-900/30",
    SUSPENDED: "text-amber-400 bg-amber-900/30",
    INACTIVE:  "text-zinc-500 bg-zinc-800",
  };

  return (
    <div className="space-y-4">
      {/* Temp password dialog */}
      {createdPw && (
        <div className="rounded-xl border border-green-800 bg-green-900/20 p-4 space-y-2">
          <p className="text-sm font-semibold text-green-400 flex items-center gap-1.5"><Check className="h-4 w-4" />Viewer created successfully</p>
          <p className="text-xs text-zinc-400">Share this temporary password — it will not be shown again:</p>
          <code className="block px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-700 text-sm font-mono text-green-300 tracking-widest">
            {createdPw}
          </code>
          <Button size="sm" variant="ghost" onClick={() => setCreatedPw(null)} className="text-xs h-7">Dismiss</Button>
        </div>
      )}

      <Section title="Viewers" icon={Users}
        action={
          <Button size="sm" onClick={() => setShowCreate(p => !p)} className="h-7 text-xs gap-1.5 bg-blue-600 hover:bg-blue-700">
            <UserPlus className="h-3.5 w-3.5" /> New Viewer
          </Button>
        }
      >
        {/* Create form */}
        {showCreate && (
          <div className="flex items-end gap-3 p-3 rounded-lg bg-zinc-900 border border-zinc-700 mb-4">
            <div className="flex-1">
              <label className="text-xs text-zinc-400 block mb-1">Full Name</label>
              <Input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Jane Smith" className="h-8 text-sm bg-zinc-800 border-zinc-700" />
            </div>
            <div className="flex-1">
              <label className="text-xs text-zinc-400 block mb-1">Email</label>
              <Input type="email" value={newEmail} onChange={e => setNewEmail(e.target.value)} placeholder="jane@acme.edu" className="h-8 text-sm bg-zinc-800 border-zinc-700" />
            </div>
            <Button onClick={handleCreate} disabled={creating || !newName || !newEmail} className="h-8 text-xs bg-blue-600 hover:bg-blue-700 gap-1.5 whitespace-nowrap">
              {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
              Create
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setShowCreate(false)} className="h-8">
              <X className="h-3.5 w-3.5 text-zinc-500" />
            </Button>
          </div>
        )}

        {isLoading && <div className="py-8 text-center text-zinc-600 text-sm"><Loader2 className="h-4 w-4 animate-spin inline mr-2" />Loading…</div>}

        {!isLoading && viewers.length === 0 && (
          <div className="py-8 text-center text-zinc-600 text-sm flex flex-col items-center gap-2">
            <Users className="h-7 w-7 opacity-30" />
            No viewers yet — add one above
          </div>
        )}

        {viewers.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800">
                  {["Name", "Email", "Status", "Last Login", "Created", ""].map(h => (
                    <th key={h} className="px-3 py-2 text-left text-zinc-500 font-semibold whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {viewers.map(v => (
                  <tr key={v.user_id} className="border-b border-zinc-800/50 hover:bg-zinc-900/30">
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-6 w-6 rounded-full bg-zinc-800 flex items-center justify-center shrink-0">
                          <User className="h-3 w-3 text-zinc-500" />
                        </div>
                        <span className="font-medium text-zinc-200">{v.name}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-zinc-400">{v.email}</td>
                    <td className="px-3 py-2.5">
                      <span className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", STATUS_STYLE[v.status])}>
                        {v.status}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-zinc-500">{v.last_login ? fmtDatetime(v.last_login) : "—"}</td>
                    <td className="px-3 py-2.5 text-zinc-600">{fmtDatetime(v.created_at)}</td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-1.5 justify-end">
                        <Button
                          size="sm" variant="ghost"
                          onClick={() => toggleStatus(v)}
                          className={cn("h-6 text-[10px] px-2 font-semibold",
                            v.status === "ACTIVE" ? "text-amber-400 hover:text-amber-300" : "text-green-400 hover:text-green-300"
                          )}
                        >
                          {v.status === "ACTIVE" ? "Suspend" : "Activate"}
                        </Button>
                        <Button
                          size="sm" variant="ghost"
                          onClick={() => deleteViewer(v)}
                          className="h-6 w-6 p-0 text-zinc-600 hover:text-red-400"
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}

// ── X button ──────────────────────────────────────────────────────────────────

function X({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

type SettingsTab = "kafka" | "thresholds" | "ai-ptz" | "erp" | "viewers";

const TABS: Array<{
  id:    SettingsTab;
  label: string;
  icon:  React.ComponentType<{ className?: string }>;
  clientAdminOnly?: boolean;
}> = [
  { id: "kafka",      label: "Kafka Pipeline",   icon: Activity  },
  { id: "thresholds", label: "Thresholds",        icon: Settings2 },
  { id: "ai-ptz",     label: "AI & PTZ",          icon: Cpu       },
  { id: "erp",        label: "ERP Adapters",      icon: Globe     },
  { id: "viewers",    label: "Manage Viewers",    icon: Users, clientAdminOnly: true },
];

export default function SettingsPage() {
  const { isViewer, isClientAdmin, isSuperAdmin } = useAuth();
  const router = useRouter();
  const [tab, setTab] = useState<SettingsTab>("kafka");

  // Redirect viewers — entire page hidden
  useEffect(() => {
    if (isViewer) router.replace("/attendance");
  }, [isViewer, router]);

  if (isViewer) return null;

  const readOnly = isClientAdmin; // Kafka tab read-only for CLIENT_ADMIN

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] overflow-hidden bg-zinc-950">
      {/* Header + tabs */}
      <div className="shrink-0 border-b border-zinc-800 px-5 pt-4 pb-0">
        <div className="flex items-center gap-2 mb-3">
          <Settings2 className="h-5 w-5 text-blue-400" />
          <h1 className="text-base font-bold text-zinc-100">Settings</h1>
          {isSuperAdmin && (
            <span className="ml-1 text-[10px] font-bold text-amber-400 bg-amber-900/30 border border-amber-800 px-1.5 py-0.5 rounded">
              SUPER ADMIN
            </span>
          )}
        </div>
        {/* Tab bar */}
        <div className="flex gap-0 -mb-px">
          {TABS
            .filter(t => !t.clientAdminOnly || isClientAdmin || isSuperAdmin)
            .map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={cn(
                  "flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors whitespace-nowrap",
                  tab === id
                    ? "border-blue-500 text-blue-400"
                    : "border-transparent text-zinc-500 hover:text-zinc-300",
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))
          }
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-5">
        <div className="max-w-[1200px] mx-auto">
          {tab === "kafka"      && <KafkaTab      readOnly={readOnly} />}
          {tab === "thresholds" && <ThresholdsTab readOnly={readOnly} />}
          {tab === "ai-ptz"     && <AIPTZTab      readOnly={readOnly} />}
          {tab === "erp"        && <ERPTab />}
          {tab === "viewers"    && (isClientAdmin || isSuperAdmin) && <ViewersTab />}
        </div>
      </div>
    </div>
  );
}
