"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { timetableApi, camerasApi } from "@/lib/api";
import { Plus, Pencil, Trash2, Clock, BookOpen, ChevronDown, ChevronUp, Camera, AlertTriangle } from "lucide-react";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

type TimetableEntry = {
  entry_id: string;
  day_of_week: number;
  start_time: string;
  end_time: string;
  course_id?: string;
  course_name?: string;
  faculty_id?: string;
  dataset_ids?: string[];
  roster_ids?: string[];
};

type Timetable = {
  timetable_id: string;
  name: string;
  description?: string;
  timezone?: string;
  entries: TimetableEntry[];
  created_at: number;
};

// Common IANA zones — the input is also a free-text field so any valid zone works.
const TIMEZONE_OPTIONS = [
  "UTC",
  "Asia/Kolkata",
  "Asia/Tokyo",
  "Asia/Singapore",
  "Asia/Dubai",
  "Asia/Shanghai",
  "Asia/Seoul",
  "Europe/London",
  "Europe/Paris",
  "Europe/Berlin",
  "America/New_York",
  "America/Chicago",
  "America/Los_Angeles",
  "Australia/Sydney",
];

type CameraRow = {
  camera_id: string;
  name: string;
  timetable_id: string | null;
};

const emptyEntry = (): Omit<TimetableEntry, "entry_id"> => ({
  day_of_week:  0,
  start_time:   "09:00",
  end_time:     "10:00",
  course_id:    "",
  course_name:  "",
  faculty_id:   "",
  dataset_ids:  [],
  roster_ids:   [],
});

// Returns true if two [start, end] HH:MM ranges overlap
function timesOverlap(s1: string, e1: string, s2: string, e2: string) {
  return s1 < e2 && s2 < e1;
}

export default function TimetablesPage() {
  const [timetables, setTimetables] = useState<Timetable[]>([]);
  const [cameras,    setCameras]    = useState<CameraRow[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [expanded,   setExpanded]   = useState<string | null>(null);

  const [showTTDialog,    setShowTTDialog]    = useState(false);
  const [editingTT,       setEditingTT]       = useState<Timetable | null>(null);
  const [ttForm,          setTTForm]          = useState({ name: "", description: "", timezone: "Asia/Kolkata" });

  const [showEntryDialog, setShowEntryDialog] = useState(false);
  const [editingEntry,    setEditingEntry]    = useState<TimetableEntry | null>(null);
  const [entryTTId,       setEntryTTId]       = useState("");
  const [entryForm,       setEntryForm]       = useState<Omit<TimetableEntry, "entry_id">>(emptyEntry());
  const [overlapWarn,     setOverlapWarn]     = useState("");

  const [saving,  setSaving]  = useState(false);
  const [seeding, setSeeding] = useState(false);
  const [error,   setError]   = useState("");

  const load = useCallback(async () => {
    try {
      const [ttRes, camRes] = await Promise.all([
        timetableApi.list(),
        camerasApi.list({ limit: 100 }).catch(() => ({ data: { items: [] } })),
      ]);
      setTimetables(ttRes.data);
      const items = (camRes.data as { items?: CameraRow[] }).items ?? [];
      setCameras(items);
    } catch {
      setError("Failed to load timetables");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Map timetable_id → camera names
  const ttCameras = useMemo(() => {
    const m: Record<string, string[]> = {};
    for (const c of cameras) {
      if (c.timetable_id) {
        if (!m[c.timetable_id]) m[c.timetable_id] = [];
        m[c.timetable_id].push(c.name);
      }
    }
    return m;
  }, [cameras]);

  // ── Timetable CRUD ──────────────────────────────────────────────────────────

  function openCreateTT() {
    setEditingTT(null);
    setTTForm({ name: "", description: "", timezone: "Asia/Kolkata" });
    setError("");
    setShowTTDialog(true);
  }

  function openEditTT(tt: Timetable) {
    setEditingTT(tt);
    setTTForm({
      name: tt.name,
      description: tt.description || "",
      timezone: tt.timezone || "UTC",
    });
    setError("");
    setShowTTDialog(true);
  }

  async function saveTT() {
    if (!ttForm.name.trim()) return;
    const tz = (ttForm.timezone || "UTC").trim() || "UTC";
    setSaving(true); setError("");
    try {
      if (editingTT) {
        await timetableApi.update(editingTT.timetable_id, { ...ttForm, timezone: tz });
      } else {
        await timetableApi.create({
          name:        ttForm.name,
          description: ttForm.description,
          timezone:    tz,
          entries:     [],
        });
      }
      setShowTTDialog(false);
      await load();
    } catch {
      setError("Failed to save timetable");
    } finally {
      setSaving(false);
    }
  }

  async function deleteTT(id: string) {
    const linked = ttCameras[id];
    const msg = linked?.length
      ? `This timetable is used by ${linked.join(", ")}. Deleting it will remove the schedule from those cameras. Continue?`
      : "Delete this timetable and all its time slots?";
    if (!confirm(msg)) return;
    try {
      await timetableApi.remove(id);
      await load();
    } catch {
      setError("Failed to delete timetable");
    }
  }

  async function seedDefault() {
    setSeeding(true); setError("");
    try {
      await timetableApi.seedDefault();
      await load();
    } catch {
      setError("Failed to create default timetable");
    } finally {
      setSeeding(false);
    }
  }

  // ── Entry CRUD ──────────────────────────────────────────────────────────────

  function checkOverlap(ttId: string, form: typeof entryForm, skipId?: string): string {
    const tt = timetables.find(t => t.timetable_id === ttId);
    if (!tt) return "";
    for (const e of tt.entries) {
      if (e.entry_id === skipId) continue;
      if (e.day_of_week !== form.day_of_week) continue;
      if (timesOverlap(form.start_time, form.end_time, e.start_time, e.end_time)) {
        return `Overlaps with "${e.course_name || "another slot"}" (${e.start_time}–${e.end_time}) on ${DAYS[form.day_of_week]}`;
      }
    }
    return "";
  }

  function openAddEntry(ttId: string) {
    setEntryTTId(ttId);
    setEditingEntry(null);
    const f = emptyEntry();
    setEntryForm(f);
    setOverlapWarn(checkOverlap(ttId, f));
    setError("");
    setShowEntryDialog(true);
  }

  function openEditEntry(ttId: string, entry: TimetableEntry) {
    setEntryTTId(ttId);
    setEditingEntry(entry);
    const f = {
      day_of_week:  entry.day_of_week,
      start_time:   entry.start_time,
      end_time:     entry.end_time,
      course_id:    entry.course_id    || "",
      course_name:  entry.course_name  || "",
      faculty_id:   entry.faculty_id   || "",
      dataset_ids:  entry.dataset_ids  || [],
      roster_ids:   entry.roster_ids   || [],
    };
    setEntryForm(f);
    setOverlapWarn(checkOverlap(ttId, f, entry.entry_id));
    setError("");
    setShowEntryDialog(true);
  }

  function updateEntryField(patch: Partial<typeof entryForm>) {
    const next = { ...entryForm, ...patch };
    setEntryForm(next);
    setOverlapWarn(checkOverlap(entryTTId, next, editingEntry?.entry_id));
  }

  async function saveEntry() {
    if (overlapWarn) return;          // block save if overlap
    if (entryForm.end_time <= entryForm.start_time) {
      setError("End time must be after start time");
      return;
    }
    setSaving(true); setError("");
    const payload = {
      ...entryForm,
      course_id:   entryForm.course_id   || null,
      course_name: entryForm.course_name || null,
      faculty_id:  entryForm.faculty_id  || null,
      dataset_ids: entryForm.dataset_ids?.length ? entryForm.dataset_ids : null,
      roster_ids:  entryForm.roster_ids?.length  ? entryForm.roster_ids  : null,
    };
    try {
      if (editingEntry) {
        await timetableApi.updateEntry(entryTTId, editingEntry.entry_id, payload);
      } else {
        await timetableApi.addEntry(entryTTId, payload);
      }
      setShowEntryDialog(false);
      await load();
    } catch {
      setError("Failed to save time slot");
    } finally {
      setSaving(false);
    }
  }

  async function deleteEntry(ttId: string, entryId: string) {
    if (!confirm("Remove this time slot?")) return;
    try {
      await timetableApi.removeEntry(ttId, entryId);
      await load();
    } catch {
      setError("Failed to delete time slot");
    }
  }

  // ── Weekly grid for a timetable ─────────────────────────────────────────────

  function WeeklyGrid({ tt }: { tt: Timetable }) {
    const byDay: Record<number, TimetableEntry[]> = {};
    for (const e of tt.entries) {
      if (!byDay[e.day_of_week]) byDay[e.day_of_week] = [];
      byDay[e.day_of_week].push(e);
    }
    // Sort entries within each day by start_time
    Object.values(byDay).forEach(arr => arr.sort((a, b) => a.start_time.localeCompare(b.start_time)));

    const activeDays = [0,1,2,3,4,5,6].filter(d => byDay[d]?.length);

    if (activeDays.length === 0) {
      return (
        <div className="text-xs text-zinc-500 py-2">
          No time slots yet. Add a slot to get started.
        </div>
      );
    }

    return (
      <div className="space-y-3">
        {[0,1,2,3,4,5,6].map(dow => {
          const entries = byDay[dow];
          if (!entries?.length) return null;
          return (
            <div key={dow} className="flex gap-3">
              <div className="w-10 shrink-0 pt-2.5">
                <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
                  {DAY_SHORT[dow]}
                </span>
              </div>
              <div className="flex-1 space-y-1.5">
                {entries.map(e => (
                  <div
                    key={e.entry_id}
                    className="flex items-center justify-between bg-zinc-800/80 border border-zinc-700/60 rounded-lg px-3 py-2 group hover:border-zinc-600 transition-colors"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      <span className="text-xs font-mono text-zinc-400 shrink-0 tabular-nums">
                        {e.start_time} – {e.end_time}
                      </span>
                      {e.course_name ? (
                        <span className="text-sm font-medium text-white truncate">{e.course_name}</span>
                      ) : (
                        <span className="text-xs text-zinc-500 italic">No course name</span>
                      )}
                      {e.course_id && (
                        <span className="text-[10px] text-zinc-600 shrink-0">({e.course_id})</span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0 ml-2">
                      <button
                        onClick={() => openEditEntry(tt.timetable_id, e)}
                        className="p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                        title="Edit slot"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => deleteEntry(tt.timetable_id, e.entry_id)}
                        className="p-1 rounded hover:bg-red-900/40 text-zinc-400 hover:text-red-400 transition-colors"
                        title="Delete slot"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white flex items-center gap-2">
            <Clock className="h-5 w-5 text-blue-400" /> Timetables
          </h1>
          <p className="text-xs text-zinc-500 mt-1">
            Define weekly schedules — one course per time slot. Cameras follow these schedules automatically.
          </p>
        </div>
        <button
          onClick={openCreateTT}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          <Plus className="h-4 w-4" /> New Timetable
        </button>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-300 text-sm px-4 py-2.5 rounded-lg flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-3">
          {[1,2].map(i => (
            <div key={i} className="h-16 bg-zinc-900 border border-zinc-800 rounded-xl animate-pulse" />
          ))}
        </div>
      ) : timetables.length === 0 ? (
        <div className="text-center py-20 border border-dashed border-zinc-700 rounded-xl">
          <BookOpen className="h-10 w-10 mx-auto mb-3 text-zinc-600" />
          <p className="text-sm text-zinc-500 mb-4">No timetables yet.</p>
          <button
            onClick={seedDefault}
            disabled={seeding}
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
          >
            <Clock className="h-4 w-4" />
            {seeding ? "Creating…" : "Create Sample Timetable"}
          </button>
          <p className="text-xs mt-2 text-zinc-600">Mon–Sun · 10:00–19:00 · with sample courses</p>
        </div>
      ) : (
        <div className="space-y-4">
          {timetables.map(tt => {
            const isOpen = expanded === tt.timetable_id;
            const assignedCams = ttCameras[tt.timetable_id] ?? [];
            return (
              <div key={tt.timetable_id} className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">

                {/* Card header */}
                <div
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-zinc-800/40 transition-colors"
                  onClick={() => setExpanded(isOpen ? null : tt.timetable_id)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <BookOpen className="h-4 w-4 text-blue-400 shrink-0" />
                    <div className="min-w-0">
                      <p className="font-medium text-white">{tt.name}</p>
                      {tt.description && (
                        <p className="text-xs text-zinc-500 truncate">{tt.description}</p>
                      )}
                    </div>

                    {/* Slot count */}
                    <span className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full border border-zinc-700 shrink-0">
                      {tt.entries.length} {tt.entries.length === 1 ? "slot" : "slots"}
                    </span>

                    {/* Timezone badge */}
                    <span
                      className="text-[10px] font-mono bg-zinc-800 text-blue-300 px-2 py-0.5 rounded-full border border-zinc-700 shrink-0"
                      title="Timezone used for matching time slots"
                    >
                      {tt.timezone || "UTC"}
                    </span>

                    {/* Camera assignment badges */}
                    {assignedCams.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {assignedCams.map(name => (
                          <span
                            key={name}
                            className="flex items-center gap-1 text-[10px] text-green-400 bg-green-900/20 border border-green-800/40 px-1.5 py-0.5 rounded-full shrink-0"
                          >
                            <Camera className="h-2.5 w-2.5" /> {name}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-1 shrink-0 ml-3">
                    <button
                      onClick={e => { e.stopPropagation(); openEditTT(tt); }}
                      className="p-1.5 rounded hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                      title="Edit timetable name"
                    >
                      <Pencil className="h-4 w-4" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); deleteTT(tt.timetable_id); }}
                      className="p-1.5 rounded hover:bg-red-900/40 text-zinc-400 hover:text-red-400 transition-colors"
                      title="Delete timetable"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                    {isOpen
                      ? <ChevronUp className="h-4 w-4 text-zinc-500" />
                      : <ChevronDown className="h-4 w-4 text-zinc-500" />
                    }
                  </div>
                </div>

                {/* Expanded: weekly grid */}
                {isOpen && (
                  <div className="border-t border-zinc-800 px-4 py-4 space-y-4">
                    <WeeklyGrid tt={tt} />
                    <button
                      onClick={() => openAddEntry(tt.timetable_id)}
                      className="flex items-center gap-1.5 text-sm text-blue-400 hover:text-blue-300 transition-colors"
                    >
                      <Plus className="h-4 w-4" /> Add time slot
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Timetable dialog ───────────────────────────────────────────────────── */}
      {showTTDialog && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-md p-6 space-y-4">
            <h2 className="text-base font-semibold text-white">
              {editingTT ? "Edit Timetable" : "New Timetable"}
            </h2>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Name <span className="text-red-400">*</span></label>
                <input
                  autoFocus
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={ttForm.name}
                  onChange={e => setTTForm({ ...ttForm, name: e.target.value })}
                  placeholder="e.g. Semester 1 — 2026"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Description</label>
                <input
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={ttForm.description}
                  onChange={e => setTTForm({ ...ttForm, description: e.target.value })}
                  placeholder="Optional"
                />
              </div>
              <div>
                <label className="text-xs text-zinc-400 block mb-1">
                  Timezone <span className="text-red-400">*</span>
                </label>
                <input
                  list="tt-timezone-options"
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 font-mono"
                  value={ttForm.timezone}
                  onChange={e => setTTForm({ ...ttForm, timezone: e.target.value })}
                  placeholder="e.g. Asia/Tokyo"
                />
                <datalist id="tt-timezone-options">
                  {TIMEZONE_OPTIONS.map(z => <option key={z} value={z} />)}
                </datalist>
                <p className="text-[11px] text-zinc-500 mt-1">
                  IANA zone (e.g. <span className="font-mono">Asia/Kolkata</span>,{" "}
                  <span className="font-mono">Asia/Tokyo</span>). Slot start/end times are interpreted in this zone.
                </p>
              </div>
            </div>
            {error && <p className="text-xs text-red-400">{error}</p>}
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setShowTTDialog(false)}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveTT}
                disabled={saving || !ttForm.name.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Entry dialog ───────────────────────────────────────────────────────── */}
      {showEntryDialog && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-w-md p-6 space-y-4">
            <h2 className="text-base font-semibold text-white">
              {editingEntry ? "Edit Time Slot" : "Add Time Slot"}
            </h2>

            <div className="space-y-3">
              {/* Day */}
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Day <span className="text-red-400">*</span></label>
                <select
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={entryForm.day_of_week}
                  onChange={e => updateEntryField({ day_of_week: Number(e.target.value) })}
                >
                  {DAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
                </select>
              </div>

              {/* Time range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">Start Time <span className="text-red-400">*</span></label>
                  <input
                    type="time"
                    className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                    value={entryForm.start_time}
                    onChange={e => updateEntryField({ start_time: e.target.value })}
                  />
                </div>
                <div>
                  <label className="text-xs text-zinc-400 block mb-1">End Time <span className="text-red-400">*</span></label>
                  <input
                    type="time"
                    className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                    value={entryForm.end_time}
                    onChange={e => updateEntryField({ end_time: e.target.value })}
                  />
                </div>
              </div>

              {/* Overlap warning */}
              {overlapWarn && (
                <div className="flex items-start gap-2 text-xs text-amber-400 bg-amber-900/20 border border-amber-800/40 rounded-lg px-3 py-2">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                  {overlapWarn}
                </div>
              )}

              {/* Course */}
              <div>
                <label className="text-xs text-zinc-400 block mb-1">Course Name</label>
                <input
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={entryForm.course_name}
                  onChange={e => updateEntryField({ course_name: e.target.value })}
                  placeholder="e.g. Data Structures"
                />
              </div>

              <div>
                <label className="text-xs text-zinc-400 block mb-1">Course ID <span className="text-zinc-600">(optional)</span></label>
                <input
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  value={entryForm.course_id}
                  onChange={e => updateEntryField({ course_id: e.target.value })}
                  placeholder="e.g. CS301"
                />
              </div>

              {/* Dataset IDs */}
              <div>
                <label className="text-xs text-zinc-400 block mb-1">
                  Dataset IDs <span className="text-zinc-600">(comma-separated, for roster)</span>
                </label>
                <input
                  className="w-full bg-zinc-800 border border-zinc-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 font-mono"
                  value={(entryForm.dataset_ids || []).join(", ")}
                  onChange={e => updateEntryField({
                    dataset_ids: e.target.value.split(",").map(s => s.trim()).filter(Boolean),
                  })}
                  placeholder="uuid-1, uuid-2"
                />
              </div>
            </div>

            {error && (
              <p className="text-xs text-red-400 flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" /> {error}
              </p>
            )}

            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setShowEntryDialog(false)}
                className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveEntry}
                disabled={saving || !!overlapWarn}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
                title={overlapWarn ? "Resolve overlap before saving" : undefined}
              >
                {saving ? "Saving…" : "Save Slot"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
