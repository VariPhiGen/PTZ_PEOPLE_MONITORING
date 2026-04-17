"use client";

/**
 * Enrollment Tokens — admin dashboard page
 *
 * Allows admins to create shareable enrollment links and view/deactivate existing ones.
 * Accessible at /enrollment-tokens (inside the authenticated dashboard group).
 */

import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Copy, Plus, Trash2, Link, Clock, Users, ChevronDown, ChevronUp } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";

// ── Types ─────────────────────────────────────────────────────────────────────

interface EnrollmentToken {
  token_id:     string;
  token:        string;
  label:        string | null;
  role_default: string;
  dataset_id:   string | null;
  expires_at:   number | null;
  max_uses:     number | null;
  use_count:    number;
  is_active:    boolean;
  is_expired:   boolean;
  is_exhausted: boolean;
  created_at:   number;
}

interface CreateResult {
  token_id:    string;
  token:       string;
  enroll_url:  string;
  label:       string | null;
  role_default: string;
  expires_at:  number | null;
  max_uses:    number | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(epoch: number | null) {
  if (!epoch) return "Never";
  return new Date(epoch * 1000).toLocaleDateString(undefined, {
    day: "numeric", month: "short", year: "numeric",
  });
}

function tokenStatus(t: EnrollmentToken) {
  if (!t.is_active) return { label: "Inactive",  color: "bg-slate-700 text-slate-400" };
  if (t.is_expired)  return { label: "Expired",   color: "bg-red-900/50 text-red-400" };
  if (t.is_exhausted) return { label: "Used up",  color: "bg-orange-900/50 text-orange-400" };
  return { label: "Active", color: "bg-green-900/50 text-green-400" };
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function EnrollmentTokensPage() {
  const qc              = useQueryClient();
  const [showCreate,    setShowCreate]    = useState(false);
  const [newLink,       setNewLink]       = useState<CreateResult | null>(null);
  const [expandedId,    setExpandedId]    = useState<string | null>(null);

  // Form state
  const [label,         setLabel]         = useState("");
  const [roleDefault,   setRoleDefault]   = useState("STUDENT");
  const [expiresInDays, setExpiresInDays] = useState<string>("");
  const [maxUses,       setMaxUses]       = useState<string>("");

  // ── Fetch tokens ────────────────────────────────────────────────────────

  const { data: tokens = [], isLoading } = useQuery<EnrollmentToken[]>({
    queryKey: ["enrollment-tokens"],
    queryFn:  () => api.get("/api/enrollment-tokens/").then((r) => r.data),
  });

  // ── Create token ────────────────────────────────────────────────────────

  const createMutation = useMutation({
    mutationFn: () =>
      api.post("/api/enrollment-tokens/", {
        label:           label.trim() || null,
        role_default:    roleDefault,
        expires_in_days: expiresInDays ? parseInt(expiresInDays) : null,
        max_uses:        maxUses ? parseInt(maxUses) : null,
      }).then((r) => r.data as CreateResult),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["enrollment-tokens"] });
      setNewLink(data);
      setShowCreate(false);
      setLabel(""); setExpiresInDays(""); setMaxUses("");
      toast.success("Enrollment link created");
    },
    onError: () => toast.error("Failed to create link"),
  });

  // ── Deactivate token ────────────────────────────────────────────────────

  const deleteMutation = useMutation({
    mutationFn: (tokenId: string) =>
      api.delete(`/api/enrollment-tokens/${tokenId}`).then((r) => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["enrollment-tokens"] });
      toast.success("Link deactivated");
    },
    onError: () => toast.error("Failed to deactivate link"),
  });

  // ── Copy URL helper ─────────────────────────────────────────────────────

  const copyUrl = useCallback((url: string) => {
    navigator.clipboard.writeText(url).then(
      () => toast.success("Link copied to clipboard"),
      () => toast.error("Copy failed — please copy manually"),
    );
  }, []);

  // ── Build enrollment URL for a token ───────────────────────────────────

  const getEnrollUrl = useCallback((token: string) => {
    const base = typeof window !== "undefined" ? window.location.origin : "";
    return `${base}/enroll/${token}`;
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="p-6 max-w-3xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold">Enrollment Links</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            Create shareable links so anyone can self-enroll their face via webcam.
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)} className="gap-2">
          <Plus className="w-4 h-4" />
          New Link
        </Button>
      </div>

      {/* ── New link result ────────────────────────────────────────────────── */}
      {newLink && (
        <div className="mb-6 bg-green-950/40 border border-green-800/40 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="w-5 h-5 rounded-full bg-green-600 flex items-center justify-center text-xs text-white font-bold">✓</span>
            <span className="font-medium text-green-300 text-sm">Link created — share this URL</span>
          </div>
          <div className="flex items-center gap-2 bg-slate-900/60 rounded-lg px-3 py-2">
            <Link className="w-4 h-4 text-slate-400 flex-shrink-0" />
            <span className="text-sm text-slate-300 flex-1 truncate font-mono">
              {newLink.enroll_url}
            </span>
            <button
              onClick={() => copyUrl(newLink.enroll_url)}
              className="flex-shrink-0 text-blue-400 hover:text-blue-300 transition-colors"
            >
              <Copy className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Anyone with this link can enroll their face — no login required.
          </p>
          <button
            onClick={() => setNewLink(null)}
            className="text-xs text-slate-500 hover:text-slate-400 mt-2 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* ── Token list ─────────────────────────────────────────────────────── */}
      {isLoading ? (
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          Loading…
        </div>
      ) : tokens.length === 0 ? (
        <div className="text-center py-16 border border-dashed border-border rounded-xl">
          <Link className="w-8 h-8 mx-auto mb-3 text-muted-foreground opacity-50" />
          <p className="text-muted-foreground text-sm">No enrollment links yet</p>
          <p className="text-muted-foreground text-xs mt-1">
            Create a link to share with employees, students, or visitors
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {tokens.map((tok) => {
            const status   = tokenStatus(tok);
            const enrollUrl = getEnrollUrl(tok.token);
            const expanded  = expandedId === tok.token_id;

            return (
              <div
                key={tok.token_id}
                className="border border-border rounded-xl overflow-hidden transition-all"
              >
                {/* Row */}
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-sm">
                        {tok.label || "Unnamed link"}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${status.color}`}>
                        {status.label}
                      </span>
                      <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                        {tok.role_default}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        {tok.use_count} enrolled
                        {tok.max_uses ? ` / ${tok.max_uses} max` : ""}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Expires: {fmtDate(tok.expires_at)}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      onClick={() => copyUrl(enrollUrl)}
                      title="Copy enrollment URL"
                      className="p-2 hover:bg-muted rounded-lg transition-colors text-muted-foreground hover:text-foreground"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => deleteMutation.mutate(tok.token_id)}
                      title="Deactivate link"
                      className="p-2 hover:bg-red-900/30 rounded-lg transition-colors text-muted-foreground hover:text-red-400"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setExpandedId(expanded ? null : tok.token_id)}
                      className="p-2 hover:bg-muted rounded-lg transition-colors text-muted-foreground"
                    >
                      {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {/* Expanded URL */}
                {expanded && (
                  <div className="border-t border-border px-4 py-3 bg-muted/30">
                    <p className="text-xs text-muted-foreground mb-2">Shareable URL:</p>
                    <div className="flex items-center gap-2 bg-background rounded-lg px-3 py-2 border border-border">
                      <span className="text-xs text-foreground flex-1 truncate font-mono">
                        {enrollUrl}
                      </span>
                      <button
                        onClick={() => copyUrl(enrollUrl)}
                        className="flex-shrink-0 text-blue-400 hover:text-blue-300 transition-colors"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Created: {fmtDate(tok.created_at)} •
                      Token: <code className="font-mono text-xs">{tok.token.slice(0, 12)}…</code>
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Create dialog ──────────────────────────────────────────────────── */}
      <Dialog open={showCreate} onOpenChange={setShowCreate}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Create Enrollment Link</DialogTitle>
          </DialogHeader>

          <div className="space-y-4 pt-2">
            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                Label (optional)
              </label>
              <Input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. Engineering Batch 2026"
              />
            </div>

            <div>
              <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                Default Role
              </label>
              <select
                value={roleDefault}
                onChange={(e) => setRoleDefault(e.target.value)}
                className="w-full bg-background border border-input rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="STUDENT">Student</option>
                <option value="FACULTY">Faculty</option>
                <option value="STAFF">Staff</option>
                <option value="VISITOR">Visitor</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                  Expires in (days)
                </label>
                <Input
                  type="number"
                  min={1}
                  value={expiresInDays}
                  onChange={(e) => setExpiresInDays(e.target.value)}
                  placeholder="Never"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground block mb-1.5">
                  Max uses
                </label>
                <Input
                  type="number"
                  min={1}
                  value={maxUses}
                  onChange={(e) => setMaxUses(e.target.value)}
                  placeholder="Unlimited"
                />
              </div>
            </div>

            <div className="flex gap-3 pt-2">
              <Button
                variant="outline"
                className="flex-1"
                onClick={() => setShowCreate(false)}
              >
                Cancel
              </Button>
              <Button
                className="flex-1"
                onClick={() => createMutation.mutate()}
                disabled={createMutation.isPending}
              >
                {createMutation.isPending ? "Creating…" : "Create Link"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
