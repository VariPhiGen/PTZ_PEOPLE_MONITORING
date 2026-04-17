"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, Loader2, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { authApi } from "@/lib/api";
import { setTokens, homeForRole } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { LoginResponse, Role } from "@/types";
import type { AxiosError } from "axios";

// ── Schemas ───────────────────────────────────────────────────────────────────

const credSchema = z.object({
  email:    z.string().min(1, "Email is required"),
  password: z.string().min(1, "Password is required"),
});

const mfaSchema = z.object({
  mfa_code: z.string().length(6, "MFA code must be 6 digits").regex(/^\d+$/, "Digits only"),
});

type CredForm = z.infer<typeof credSchema>;
type MfaForm  = z.infer<typeof mfaSchema>;

// ── Component ─────────────────────────────────────────────────────────────────

function LoginPageInner() {
  const router       = useRouter();
  const searchParams = useSearchParams();
  const redirectTo   = searchParams.get("redirect") ?? "";

  const [step,        setStep]        = useState<"creds" | "mfa">("creds");
  const [loading,     setLoading]     = useState(false);
  const [showPass,    setShowPass]    = useState(false);
  const [pendingCreds, setPendingCreds] = useState<CredForm | null>(null);

  const credForm = useForm<CredForm>({ resolver: zodResolver(credSchema) });
  const mfaForm  = useForm<MfaForm>({ resolver: zodResolver(mfaSchema) });

  // ── Step 1 — email + password ───────────────────────────────────────────

  async function onCredSubmit(data: CredForm) {
    setLoading(true);
    try {
      const res = await authApi.login(data.email, data.password);
      await handleSuccess(res.data);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail: string | { msg: string }[] }>;
      const status   = axiosErr.response?.status;
      const detail   = axiosErr.response?.data?.detail;

      if (status === 422 && typeof detail !== "string" && detail?.[0]?.msg) {
        // MFA required — move to step 2
        if (detail[0].msg.toLowerCase().includes("mfa")) {
          setPendingCreds(data);
          setStep("mfa");
          setLoading(false);
          return;
        }
      }
      // Generic MFA trigger: server returned 401 with mfa_required flag
      if (status === 401 && (axiosErr.response?.data as Record<string, unknown>)?.mfa_required) {
        setPendingCreds(data);
        setStep("mfa");
        setLoading(false);
        return;
      }

      const msg =
        typeof detail === "string"
          ? detail
          : detail?.[0]?.msg ?? "Login failed. Please check your credentials.";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }

  // ── Step 2 — MFA ────────────────────────────────────────────────────────

  async function onMfaSubmit(data: MfaForm) {
    if (!pendingCreds) return;
    setLoading(true);
    try {
      const res = await authApi.login(pendingCreds.email, pendingCreds.password, data.mfa_code);
      await handleSuccess(res.data);
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail: string }>;
      toast.error(axiosErr.response?.data?.detail ?? "Invalid MFA code");
    } finally {
      setLoading(false);
    }
  }

  // ── On success ──────────────────────────────────────────────────────────

  async function handleSuccess(data: LoginResponse) {
    setTokens(data.access_token, data.refresh_token);
    const role     = data.user.role as Role;
    const dest     = redirectTo || homeForRole(role);
    router.replace(dest);
  }

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      {/* Subtle grid background */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.02]"
        style={{
          backgroundImage:
            "linear-gradient(hsl(var(--border)) 1px, transparent 1px), linear-gradient(to right, hsl(var(--border)) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="w-full max-w-sm animate-fade-in">
        {/* Logo / brand */}
        <div className="mb-8 flex flex-col items-center gap-4">
          {/* Platform logo */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/Logowhite.png"
            alt="VGI ACFR"
            className="h-12 w-auto object-contain"
          />
          <div className="text-center">
            <h1 className="text-xl font-semibold tracking-tight text-zinc-50">
              {step === "mfa" ? (
                <span className="flex items-center justify-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-violet-400" />
                  Two-Factor Auth
                </span>
              ) : (
                "Sign in to VGI ACFR"
              )}
            </h1>
            <p className="mt-1 text-sm text-zinc-500">
              {step === "mfa"
                ? "Enter the 6-digit code from your authenticator app"
                : "AI Camera Face Recognition Platform"}
            </p>
          </div>
        </div>

        {/* Card */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/80 p-6 shadow-2xl backdrop-blur-sm">
          {/* ── Step 1: Credentials ─────────────────────────── */}
          {step === "creds" && (
            <form onSubmit={credForm.handleSubmit(onCredSubmit)} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-zinc-400 text-xs">
                  Email address
                </Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  autoFocus
                  placeholder="admin@acas.local"
                  className="bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-600 focus-visible:ring-zinc-600"
                  {...credForm.register("email")}
                />
                {credForm.formState.errors.email && (
                  <p className="text-xs text-red-400">
                    {credForm.formState.errors.email.message}
                  </p>
                )}
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="password" className="text-zinc-400 text-xs">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPass ? "text" : "password"}
                    autoComplete="current-password"
                    placeholder="••••••••••"
                    className="bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-600 focus-visible:ring-zinc-600 pr-10"
                    {...credForm.register("password")}
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setShowPass((s) => !s)}
                    className="absolute inset-y-0 right-0 flex items-center px-3 text-zinc-500 hover:text-zinc-300 transition-colors"
                  >
                    {showPass ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {credForm.formState.errors.password && (
                  <p className="text-xs text-red-400">
                    {credForm.formState.errors.password.message}
                  </p>
                )}
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-zinc-100 text-zinc-900 hover:bg-white font-medium"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Signing in…
                  </>
                ) : (
                  "Continue"
                )}
              </Button>
            </form>
          )}

          {/* ── Step 2: MFA ─────────────────────────────────── */}
          {step === "mfa" && (
            <form onSubmit={mfaForm.handleSubmit(onMfaSubmit)} className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="mfa_code" className="text-zinc-400 text-xs">
                  Authentication code
                </Label>
                <Input
                  id="mfa_code"
                  type="text"
                  inputMode="numeric"
                  autoComplete="one-time-code"
                  autoFocus
                  maxLength={6}
                  placeholder="000000"
                  className="bg-zinc-800 border-zinc-700 text-zinc-100 placeholder:text-zinc-600 text-center text-lg tracking-[0.4em] focus-visible:ring-zinc-600"
                  {...mfaForm.register("mfa_code")}
                />
                {mfaForm.formState.errors.mfa_code && (
                  <p className="text-xs text-red-400 text-center">
                    {mfaForm.formState.errors.mfa_code.message}
                  </p>
                )}
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-zinc-100 text-zinc-900 hover:bg-white font-medium"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Verifying…
                  </>
                ) : (
                  "Verify"
                )}
              </Button>

              <button
                type="button"
                onClick={() => { setStep("creds"); mfaForm.reset(); }}
                className="w-full text-xs text-zinc-500 hover:text-zinc-300 transition-colors text-center"
              >
                ← Back to sign in
              </button>
            </form>
          )}
        </div>

        <p className="mt-4 text-center text-[11px] text-zinc-600">
          VGI ACFR v1.0 · VGI
        </p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginPageInner />
    </Suspense>
  );
}
