"use client";

/**
 * Public Self-Enrollment Page
 *
 * Accessible at /enroll/[token] — no login required.
 * Guides the user through 5 face poses (center, right, left, up, down),
 * runs real-time quality checks via the backend, then submits enrollment.
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useParams } from "next/navigation";

// ── Types ─────────────────────────────────────────────────────────────────────

interface OrgInfo {
  client_name: string;
  label: string;
  role_default: string;
  expires_at: number | null;
  uses_remaining: number | null;
}

interface CapturedPose {
  pose: PoseConfig;
  tempKey: string;
  dataUrl: string;         // for preview thumbnail
  quality: Record<string, unknown>;
}

interface PoseConfig {
  id: string;
  label: string;
  instruction: string;
  subtext: string;
  icon: React.ReactNode;
}

type Step = "loading" | "error" | "details" | "camera" | "review" | "submitting" | "success";

// ── Constants ─────────────────────────────────────────────────────────────────
//
// API calls use a relative base so the page works from any hostname:
//   Via Cloudflare (nginx proxy on 7080): /api/* → nginx → backend:18000
//   Direct local access:                  /api/* → Next.js rewrites → backend
// An explicit NEXT_PUBLIC_API_URL override is still respected if set.
const API_BASE = (() => {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  // In browser context use relative path (works behind any reverse proxy)
  if (typeof window !== "undefined") return "";
  // SSR fallback — not used since this is a "use client" page
  return "";
})();

const POSES: PoseConfig[] = [
  {
    id: "center",
    label: "Look Straight",
    instruction: "Look directly at the camera",
    subtext: "Keep your face centred in the oval",
    icon: <FaceIconCenter />,
  },
  {
    id: "right",
    label: "Turn Right",
    instruction: "Slowly turn your head to the right",
    subtext: "About 30° — keep your eyes visible",
    icon: <FaceIconRight />,
  },
  {
    id: "left",
    label: "Turn Left",
    instruction: "Slowly turn your head to the left",
    subtext: "About 30° — keep your eyes visible",
    icon: <FaceIconLeft />,
  },
  {
    id: "up",
    label: "Tilt Up",
    instruction: "Tilt your chin slightly upward",
    subtext: "Just a gentle tilt — don't look at the ceiling",
    icon: <FaceIconUp />,
  },
  {
    id: "down",
    label: "Tilt Down",
    instruction: "Look slightly downward",
    subtext: "Eyes still forward, chin gently tucked",
    icon: <FaceIconDown />,
  },
];

// ── SVG Pose Icons ─────────────────────────────────────────────────────────────

function FaceIconCenter() {
  return (
    <svg viewBox="0 0 80 80" className="w-16 h-16">
      <ellipse cx="40" cy="40" rx="22" ry="28" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="32" cy="34" r="3" fill="currentColor" />
      <circle cx="48" cy="34" r="3" fill="currentColor" />
      <path d="M33 50 Q40 56 47 50" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="40" cy="43" r="2" fill="currentColor" opacity="0.5" />
    </svg>
  );
}

function FaceIconRight() {
  return (
    <svg viewBox="0 0 80 80" className="w-16 h-16">
      <ellipse cx="44" cy="40" rx="18" ry="28" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="39" cy="34" r="3" fill="currentColor" />
      <circle cx="52" cy="33" r="2.5" fill="currentColor" />
      <path d="M37 50 Q45 55 52 49" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      {/* Arrow right */}
      <path d="M10 40 L24 40" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
      <path d="M18 33 L26 40 L18 47" fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FaceIconLeft() {
  return (
    <svg viewBox="0 0 80 80" className="w-16 h-16">
      <ellipse cx="36" cy="40" rx="18" ry="28" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="28" cy="33" r="2.5" fill="currentColor" />
      <circle cx="41" cy="34" r="3" fill="currentColor" />
      <path d="M28 49 Q36 55 43 50" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      {/* Arrow left */}
      <path d="M70 40 L56 40" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
      <path d="M62 33 L54 40 L62 47" fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FaceIconUp() {
  return (
    <svg viewBox="0 0 80 80" className="w-16 h-16">
      <ellipse cx="40" cy="43" rx="22" ry="24" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="32" cy="39" r="3" fill="currentColor" />
      <circle cx="48" cy="39" r="3" fill="currentColor" />
      <path d="M34 53 Q40 58 46 53" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      {/* Arrow up */}
      <path d="M40 28 L40 14" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
      <path d="M33 21 L40 13 L47 21" fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FaceIconDown() {
  return (
    <svg viewBox="0 0 80 80" className="w-16 h-16">
      <ellipse cx="40" cy="37" rx="22" ry="24" fill="none" stroke="currentColor" strokeWidth="3" />
      <circle cx="32" cy="33" r="3" fill="currentColor" />
      <circle cx="48" cy="33" r="3" fill="currentColor" />
      <path d="M33 46 Q40 52 47 46" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
      {/* Arrow down */}
      <path d="M40 52 L40 66" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" />
      <path d="M33 59 L40 67 L47 59" fill="none" stroke="#3b82f6" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Webcam component ──────────────────────────────────────────────────────────

interface WebcamProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  onReady: (ready: boolean) => void;
}

function Webcam({ videoRef, onReady }: WebcamProps) {
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          onReady(true);
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Camera access denied";
        setError(msg.includes("denied") || msg.includes("Permission")
          ? "Camera access denied — please allow camera permission and reload"
          : "Could not access camera — make sure it is connected");
        onReady(false);
      }
    }

    start();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      if (videoRef.current) videoRef.current.srcObject = null;
      onReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-2xl text-center p-6">
        <div>
          <div className="text-red-400 text-4xl mb-3">📷</div>
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-2xl bg-black aspect-[4/3] w-full">
      {/* Mirrored video feed */}
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className="w-full h-full object-cover"
        style={{ transform: "scaleX(-1)" }}
      />

      {/* Face oval guide overlay */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Semi-transparent overlay with oval cutout */}
        <defs>
          <mask id="oval-mask">
            <rect width="400" height="300" fill="white" />
            <ellipse cx="200" cy="145" rx="105" ry="130" fill="black" />
          </mask>
        </defs>
        <rect width="400" height="300" fill="rgba(0,0,0,0.45)" mask="url(#oval-mask)" />
        {/* Oval border */}
        <ellipse
          cx="200" cy="145" rx="105" ry="130"
          fill="none"
          stroke="rgba(255,255,255,0.85)"
          strokeWidth="2.5"
          strokeDasharray="8 4"
        />
        {/* Corner alignment guides */}
        <line x1="84" y1="25" x2="315" y2="25" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
        <line x1="84" y1="265" x2="315" y2="265" stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
      </svg>
    </div>
  );
}

// ── Countdown circle ──────────────────────────────────────────────────────────

function CountdownCircle({ count }: { count: number }) {
  return (
    <div className="relative w-20 h-20 mx-auto mb-2">
      <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
        <circle cx="40" cy="40" r="34" fill="none" stroke="#1e293b" strokeWidth="6" />
        <circle
          cx="40" cy="40" r="34"
          fill="none"
          stroke="#3b82f6"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${213.6 * (count / 3)} 213.6`}
          style={{ transition: "stroke-dasharray 0.9s linear" }}
        />
      </svg>
      <span className="absolute inset-0 flex items-center justify-center text-3xl font-bold text-white">
        {count}
      </span>
    </div>
  );
}

// ── Main page component ───────────────────────────────────────────────────────

export default function EnrollPage() {
  const params = useParams<{ token: string }>();
  const token  = params?.token ?? "";

  const [step,        setStep]        = useState<Step>("loading");
  const [orgInfo,     setOrgInfo]     = useState<OrgInfo | null>(null);
  const [errorMsg,    setErrorMsg]    = useState<string>("");
  const [poseIndex,   setPoseIndex]   = useState(0);
  const [captures,    setCaptures]    = useState<CapturedPose[]>([]);
  const [capturing,   setCapturing]   = useState(false);
  const [countdown,   setCountdown]   = useState<number | null>(null);
  const [feedback,    setFeedback]    = useState<{ ok: boolean; msg: string } | null>(null);
  const [camReady,    setCamReady]    = useState(false);
  const [submitError, setSubmitError] = useState<string>("");

  // Form fields
  const [name,       setName]       = useState("");
  const [extId,      setExtId]      = useState("");
  const [email,      setEmail]      = useState("");
  const [phone,      setPhone]      = useState("");
  const [department, setDepartment] = useState("");
  const [consent,    setConsent]    = useState(false);

  const videoRef  = useRef<HTMLVideoElement>(null!);
  const canvasRef = useRef<HTMLCanvasElement>(null!);

  // ── Fetch org info on mount ──────────────────────────────────────────────

  useEffect(() => {
    if (!token) { setStep("error"); setErrorMsg("Invalid enrollment link"); return; }

    fetch(`${API_BASE}/api/public/enroll/${token}`)
      .then(async (r) => {
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(d.detail ?? `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((data: OrgInfo) => {
        setOrgInfo(data);
        setStep("details");
      })
      .catch((e) => {
        setErrorMsg(
          e.message.includes("404") ? "This enrollment link doesn't exist"
          : e.message.includes("410") ? "This enrollment link has expired or reached its limit"
          : "Could not load enrollment page — check your connection"
        );
        setStep("error");
      });
  }, [token]);

  // ── Capture a frame from the webcam ──────────────────────────────────────

  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video  = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Mirror to match display
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);

    // Convert to Blob for upload
    const blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob((b) => res(b), "image/jpeg", 0.92)
    );
    if (!blob) return;

    setCapturing(true);
    setFeedback(null);

    // Start 3-2-1 countdown for UX
    for (const n of [3, 2, 1]) {
      setCountdown(n);
      await new Promise((r) => setTimeout(r, 900));
    }
    setCountdown(null);

    // Upload to backend for quality check
    const fd = new FormData();
    fd.append("file", blob, "capture.jpg");

    try {
      const res = await fetch(
        `${API_BASE}/api/public/enroll/${token}/upload`,
        { method: "POST", body: fd }
      );
      const data = await res.json();

      if (data.passed) {
        const pose = POSES[poseIndex];
        setCaptures((prev) => [...prev, {
          pose,
          tempKey:  data.temp_key,
          dataUrl,
          quality:  data.quality ?? {},
        }]);
        setFeedback({ ok: true, msg: data.reason_human ?? "Great shot!" });

        // Auto-advance to next pose after 1.2 s
        setTimeout(() => {
          setFeedback(null);
          if (poseIndex + 1 < POSES.length) {
            setPoseIndex((i) => i + 1);
          } else {
            setStep("review");
          }
        }, 1200);
      } else {
        setFeedback({
          ok:  false,
          msg: data.reason_human ?? "Please try again",
        });
      }
    } catch {
      setFeedback({ ok: false, msg: "Upload failed — check your connection" });
    } finally {
      setCapturing(false);
    }
  }, [token, poseIndex]);

  // ── Submit enrollment ──────────────────────────────────────────────────

  const handleSubmit = useCallback(async () => {
    if (!name.trim()) { setSubmitError("Please enter your name"); return; }
    setStep("submitting");
    setSubmitError("");

    try {
      const res = await fetch(
        `${API_BASE}/api/public/enroll/${token}/submit`,
        {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            name:        name.trim(),
            external_id: extId.trim() || undefined,
            email:       email.trim() || undefined,
            phone:       phone.trim() || undefined,
            department:  department.trim() || undefined,
            image_keys:  captures.map((c) => c.tempKey),
          }),
        }
      );

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Enrollment failed");
      setStep("success");
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Enrollment failed — please try again");
      setStep("review");
    }
  }, [token, name, extId, email, phone, department, captures]);

  // ── Re-take a specific pose ────────────────────────────────────────────

  const retakePose = useCallback((idx: number) => {
    setCaptures((prev) => prev.filter((_, i) => i !== idx));
    setPoseIndex(idx);
    setFeedback(null);
    setStep("camera");
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      <canvas ref={canvasRef} className="hidden" />

      {/* Header */}
      <header className="flex items-center gap-3 px-5 pt-6 pb-4">
        <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center font-bold text-sm flex-shrink-0">
          {orgInfo ? orgInfo.client_name.charAt(0).toUpperCase() : "A"}
        </div>
        <div>
          <p className="text-xs text-slate-400 leading-none">Face Enrollment</p>
          <p className="font-semibold text-sm leading-tight">
            {orgInfo?.client_name ?? "ACAS"}
          </p>
        </div>
      </header>

      <main className="px-5 pb-10 max-w-md mx-auto">

        {/* ── Loading ──────────────────────────────────────────────────────── */}
        {step === "loading" && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <div className="w-10 h-10 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-slate-400 text-sm">Loading enrollment…</p>
          </div>
        )}

        {/* ── Error ────────────────────────────────────────────────────────── */}
        {step === "error" && (
          <div className="mt-12 text-center">
            <div className="text-5xl mb-4">🔗</div>
            <h2 className="text-xl font-semibold mb-2">Link Unavailable</h2>
            <p className="text-slate-400 text-sm">{errorMsg}</p>
          </div>
        )}

        {/* ── Details form ─────────────────────────────────────────────────── */}
        {step === "details" && (
          <div>
            <div className="mb-7 mt-2">
              <h1 className="text-2xl font-bold mb-1">{orgInfo?.label}</h1>
              <p className="text-slate-400 text-sm">
                Complete the form and capture 5 face photos to register.
              </p>
            </div>

            <div className="space-y-4">
              <Field
                label="Full Name"
                required
                value={name}
                onChange={setName}
                placeholder="e.g. Priya Sharma"
                autoComplete="name"
              />
              <Field
                label={orgInfo?.role_default === "STUDENT" ? "Student / Employee ID" : "Employee ID"}
                value={extId}
                onChange={setExtId}
                placeholder="Your ID number (optional)"
              />
              <div className="grid grid-cols-2 gap-3">
                <Field
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  placeholder="you@example.com"
                  autoComplete="email"
                />
                <Field
                  label="Phone"
                  type="tel"
                  value={phone}
                  onChange={setPhone}
                  placeholder="+91 …"
                  autoComplete="tel"
                />
              </div>
              <Field
                label="Department / Class"
                value={department}
                onChange={setDepartment}
                placeholder="e.g. Engineering, Sales …"
              />

              {/* Consent */}
              <label className="flex gap-3 items-start cursor-pointer group">
                <input
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                  className="mt-1 w-4 h-4 accent-blue-500 flex-shrink-0"
                />
                <span className="text-xs text-slate-400 group-hover:text-slate-300 transition-colors">
                  I consent to my facial biometric data being stored and used for
                  attendance and access management by {orgInfo?.client_name}.
                </span>
              </label>
            </div>

            <button
              onClick={() => {
                if (!name.trim()) return;
                if (!consent) return;
                setPoseIndex(0);
                setCaptures([]);
                setStep("camera");
              }}
              disabled={!name.trim() || !consent}
              className="mt-6 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold py-3.5 rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              Continue to Camera
              <span className="text-lg">→</span>
            </button>
          </div>
        )}

        {/* ── Camera capture ────────────────────────────────────────────────── */}
        {step === "camera" && (
          <div>
            {/* Progress dots */}
            <div className="flex items-center justify-center gap-2 mb-5 mt-2">
              {POSES.map((p, i) => (
                <div
                  key={p.id}
                  className={`rounded-full transition-all duration-300 ${
                    i < captures.length
                      ? "w-3 h-3 bg-green-500"
                      : i === poseIndex
                      ? "w-4 h-4 bg-blue-500 ring-2 ring-blue-400 ring-offset-2 ring-offset-slate-900"
                      : "w-3 h-3 bg-slate-700"
                  }`}
                />
              ))}
            </div>

            {/* Pose instruction */}
            <div className="flex items-center gap-4 mb-5 bg-slate-800/60 rounded-2xl p-4">
              <div className="text-blue-400 flex-shrink-0">
                {POSES[poseIndex].icon}
              </div>
              <div>
                <p className="font-semibold text-white">{POSES[poseIndex].instruction}</p>
                <p className="text-xs text-slate-400 mt-0.5">{POSES[poseIndex].subtext}</p>
              </div>
            </div>

            {/* Webcam */}
            <Webcam videoRef={videoRef} onReady={setCamReady} />

            {/* Countdown overlay */}
            {countdown !== null && (
              <div className="mt-4 flex justify-center">
                <CountdownCircle count={countdown} />
              </div>
            )}

            {/* Feedback */}
            {feedback && !countdown && (
              <div
                className={`mt-4 flex items-center gap-2.5 rounded-xl px-4 py-3 text-sm font-medium ${
                  feedback.ok
                    ? "bg-green-900/60 text-green-300 border border-green-700/40"
                    : "bg-red-900/60 text-red-300 border border-red-700/40"
                }`}
              >
                <span className="text-lg">{feedback.ok ? "✓" : "✗"}</span>
                {feedback.msg}
              </div>
            )}

            {/* Capture button */}
            {!countdown && !feedback?.ok && (
              <button
                onClick={captureFrame}
                disabled={capturing || !camReady}
                className="mt-5 w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold py-4 rounded-xl transition-colors flex items-center justify-center gap-2 text-base"
              >
                {capturing ? (
                  <>
                    <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Checking…
                  </>
                ) : (
                  <>
                    <span className="text-xl">📸</span>
                    {`Capture ${POSES[poseIndex].label}`}
                  </>
                )}
              </button>
            )}

            {/* Tip */}
            <p className="mt-4 text-center text-xs text-slate-500">
              Pose {poseIndex + 1} of {POSES.length} — good lighting helps accuracy
            </p>
          </div>
        )}

        {/* ── Review ───────────────────────────────────────────────────────── */}
        {step === "review" && (
          <div>
            <div className="mb-5 mt-2">
              <h2 className="text-xl font-bold mb-1">Review Photos</h2>
              <p className="text-slate-400 text-sm">
                Tap a photo to retake it, or submit to complete enrollment.
              </p>
            </div>

            <div className="grid grid-cols-5 gap-2 mb-6">
              {POSES.map((pose, i) => {
                const cap = captures[i];
                return (
                  <div
                    key={pose.id}
                    className="flex flex-col items-center gap-1"
                  >
                    <button
                      onClick={() => cap && retakePose(i)}
                      className="relative w-full aspect-square rounded-xl overflow-hidden ring-2 ring-slate-700 hover:ring-blue-500 transition-all"
                    >
                      {cap ? (
                        <>
                          <img
                            src={cap.dataUrl}
                            alt={pose.label}
                            className="w-full h-full object-cover"
                          />
                          <div className="absolute top-1 right-1 w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                            <span className="text-white text-[9px] font-bold">✓</span>
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full bg-slate-800 flex items-center justify-center">
                          <span className="text-slate-600 text-xl">?</span>
                        </div>
                      )}
                    </button>
                    <span className="text-[10px] text-slate-500 text-center leading-tight">
                      {pose.label}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Summary */}
            <div className="bg-slate-800/50 rounded-xl p-4 mb-5 text-sm space-y-1.5">
              <div className="flex justify-between">
                <span className="text-slate-400">Name</span>
                <span className="font-medium">{name}</span>
              </div>
              {extId && (
                <div className="flex justify-between">
                  <span className="text-slate-400">ID</span>
                  <span className="font-medium">{extId}</span>
                </div>
              )}
              {email && (
                <div className="flex justify-between">
                  <span className="text-slate-400">Email</span>
                  <span className="font-medium truncate max-w-[180px]">{email}</span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-slate-400">Photos captured</span>
                <span className="font-medium text-green-400">{captures.length} / {POSES.length}</span>
              </div>
            </div>

            {submitError && (
              <div className="mb-4 bg-red-900/50 border border-red-700/40 text-red-300 text-sm rounded-xl px-4 py-3">
                {submitError}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={captures.length < POSES.length}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold py-4 rounded-xl transition-colors flex items-center justify-center gap-2 text-base"
            >
              Submit Enrollment
            </button>

            {captures.length < POSES.length && (
              <button
                onClick={() => { setStep("camera"); }}
                className="mt-3 w-full border border-slate-700 hover:border-slate-600 text-slate-300 font-medium py-3 rounded-xl transition-colors text-sm"
              >
                Continue capturing remaining photos
              </button>
            )}
          </div>
        )}

        {/* ── Submitting ───────────────────────────────────────────────────── */}
        {step === "submitting" && (
          <div className="flex flex-col items-center justify-center h-64 gap-5">
            <div className="relative w-16 h-16">
              <div className="w-16 h-16 border-2 border-blue-900 rounded-full" />
              <div className="absolute inset-0 w-16 h-16 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div className="absolute inset-3 flex items-center justify-center text-2xl">🔐</div>
            </div>
            <div className="text-center">
              <p className="font-semibold">Enrolling your face…</p>
              <p className="text-sm text-slate-400 mt-1">This takes a few seconds</p>
            </div>
          </div>
        )}

        {/* ── Success ──────────────────────────────────────────────────────── */}
        {step === "success" && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
            <div className="w-20 h-20 bg-green-900/40 rounded-full flex items-center justify-center mb-6 ring-4 ring-green-700/30">
              <span className="text-4xl">✓</span>
            </div>
            <h2 className="text-2xl font-bold mb-2">Enrolled Successfully!</h2>
            <p className="text-slate-400 text-sm mb-1">
              Welcome, <span className="text-white font-medium">{name}</span>
            </p>
            <p className="text-slate-500 text-xs mt-2 max-w-xs">
              Your face has been registered with {orgInfo?.client_name}.
              You can close this window.
            </p>

            <div className="mt-8 bg-slate-800/50 rounded-xl px-6 py-4 text-sm space-y-2 w-full">
              <div className="flex items-center gap-2 text-green-400">
                <span>✓</span>
                <span>{captures.length} face photos captured</span>
              </div>
              <div className="flex items-center gap-2 text-green-400">
                <span>✓</span>
                <span>Front, left, right, up, and down poses</span>
              </div>
              <div className="flex items-center gap-2 text-green-400">
                <span>✓</span>
                <span>Face embeddings generated</span>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

// ── Reusable form field ────────────────────────────────────────────────────────

function Field({
  label,
  required,
  value,
  onChange,
  placeholder,
  type = "text",
  autoComplete,
}: {
  label: string;
  required?: boolean;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  autoComplete?: string;
}) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-400 mb-1.5">
        {label}
        {required && <span className="text-blue-400 ml-1">*</span>}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoComplete={autoComplete}
        className="w-full bg-slate-800 border border-slate-700 focus:border-blue-500 focus:ring-1 focus:ring-blue-500/30 rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 outline-none transition-all"
      />
    </div>
  );
}
