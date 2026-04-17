/**
 * JWT utilities — HS256 verification using the Web Crypto API.
 * No Node.js dependencies; runs natively in Cloudflare Workers.
 */
import type { JWTPayload } from "../types.js";

function base64UrlDecode(s: string): Uint8Array {
  // Pad and convert base64url → base64
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const padded = b64.padEnd(b64.length + (4 - (b64.length % 4)) % 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

/**
 * Verify an HS256 JWT.  Returns the decoded payload or null if invalid/expired.
 */
export async function verifyJWT(
  token: string,
  secret: string,
): Promise<JWTPayload | null> {
  const parts = token.split(".");
  if (parts.length !== 3) return null;

  const [header, payload, signature] = parts;

  // Import the secret key
  let key: CryptoKey;
  try {
    key = await crypto.subtle.importKey(
      "raw",
      new TextEncoder().encode(secret),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["verify"],
    );
  } catch {
    return null;
  }

  // Verify signature
  const sigBytes = base64UrlDecode(signature);
  const data     = new TextEncoder().encode(`${header}.${payload}`);
  const valid    = await crypto.subtle.verify("HMAC", key, sigBytes, data);
  if (!valid) return null;

  // Decode payload
  let claims: JWTPayload;
  try {
    claims = JSON.parse(
      new TextDecoder().decode(base64UrlDecode(payload)),
    ) as JWTPayload;
  } catch {
    return null;
  }

  // Check expiry
  if (claims.exp && claims.exp < Math.floor(Date.now() / 1000)) return null;

  // Must be an access token
  if (claims.type !== "access") return null;

  return claims;
}

/**
 * Extract a bearer token from an Authorization header.
 */
export function extractBearer(authHeader: string | null | undefined): string | null {
  if (!authHeader?.startsWith("Bearer ")) return null;
  return authHeader.slice(7).trim() || null;
}
