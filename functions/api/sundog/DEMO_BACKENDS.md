# Cross-vendor demo backends — setup (ship-ticket #8)

`chat.html` has a **"Try it yourself"** panel (`#sd-demo`) that runs a prompt through the
**same claim gate** the live widget uses, against a backend you pick:

| Backend in the selector | Runs where        | Needs a key? |
| ----------------------- | ----------------- | ------------ |
| Deterministic           | **browser** (no network, no key) | no — always works |
| `gpt-4o-mini`           | `POST /api/sundog/draft` → OpenAI    | `OPENAI_API_KEY` |
| `claude-haiku-4-5`      | `POST /api/sundog/draft` → Anthropic | `ANTHROPIC_API_KEY` |
| `llama-3.3` (llama-3.3-70b-versatile) | `POST /api/sundog/draft` → Groq | `GROQ_API_KEY` |
| `qwen` (qwen3-32b)      | `POST /api/sundog/draft` → Groq      | `GROQ_API_KEY` |

The deterministic backend is the flagship guarantee and needs nothing. The hosted backends
are **optional**: the Function returns a graceful `503 backend_unavailable` when a key isn't
configured, and the UI shows "not enabled on this deployment" — so it is always safe to ship
without the keys.

## Enabling the hosted backends in production (Cloudflare Pages)

Set the three keys as **encrypted Pages secrets** on the `sundog` project — never commit them.

```sh
# from the repo root, once per key (paste the value when prompted; it is not echoed)
npx wrangler pages secret put OPENAI_API_KEY
npx wrangler pages secret put ANTHROPIC_API_KEY
npx wrangler pages secret put GROQ_API_KEY
```

(Or set them in the Cloudflare dashboard → Pages → **sundog** → Settings → Environment
variables → **Encrypt**.) They become `env.OPENAI_API_KEY` etc. inside `draft.js`.

You only need the providers you want live. Add `GROQ_API_KEY` alone and both `llama-3.3`
and `qwen` light up; the others keep showing "not enabled."

## Rate limiting

Per-IP, **10 hosted calls / hour**, via the existing `sundog_rate_limit` KV binding
(already declared in `wrangler.toml`, shared with the other `/api/sundog/*` endpoints).
No extra setup — it works in production automatically. The deterministic backend is never
rate-limited (it never hits the Function). When the KV binding is absent (rare), the limiter
is skipped rather than failing closed.

## Local development

`.dev.vars` (git-ignored) holds the three keys for local testing. Run the full stack with:

```sh
npm run build                       # produces dist/ (the demo module ships to dist/js/)
npx wrangler pages dev --port 8788  # serves dist/ + functions/ + reads .dev.vars
```

Then open `http://127.0.0.1:8788/chat`. `npm run dev` (vite, port 5179) serves the page but
**not** the Functions, so only the deterministic backend works there.

## How the discipline stays identical to the widget

`draft.js` only returns the model's **raw draft** (built with the same heavy-trace prompt the
eval harness uses — it imports `chat/eval/lib/adapters/*`). The **browser** then runs
`gateModelDraft()` on that draft — the exact gate the deployed widget runs — so an overclaiming
draft is **rejected** and replaced with the supported answer, live, in front of the visitor.
Untrusted model text is rendered with `textContent` only (never `innerHTML`).
