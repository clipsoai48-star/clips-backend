# Clipso AI — Frontend

Next.js app that talks to the `clipso-backend` API. Landing page, signup/login, and a dashboard where you paste a link and watch clips get generated.

## Setup

**1. Install Node.js** if you don't have it — download from nodejs.org (LTS version), or `brew install node`.

**2. Install dependencies:**
```bash
npm install
```
This is a much lighter install than the backend — no torch/whisper here, just the web app itself. Should take under a minute.

**3. Set up your environment file:**
```bash
cp .env.local.example .env.local
```
The default (`http://localhost:8000`) already matches your backend's local address — no changes needed unless you deploy the backend elsewhere later.

**4. Run it:**
```bash
npm run dev
```
Opens at `http://localhost:3000`.

## Important — your backend must be running too

This frontend doesn't do anything on its own; it's just the interface. You need **three** things running at once now:
1. Redis (`brew services start redis` — probably already running from backend setup)
2. Backend API (`uvicorn main:app --reload` in `clipso-backend`)
3. Backend worker (`rq worker --worker-class rq.worker.SimpleWorker clipso_jobs` in `clipso-backend`)
4. This frontend (`npm run dev` in `clipso-frontend`)

Four terminals total. Yes, that's a lot — this is normal for local development with this many moving pieces; it collapses into a single deploy click later when you host it for real.

## What's built

- **Landing page** (`/`) — marketing page with pricing
- **Signup / Login** (`/signup`, `/login`) — creates a real account against your backend, stores the JWT in the browser
- **Dashboard** (`/dashboard`) — paste a URL, pick clip count/length/caption style, submit. Polls automatically every 4 seconds while a job is queued/processing, then shows the rendered clips as playable videos once done.

Free vs. paid tier is enforced by reading `is_paid_tier` from `/auth/me` — paid-only caption styles show as disabled options for free accounts. The backend re-checks this server-side too (see `main.py`), so this is a UX nicety, not the actual security boundary.

## Not yet built

- Stripe checkout (the "Go Pro" button currently just goes to signup — wiring up real payment is the next step once you're ready)
- File upload from the dashboard (backend supports it at `POST /jobs/upload`, just not wired into this UI yet — only URL submission is hooked up)
- Production file storage (clips are served straight off local disk via the backend's `/storage` static mount — fine for dev, swap for S3/R2 + signed URLs before real users)
