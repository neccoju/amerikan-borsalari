# External trigger setup — cron-job.org → GitHub Actions

This bot is **not** scheduled with GitHub's native `schedule:` cron. Instead an
external scheduler (cron-job.org) calls the GitHub API once per trading day,
after the US market close, to dispatch the `usbot-daily` workflow.

> No real tokens appear in this document. Everything below is a template.

---

## 1. Why external cron?

- GitHub's `schedule:` cron is best-effort and can be delayed/dropped on busy
  runners. cron-job.org gives precise, reliable timing.
- The bot still runs its own **trading-day / US-holiday guard**: on weekends or
  holidays it exits cleanly and emits a "market closed, skipped" report, so an
  occasional extra trigger is harmless.

---

## 2. Create a GitHub fine-grained personal access token

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. **Resource owner:** your account (`neccoju`).
3. **Repository access:** *Only select repositories* → `neccoju/amerikan-borsalari`.
4. **Permissions → Repository permissions:**
   - **Contents:** Read-only (needed for checkout).
   - **Metadata:** Read-only (auto-selected).
   - **Actions:** Read and write  ← required to dispatch the workflow.
5. **Expiration:** pick a reasonable window (e.g. 90 days) and set a reminder to
   rotate it.
6. Generate and copy the token **once** (you cannot view it again).

> The token is used by cron-job.org only. It does **not** need to be added to
> GitHub Secrets for the dispatch itself. Add `CRON_SECRET_TOKEN` to GitHub
> Secrets only if you want the workflow to sanity-check a shared payload token
> (optional, see step 5).

---

## 3. Choose the endpoint and method

Two API options — we use **repository_dispatch** (simplest, matches the workflow):

| | repository_dispatch | workflow_dispatch |
|---|---|---|
| URL | `https://api.github.com/repos/neccoju/amerikan-borsalari/dispatches` | `https://api.github.com/repos/neccoju/amerikan-borsalari/actions/workflows/daily.yml/dispatches` |
| Method | `POST` | `POST` |
| Needs branch ref | no | yes (`"ref":"main"`) |
| Event filter | `types: [daily-run]` | `inputs:` |

### Endpoint (repository_dispatch)
```
POST https://api.github.com/repos/neccoju/amerikan-borsalari/dispatches
```

### Headers
```
Accept: application/vnd.github+json
Authorization: Bearer <YOUR_FINE_GRAINED_TOKEN>
X-GitHub-Api-Version: 2022-11-28
Content-Type: application/json
```

### Body / payload
```json
{
  "event_type": "daily-run",
  "client_payload": { "token": "<OPTIONAL_SHARED_SECRET>" }
}
```
`client_payload.token` is optional; include it only if you set `CRON_SECRET_TOKEN`
in GitHub Secrets and want the workflow to verify it.

---

## 4. Configure cron-job.org

1. Create a free account at https://cron-job.org and **Create cronjob**.
2. **URL:** the endpoint from step 3.
3. **Request method:** `POST`.
4. **Headers:** add the four headers from step 3 (put the token in
   `Authorization`).
5. **Request body:** the JSON payload from step 3.
6. **Schedule (US market close + data settle):** US equities close 16:00 ET.
   Trigger ~21:30–22:00 **UTC** on weekdays. cron-job.org schedules in a fixed
   timezone, so prefer **UTC** to stay DST-proof:
   - Cron expression (UTC): `30 21 * * 1-5`
   - The bot's own calendar guard handles US holidays.
7. **Save.** Enable failure notifications so a dead trigger is visible.

---

## 5. Optional shared-secret check

If you want to reject spoofed dispatches:
1. Add a GitHub Secret `CRON_SECRET_TOKEN` (any random string).
2. Put the same value in `client_payload.token` (step 3 body).
3. The workflow step "Optional sanity check of external trigger token" compares
   them and fails the run on mismatch. If `CRON_SECRET_TOKEN` is unset, the check
   is skipped.

---

## 6. Testing

**A. Manual UI run (no cron needed):**
GitHub → **Actions → usbot-daily → Run workflow** (optionally tick `force` /
`dry_run`).

**B. Test the API dispatch from your machine (curl):**
```bash
curl -i -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer <YOUR_FINE_GRAINED_TOKEN>" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/neccoju/amerikan-borsalari/dispatches \
  -d '{"event_type":"daily-run","client_payload":{"token":"<OPTIONAL>"}}'
```
A `204 No Content` means accepted. Check **Actions** for the new run.

**C. cron-job.org "Run now":** use the *Test run* / *Execute now* button; it
shows the HTTP response code (expect `204`).

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `404` from API | token lacks repo access, or repo path typo |
| `403` | token missing **Actions: write** permission |
| `422` | bad JSON body / wrong `event_type` |
| Run starts but exits "market closed" | non-trading day — expected |
| Token mismatch error | `CRON_SECRET_TOKEN` set but payload token differs |
