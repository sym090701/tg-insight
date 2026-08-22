# tg-insight

Lightweight Telegram group archiving, daily summaries, and grounded history Q&A.

## Behavior

- A Telethon user session discovers its joined group chats; sources can be fixed in
  configuration or selected through the private bot.
- Messages are stored in a local SQLite database with FTS5 trigram search.
- A separate Telegram bot exposes `/ask`, `/summary`, `/settings`, `/checkin`, `/status`, and `/help`.
- Daily summaries are sent only to `TG_SUMMARY_TARGET`, at the configured UTC+8
  (Asia/Shanghai) time.
- Daily summaries rank cross-group, high-information developments first, then give
  a separate intelligence update for every group with archived activity.
- Scheduled summaries are incremental per group: after a successful scheduled push,
  each group's cursor advances. A manual `/summary` does not advance cursors, and
  failed pushes are retried without losing unread messages. A small overlap is kept
  so ongoing topics can be recognized as updates.
- Digest records include their precise timestamp and age at analysis time. Older
  messages are not restated as fresh news; they can reappear only as context for a
  genuinely active, recently updated topic, and the summary must describe the new
  development rather than repeat the original item.
- `/content` classifies the recent text of selected groups. Adult groups identified
  automatically or marked manually are excluded from daily and on-demand summaries,
  while their messages remain archived.
- `LLM_FALLBACK_MODEL` can provide a backup model for temporary primary-model
  connection, timeout, rate-limit, or server failures.
- `/checkin` sends configurable plain-text daily check-ins through the authorized
  personal Telegram account. Each target has its own UTC+8 schedule, message, and
  on/off switch. New targets default to 00:30, then send after a persisted random
  300-800ms delay. A successful manual or scheduled check-in is recorded for the
  day, and failed scheduled sends retry at most three times. Verification accepts
  trusted Bot replies containing success phrases, a bare `签到`/`打卡` confirmation,
  or `/qd`/`/checkin` commands, including punctuation around the command. Ordinary
  group members and the user's own outbound command are not treated as verification.
  `/checkin` also shows status/history, supports Topics, and can refresh the group
  and Bot target list.
- When any joined group contains an explicit `/qd`, `/checkin`, or Bot-directed
  check-in action, the service waits briefly for that message's direct Bot reply.
  It proposes automation only after a high-confidence AI review confirms the
  exchange represents a real reusable check-in. The private management Bot shows
  the source message, sender/time/link, Bot reply, AI reason, and exact proposed
  command. Only “同意并部署” creates the default 00:30 automatic check-in; “忽略”
  does nothing.
- Check-in evidence is monitored in every joined group, including groups not
  selected for archiving. Unselected groups are not written to the message
  database: their candidate message and Bot reply exist only in memory until the
  decision is sent. Suggested actions can be ignored for today, seven days, or
  permanently per group. If the configured AI endpoint is temporarily unavailable,
  a candidate with a direct Bot reply remains in memory and is retried with bounded
  exponential backoff (30 seconds up to 15 minutes). The service checks
  `https://status.input.im/api/status` before retries when possible, but treats the
  actual API response as authoritative; malformed or unreachable status data never
  discards evidence. Candidates older than 24 hours expire automatically, and a
  successful suggestion is sent at most once per group per day.
- `/alerts` enables major-event alerts. Keywords are only candidate signals: AI
  verifies the target message against its timestamp and recent same-group context,
  ignores stale forwards and ordinary discussion, and suppresses repeated topics
  unless a later message contains a material update. `/backup` creates
  a consistent SQLite backup and sends it to the authorized user; the seven newest
  backups are retained. The backup contains archived chat history, so treat it as
  sensitive data.
- `/topics` watches selected archived groups for user-defined keywords and sends a
  separate AI-verified update with a cooldown. Topic subscriptions work even when
  broad major-event alerts are disabled. Alert messages provide feedback buttons
  for marking a topic important or ignoring that topic permanently.
- Scheduled check-ins track consecutive failure days. After three consecutive failed
  days, the private notification is explicitly escalated with the target and latest
  error detail.
- Before archived messages, event context, or check-in evidence is sent to the LLM,
  common API keys, Bot tokens, hashes, email addresses, phone numbers, and IP
  addresses are replaced in the model-only copy. Local SQLite records and Telegram
  notifications retain the original source text.
- LLM prompts treat all Telegram content as untrusted data and cannot perform Telegram actions.
- Archive size, free disk reserve, digest input, and automatic retries have hard limits.

## Setup

1. Copy `.env.example` to `.env` and fill every required value.
2. Create the persistent directory with owner `10001:10001`.
3. Build the image.
4. Authorize the Telegram user session once.
5. Start the service, then choose archive sources through `/groups`; the picker supports
   pages, selecting or clearing a whole page, and a final completion button.

```bash
mkdir -p data
chown 10001:10001 data
docker compose build
docker compose run --rm tg-insight auth
docker compose up -d
```

Start a private chat with the configured bot before expecting a scheduled digest.
Only IDs in `TG_ALLOWED_USER_IDS` can use the bot.

## Commands

```text
/groups           Page through joined groups and select or clear a whole page
/recent [1-50]    Choose any joined group and read its most recent text messages
/ask <question>   Search archived group history and answer with sources
/summary          Generate the last 24-hour digest immediately
/settings         Set daily push time or turn scheduled pushes on or off
/content          Classify groups; reanalyze a page or every selected group
/checkin          Add a group/Bot, configure text/time/Topic, history, toggle, or run now
/alerts           Configure major-event alert switch and keywords
/topics           Configure per-group keyword subscriptions
/refresh          Refresh joined groups and Bot targets
/backup           Export a consistent SQLite archive backup to the private Bot chat
/status           Show archive size, source chats, schedule, and model
/help             Show command help
```

Plain text sent privately to the bot is treated as an `/ask` query. Group selection,
recent-message reads, archive queries, and summaries are restricted to configured
numeric user IDs and are available only in a private bot chat.

Check-in target selection is intentionally separate from archive-source selection:
the target can be a joined Telegram group or a Bot that appears in the authorized
account's dialog list. Set the exact text required by the target, for example
`@example_bot /checkin` in a group or `/checkin` in a Bot chat, through the Bot's
`/checkin` menu. The text is sent directly by the authorized Telegram user account
and is never sent to the configured LLM provider.

## Resource limits

The production defaults retain up to two years, up to 10 million messages, and at
most 10 GiB of SQLite archive data while preserving at least 1 GiB of free disk
space. A digest uses at most 500 messages and 120,000 serialized characters. Failed
scheduled digests retry at most three times with exponential backoff. Adjust these
values conservatively for the host capacity.

## Security

The Telethon session file grants access to the Telegram account. Keep `data/` and
`.env` root-owned or otherwise tightly restricted on the host. Prefer a dedicated
Telegram account that is a member only of required groups. Chat records are sent to
the configured LLM provider for summarization and Q&A; use a provider and retention
policy acceptable for the group. Custom LLM endpoints must use HTTPS.

The runtime image uses a pinned base digest and hash-locked Linux/Python 3.12
dependencies. Regenerate and review both lock files before changing dependency
versions. `.dockerignore` excludes the environment file, sessions, and archive from
the Docker build context.
