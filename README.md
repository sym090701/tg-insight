# tg-insight

Lightweight Telegram group archiving, daily summaries, and grounded history Q&A.

On first private-chat use, the Bot asks you to choose `中文` or `English`. The
choice is stored in the local SQLite state and controls Bot menus, notifications,
summaries, alerts, check-in reports, and AI answers. Use `/settings` to switch
languages later; proactive notifications wait until a language has been selected.

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
  does nothing. A model `404` for a verified candidate is retained in a dedicated
  SQLite queue and retried together at the next UTC+8 midnight; completed,
  ignored, configured, or seven-day-old candidates are removed.
- Check-in evidence is monitored in every joined group, including groups not
  selected for archiving. Unselected groups are not written to the message
  database: their candidate message and Bot reply exist only in memory until the
  decision is sent. Suggested actions can be ignored for today, seven days, or
  permanently per group. If the configured AI endpoint is temporarily unavailable,
  a candidate with a direct Bot reply, event alert, topic analysis, content
  classification, or scheduled digest is retried with bounded exponential backoff
  (30 seconds, 60 seconds, 120 seconds, 240 seconds, 480 seconds, then 15 minutes).
  After the bounded attempts, the candidate is stored in SQLite. The service checks
  `https://status.input.im/api/status` and retries all stored candidates once the
  configured models are reported available. The actual API response remains
  authoritative; malformed or unreachable status data never discards evidence.
  Candidates older than 7 days expire automatically, and a successful suggestion is
  sent at most once per group per day. `/retry` manually wakes every persisted AI
  task after the status probe reports an available model.
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
/retry            Manually retry persisted check-in analysis candidates
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

## 中文说明

`tg-insight` 是一个 Telegram 群组归档、每日摘要和基于历史消息问答工具。

首次私聊 Bot 时，会先让你选择 `中文` 或 `English`。选择会保存到本地 SQLite
状态中，并控制 Bot 菜单、通知、摘要、重大事件提醒、签到报告和 AI 查询结果。
之后可以通过 `/settings` 随时切换语言；在选择语言前，后台主动通知会暂缓发送。

### 功能

- Telethon 用户会话发现已加入的群组；归档来源可以在配置文件中固定，也可以通过私聊 Bot 选择。
- 消息保存在本地 SQLite 数据库中，并使用 FTS5 trigram 搜索。
- `/ask` 查询历史，`/summary` 生成摘要，`/content` 管理成人内容排除，`/status` 查看状态。
- 每日摘要只发送给 `TG_SUMMARY_TARGET`，按配置的 UTC+8（Asia/Shanghai）时间推送。
- 摘要先跨群排序高信息量、影响大、紧急或可执行的内容，再分别列出各群情报。
- 摘要使用精确时间戳和消息年龄；旧消息不会被当作新消息重复报告，只有同一话题近期出现真实进展时才作为上下文。
- `/content` 自动或手动标记成人内容的群组不会进入摘要，但消息仍保存在归档中。
- `LLM_FALLBACK_MODEL` 可在主模型临时连接、超时、限流或服务错误时提供备用模型。
- `/checkin` 使用已授权的 Telegram 个人账号发送每日签到。每个目标有独立的 UTC+8 时间、文本和开关；新目标默认 00:30，并随机延后 300-800ms。
- 签到验证会识别可信 Bot 的成功回复、明确的 `签到`/`打卡` 回复，以及 `/qd`/`/checkin` 命令；普通成员消息和自己的发送命令不会被当作成功凭据。
- AI 临时不可用时，签到建议、重大事件、话题分析、内容分类和摘要任务会按退避策略重试，之后写入 SQLite，模型恢复后再统一重试。`/retry` 可手动触发重试。
- `/alerts` 配置重大事件提醒，`/topics` 配置群组关键词订阅，`/backup` 导出 SQLite 备份到授权用户私聊。备份包含历史消息，应按敏感数据保护。
- 发送给 LLM 前，API Key、Bot Token、哈希、邮箱、手机号和 IP 地址等会在模型副本中替换；本地记录和 Telegram 通知保留原始文本。
- Telegram 内容均作为不可信数据处理，LLM 不能直接执行 Telegram 操作。归档大小、磁盘预留、摘要输入和自动重试均有硬限制。

### 部署

1. 将 `.env.example` 复制为 `.env`，填写所有必填配置。
2. 创建持久化目录，并将所有者设为 `10001:10001`。
3. 构建镜像，首次运行授权 Telegram 用户会话，然后启动服务。
4. 和配置的 Bot 私聊一次，再通过 `/groups` 选择归档群组。只有 `TG_ALLOWED_USER_IDS` 中的用户 ID 可以使用管理功能。

常用命令：`mkdir -p data`、`chown 10001:10001 data`、`docker compose build`、
`docker compose run --rm tg-insight auth`、`docker compose up -d`。

### 命令

| 命令 | 作用 |
| --- | --- |
| `/groups` | 翻页选择或清除归档群组 |
| `/recent [1-50]` | 读取指定群组最近的文字消息 |
| `/ask <问题>` | 搜索历史并引用来源回答 |
| `/summary` | 立即生成过去 24 小时摘要 |
| `/settings` | 设置每日推送时间、开关和语言 |
| `/content` | 管理群组内容分类并重新分析 |
| `/checkin` | 管理自动签到目标、文本、时间、Topic 和历史 |
| `/alerts` | 设置重大事件提醒和关键词 |
| `/topics` | 设置群组关键词订阅 |
| `/refresh` | 刷新群组和 Bot 目标 |
| `/backup` | 导出 SQLite 归档备份到私聊 Bot |
| `/status` | 查看归档、来源、计划和模型状态 |
| `/retry` | 手动重试持久化的 AI 任务 |
| `/help` | 查看帮助 |

私聊 Bot 发送的普通文本会作为 `/ask` 查询。群组选择、最近消息读取、归档查询和摘要只能在私聊 Bot 中操作。

签到目标和归档来源是独立配置：签到目标可以是已加入的 Telegram 群组，也可以是授权账号对话列表中的 Bot。
通过 `/checkin` 设置完整文本，例如群组中的 `@example_bot /checkin` 或 Bot 私聊中的 `/checkin`。
文本由已授权的 Telegram 用户账号直接发送，不会发送给配置的 LLM 服务商。

### 资源限制

默认最多保留两年消息、1000 万条记录和 10 GiB SQLite 数据，同时至少保留 1 GiB 可用磁盘空间。
单次摘要最多使用 500 条消息和 120,000 个序列化字符；失败的定时摘要最多按退避策略重试三次。

### 安全

Telethon 会话文件拥有 Telegram 账号访问权限。请严格限制 `data/` 和 `.env` 的权限，并优先使用只加入必要群组的专用账号。
聊天记录会发送给配置的 LLM 服务商用于摘要和问答，请选择符合群组隐私和留存要求的服务商。自定义 LLM 地址必须使用 HTTPS。
运行镜像使用固定摘要的基础镜像并锁定依赖哈希；`.dockerignore` 会排除环境文件、会话文件和归档数据。
