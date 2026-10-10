# gorkbot

A Matrix LLM agent built on [ragecore](https://github.com/dknr/ragecore) that
responds to @mentions with chat completions from an LLM.

- Listens for @mentions in Matrix rooms and generates responses via an
  OpenAI-compatible `/chat/completions` endpoint.
- Maintains per-room conversation history (max messages + inactivity timeout).
- Uses ragecore for login, E2EE crypto, and device verification.
- Worker queue serializes LLM calls so history stays consistent.
- Auto-joins rooms when the invite comes from a whitelisted user.

## Build

```sh
make build   # builds to ./gorkbot
```

Or manually: `go build -tags goolm -o gorkbot .` (`goolm` avoids the `libolm`
cgo dependency).

This project depends on
[ragecore](https://github.com/dknr/ragecore), pulled from GitHub as a regular
module dependency pinned by tag in `go.mod`.

## Deploy

### Systemd service

A unit file is included at `contrib/gorkbot.service`.

```bash
# Create dedicated service user
sudo useradd --system --no-create-home --shell /usr/sbin/nologin gorkbot
sudo mkdir -p /etc/gorkbot /var/lib/gorkbot
sudo chown gorkbot:gorkbot /var/lib/gorkbot
sudo cp config.json /etc/gorkbot/config.json
sudo chown gorkbot:gorkbot /etc/gorkbot/config.json

# Build and install
make build
sudo cp gorkbot /usr/local/bin/gorkbot

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now gorkbot.service

# Monitor
journalctl -u gorkbot.service -f
```

The unit expects:
- Binary at `/usr/local/bin/gorkbot`
- Config at `/etc/gorkbot/config.json`
- Database dir `/var/lib/gorkbot` (writable by `gorkbot` user)

## Configure

Create `config.json` (or pass `-config <path>`). Start from
`config.json.example`. Note that `config.json` is gitignored because it
contains credentials.

```json
{
  "homeserver": "https://matrix.example.org",
  "user": "@gork:example.org",
  "password": "change-me",
  "recovery_key": "",
  "database": "gorkbot.db",
  "pickle_key": "gorkbot",
  "device_name": "gorkbot",
  "llm": {
    "base_url": "http://localhost:8080",
    "api_key": "",
    "model": "llama-3.2-3b-instruct",
    "max_history": 100,
    "history_timeout": "15m"
  },
  "gork": {
    "system_prompt": "You are gork, a gruff and terse LLM agent. Respond succinctly.",
    "mention": "@gork:example.org"
  },
  "invite": {
    "allowed_inviters": ["@you:example.org"]
  }
}
```

- `homeserver`, `user`, `password` are required.
- `recovery_key` is only needed to re-verify the bot's device from key backup;
  if empty and no keys exist, the bot generates new cross-signing keys and logs
  the recovery key on startup.
- `llm.base_url` defaults to `http://localhost:8080`.
- `llm.api_key` is optional; omit for unauthenticated endpoints.
- `gork.mention` defaults to the bot's Matrix user ID.
- `gork.markdown` defaults to `true`; set `false` to send plain-text replies without
  markdown-rendered HTML formatting.
- `invite.allowed_inviters` is a list of Matrix user IDs. When one of them
  invites gork to a room, gork joins automatically. The list is empty by
  default, so **gork never auto-joins** unless you configure it. Matching is
  exact on the full user ID (`@name:server`); everyone else's invites are
  ignored, and invites are only acted on when they are addressed to gork.

## Local development

```sh
make run   # build and run locally
```

Or manually: `./gorkbot -config config.json`.

## Console mode (no Matrix)

`-console` runs gork against stdin/stdout instead of a homeserver, for
prompt iteration and testing without a running Matrix server:

```sh
./gorkbot -console -config config.json
```

- Each line of input is a prompt; replies are written to **stdout**
  (logs and the banner go to stderr, so stdout is clean for piping).
- The mention gate is bypassed — every line is answered.
- A line ending in `...` continues a multi-line prompt.
- `exit`/`quit` or Ctrl-D stops the session. Conversation history persists
  across prompts within the session, just like a Matrix room.

Since it's just bytes on pipes, it also works over a serial console:

```sh
socat -raw,echo=0 /dev/ttyUSB0,115200n8 - | ./gorkbot -console -config config.json
```

In console mode the Matrix credentials in the config are ignored (a config
with only `llm` and `gork` sections is enough). Vision is not available in
this mode.

## License

ISC
