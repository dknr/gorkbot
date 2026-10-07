# gorkbot

A Matrix LLM agent built on [ragecore](https://github.com/dknr/ragecore) that
responds to @mentions with chat completions from an LLM.

- Listens for @mentions in Matrix rooms and generates responses via an
  OpenAI-compatible `/chat/completions` endpoint.
- Maintains per-room conversation history (max messages + inactivity timeout).
- Uses ragecore for login, E2EE crypto, and device verification.
- Worker queue serializes LLM calls so history stays consistent.

## Build

```sh
make build   # builds to ./gorkbot
```

Or manually: `go build -tags goolm -o gorkbot .` (`goolm` avoids the `libolm`
cgo dependency).

This project depends on
[ragecore](https://github.com/dknr/ragecore). The `replace` directive in
`go.mod` expects ragecore to be checked out as a sibling directory
(`../ragecore`); adjust it if your layout differs.

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

## Local development

```sh
make run   # build and run locally
```

Or manually: `./gorkbot -config config.json`.

## License

ISC
