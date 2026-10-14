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

- `image` configures the `draw_image` tool, see [Drawing images](#drawing-images).
  Disabled while `image.base_url` is empty.

## Drawing images

With a [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
server running, gork can draw. It gets a `draw_image` tool, renders the
picture, looks at it with its vision model ("eyes"), and decides whether to
send it or redraw it.

```json
{
  "image": {
    "base_url": "http://localhost:8001",
    "width": 512,
    "height": 512,
    "steps": 20,
    "cfg_scale": 7,
    "negative_prompt": "text, watermark, blurry",
    "timeout": "2m",
    "max_retries": 2
  }
}
```

- `image.base_url` is the stable-diffusion.cpp root (its A1111-compatible
  `/sdapi/v1/txt2img` is called). Empty means no drawing at all.
- `max_retries` defaults to `2`: one draw plus two redraws. **After the second
  retry the image is sent whether or not gork likes it.**
- Drawing needs `llm_vision` as well. With no vision model configured the tool
  is not offered: gork cannot judge a picture it cannot see.
- Every generated image, including rejected retries, is described by the vision
  model and kept in room history as `[🎨 image] <description>`.

The tool loop is deliberately blunt because Llama 3.2-3B on llama.cpp cannot
be trusted with subtler tool handling. What the code assumes was measured on
the running server:

- Two tool calls in one response crash the server's PEG parser with HTTP 500
  and lose the parsed call, so only the first call in a response is ever used.
- With tools offered on a turn that already has a tool result, the model re-issues
  the same call forever. Once the draw budget is spent the tools are withdrawn
  and `tool_choice` is set to `"none"`, which is what makes it answer (measured
  0/5 to 5/5).
- A model with no way to decline calls the tool for everything, so abstention is
  itself a tool: `answer_directly` (0/8 to 3/3). Its downside is that it can
  then answer from stale weights instead of a tool it should have used.
- Descriptions are echoed back as a *user* message, not only as a tool result,
  because tool-result content is ignored while tools are still offered.

Verify the whole path against live servers:

```sh
GORK_PROMPT="draw me a purple cat wearing a top hat and reading a newspaper" \
  go test -tags goolm,integration -run TestIntegrationDrawLoop -v -timeout 13m .
```

Images land in `integration-draws/`. Endpoints come from `GORK_LLM_URL`,
`GORK_VISION_URL` and `GORK_SD_URL`.

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
