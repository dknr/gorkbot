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
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "cfg_scale": 1,
    "lora": ["qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors"],
    "timeout": "3m",
    "max_retries": 2
  }
}
```

- `image.base_url` is the stable-diffusion.cpp root (its A1111-compatible
  `/sdapi/v1/txt2img` is called). Empty means no drawing at all.
- `steps` and `cfg_scale` are sent explicitly and default to `8` and `1`, the
  distilled regime. A CFG above 1 costs a second conditioning pass per step:
  20 steps at CFG 7 measured 34.7s at 1024, 8 steps at CFG 1 measured 10.2s
  without an adapter and 14.6s with the turbo one.
- `negative_prompt` is omitted by default. At CFG 1 there is no unconditional
  pass to put it in, so it would be dead weight.
- `lora` names the adapters applied to every generation, as `name` or
  `name:multiplier` (multiplier in `(0, 2]`; a `0` is a startup error rather
  than a silently dropped adapter). **stable-diffusion.cpp has no startup LoRA
  option and no options API** — `--lora-model-dir` only makes adapters
  *available* — so anything not listed here is drawn by the bare base model.
  The server resolves the name against that directory, so a bare file name is
  enough. Nothing else in the stack can tell you an adapter was applied: the
  server's `info` echo carries steps and CFG but not this build's adapter list,
  so gorkbot logs `sd_info=…` per draw and says `lora=not reported` rather than
  pretending the list was empty.
- `width`/`height` must be sent: a request that omits them gets 512x512 even
  when the server was started with `--width 1024 --height 1024`.
- `max_retries` defaults to `2`: one draw plus two redraws. **After the second
  retry the image is sent whether or not gork likes it.**
- Drawing needs `llm_vision` as well. With no vision model configured the tool
  is not offered: gork cannot judge a picture it cannot see.
- The eyes are the slow half of a draw, and `llm_vision.max_tokens` is the knob.
  The endpoint streams about 25 tokens/s. `"Describe what you see in this
  image"` wrote 299 tokens and took 13.1s; the terse prompt self-limits to 40-63
  tokens and takes 2.5-2.8s on the same drawing — the prompt is what bought the
  speed, and `max_tokens` only ever cuts a description short (100 truncated the
  old verbose answer mid-sentence). The default prompt asks for two sentences
  naming every subject, its colour, what it is doing, then background and
  lighting — enough for the 3B to spot a mismatch in ~40 tokens.
- Thinking stays off on the vision path, and that is not only about speed:
  reasoning is charged against `max_tokens`, so a request that leaves it on at
  a 200-token budget returns an **empty** caption rather than a short one.
  `TestVisionLatency` trips on exactly that. Do not raise `max_tokens` without
  reading this twice.
- Every attempt lands in room history as two **assistant** lines, failures
  included: `[🎨 draw_image] <prompt>` before the attempt, then
  `[🎨 draw_image result] <description>` from the vision model. They are not
  user lines and carry no sender name — stored as `<sender>: <description>`,
  the model read its own tool output back as something the person had said.
- **A finished draw sends the image and nothing else.** No caption, no closing
  sentence, no assessment. The `m.image` `body` is required by the spec and is
  set to the inert `gork.png`, because clients render that field as a caption or
  tooltip and any description there reads as a second message under the
  picture. The vision description stays in the model's history, where the next
  retry decision needs it, and the model's own closing text is dropped rather
  than relayed — it tends to be an assessment, or a narration of an intention to
  redraw that the budget will never honour. Text is still sent when no image
  was: a declined request, or a draw that failed.

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

Images in an encrypted room are encrypted client-side and sent in the event's
`file` field with the mxc inside it, the way every other Matrix client does;
plaintext rooms get a plain `url`. `filename` is always set to `gork.png`
because clients name the saved download from `filename` then `body`, and a
missing extension makes their next upload come back as
`application/octet-stream`.

**Known unresolved issue:** users on a different homeserver still cannot view
these images, while the local homeserver renders them fine. Encryption, the
event shape and the file bytes have all been checked and the cause is unknown;
media federation between the two homeservers is the untested suspect.

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
