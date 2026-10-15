#!/usr/bin/env python3
"""Show what gorkbot's draw loop actually did, from gorkbot.log.

Each attempt logs twice: "Drawing" before the request, with the profile it asked
for, and "Drew image" after it, with the duration and the server's own info echo
(steps and CFG as applied). Reading the two together is what proves the quality
profile ran on the final attempt rather than merely being configured:

  ./contrib/drawlog.py                # last 12 attempts
  ./contrib/drawlog.py --all
  ./contrib/drawlog.py --log /var/log/gorkbot.log

Note on lora=: stable-diffusion.cpp does not echo applied adapters in this
build, so it reads "not reported" on every line, including the quality attempt
that asks for no adapter at all. The proof that the turbo adapter was dropped is
the duration and the steps/cfg pair, not that field.
"""
import argparse
import json
import sys

FIELDS = ("attempt", "max_draws", "quality", "draw_ms", "eyes_ms", "dur_ms", "sd_info")


def rows(path):
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("message") in ("Drawing", "Drew image", "Image generation failed"):
                yield rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="gorkbot.log")
    ap.add_argument("--all", action="store_true", help="every attempt, not just the last 12")
    ap.add_argument("--room", default=None, help="only attempts from this room id")
    a = ap.parse_args()

    out = [r for r in rows(a.log) if not a.room or a.room in json.dumps(r)]
    if not a.all:
        out = out[-12:]
    if not out:
        print(f"no draw attempts found in {a.log}", file=sys.stderr)
        return 1

    for r in out:
        msg = r.get("message")
        draw, eyes = r.get("draw_ms"), r.get("eyes_ms")
        if isinstance(draw, (int, float)):
            cost = f"{draw / 1000:6.1f}s draw"
            if isinstance(eyes, (int, float)):
                cost += f" +{eyes / 1000:4.1f}s eyes"
        elif isinstance(r.get("dur_ms"), (int, float)):
            cost = f"{r['dur_ms'] / 1000:6.1f}s total"  # builds before draw_ms/eyes_ms
        else:
            cost = "        -       "
        profile = "QUALITY" if r.get("quality") else "fast   "
        stamp = (r.get("time") or "")[11:19]
        info = r.get("sd_info") or ""
        extra = "  " + (r.get("error") or "")[:60] if msg == "Image generation failed" else ""
        print(f"{stamp}  attempt {r.get('attempt', '-')}/{r.get('max_draws', '-')}  "
              f"{profile}  {cost}  {msg:<12} {info}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
