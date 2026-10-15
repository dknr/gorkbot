#!/usr/bin/env python3
"""Check whether one Matrix media object is actually reachable.

Takes an mxc URI and probes it the two ways a client or a remote homeserver can
ask for media, which is the whole difference between Matrix 1.10 and 1.11:

  /_matrix/client/v1/media/download/{server}/{id}   authenticated   (Matrix 1.11+)
  /_matrix/media/v3/download/{server}/{id}          unauthenticated (deprecated)

A server that only speaks 1.10 has just the second. A server that only speaks
1.11 has just the first. A remote user's homeserver has to find one the origin
supports; when the two ends share no path in common the image is broken for
them and fine for you, because you ask the server that holds the bytes.

  ./mediacheck.py mxc://caos.one/abc123
  GORK_TOKEN=xxx ./mediacheck.py --base https://caos.lol mxc://caos.lol/def456
  ./mediacheck.py --both mxc://caos.one/abc123 mxc://caos.lol/def456
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CLIENT_MEDIA = "/_matrix/client/v1/media/download/"
LEGACY_MEDIA = "/_matrix/media/v3/download/"
FED_MEDIA = "/_matrix/federation/v1/media/download/"


def probe(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return r.status, r.headers.get("Content-Type", ""), len(body), ""
    except urllib.error.HTTPError as e:
        snippet = e.read()[:90].decode("utf-8", "replace").replace("\n", " ")
        return e.code, e.headers.get("Content-Type", ""), 0, snippet
    except Exception as e:  # DNS, TLS, refused
        return 0, "", 0, str(e)[:90]


def server_versions(base):
    try:
        with urllib.request.urlopen(base + "/_matrix/client/versions", timeout=15) as r:
            v = json.load(r).get("versions", [])
    except Exception as e:
        return None, None, f"versions unreachable: {e}"
    return (v[-1] if v else "?"), ("v1.11" in v), ""


def check(base, mxc, token):
    if not mxc.startswith("mxc://"):
        print(f"  not an mxc URI: {mxc}")
        return 1
    server, _, media_id = mxc[len("mxc://"):].partition("/")
    if not server or not media_id:
        print(f"  malformed mxc URI: {mxc}")
        return 1

    top, has_111, note = server_versions(base)
    print(f"\n{mxc}  via {base}")
    print(f"  spec: max {top}, v1.11 authenticated media: {'yes' if has_111 else 'NO'} {note}")

    rows = [
        ("authenticated client media", CLIENT_MEDIA + f"{server}/{media_id}", True),
        ("legacy unauthenticated", LEGACY_MEDIA + f"{server}/{media_id}", False),
        ("federation media", FED_MEDIA + f"{server}/{media_id}", False),
    ]
    # Routes that returned bytes. A remote user only needs one of these that
    # their own homeserver can reach, which is not the same as the one you use.
    reachable = 0
    for label, path, use_token in rows:
        status, ctype, size, snippet = probe(base + path, token if use_token else None)
        verdict = "OK" if status == 200 else ("route exists, needs auth" if status == 401 else
                                             ("no such media" if status == 404 and "Not found" not in snippet else "MISSING"))
        if status == 200:
            reachable += 1
        print(f"  {status or 'ERR':>3} {label:<26} {ctype or '-':<22} {size:>8}B  {verdict}")
        if snippet and status != 200:
            print(f"        {snippet}")
    if reachable == 0 and token is None:
        print("  -> no token supplied, so the authenticated route proves nothing. "
              "Set GORK_TOKEN before concluding the media is gone.")
    elif reachable == 0:
        print("  -> unreachable by every route tried")
    elif reachable == 1:
        print("  -> one route only: a server that does not speak it shows a broken image")
    return 0 if reachable else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mxc", nargs="+", help="mxc://server/id URI(s) to check")
    ap.add_argument("--base", default=None, help="homeserver to ask (default: https://<media server>)")
    ap.add_argument("--both", action="store_true", help="ask every distinct media server as well as the given base")
    ap.add_argument("--token", default=os.environ.get("GORK_TOKEN", ""), help="access token (or $GORK_TOKEN)")
    a = ap.parse_args()

    bases = [a.base] if a.base else []
    if a.both or not a.base:
        for m in a.mxc:
            s = m[len("mxc://"):].split("/")[0]
            u = "https://" + s.split(":")[0]
            if u not in bases:
                bases.append(u)

    rc = 0
    for m in a.mxc:
        for b in bases:
            rc |= check(b, m, a.token or None)
    return rc


if __name__ == "__main__":
    sys.exit(main())
