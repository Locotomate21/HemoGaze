#!/usr/bin/env python
"""Command-line client for the HemoGaze server.

The smallest useful consumer: send an image, print the estimate and the warning
that comes with it. Depends only on ``requests``, so it runs anywhere the service
is reachable and needs none of the training stack.

    python examples/cli_client.py path/to/conjunctiva.png
    python examples/cli_client.py img.png --cutoff 12 --url http://host:8000

Exit codes are meant for scripting: 0 on a successful estimate, 2 when the
service is unreachable or has no weights, 1 on a bad request. A pipeline that
treats "service down" and "prediction returned" the same way will silently
publish nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="WHO threshold in g/dL: 11 children 6-59 months, "
                         "12 adult women, 13 adult men")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"no such file: {args.image}", file=sys.stderr)
        return 1

    params = {} if args.cutoff is None else {"cutoff_g_dl": args.cutoff}
    try:
        with args.image.open("rb") as fh:
            r = requests.post(f"{args.url.rstrip('/')}/predict",
                              files={"image": (args.image.name, fh)},
                              params=params, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"cannot reach {args.url}: {exc}", file=sys.stderr)
        return 2

    if r.status_code == 503:
        print(f"service has no weights loaded: {r.json().get('detail')}",
              file=sys.stderr)
        return 2
    if not r.ok:
        print(f"HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return 1

    d = r.json()
    print(f"haemoglobin estimate : {d['hemoglobin_g_dl']:.2f} g/dL")
    print(f"conjunctiva in frame : {d['roi_fraction']:.1%}")
    if d.get("below_cutoff") is not None:
        verdict = "below" if d["below_cutoff"] else "at or above"
        print(f"vs cutoff {d['cutoff_g_dl']:.0f} g/dL : {verdict}")
    if d["roi_fraction"] < 0.05:
        print("\n[!] almost no conjunctiva detected -- is this the right crop?",
              file=sys.stderr)
    print(f"\n{d['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
