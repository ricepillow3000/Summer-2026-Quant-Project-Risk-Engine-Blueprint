"""
Outside-in security probe of a running deployment - what an attacker can reach.

    python -m tools.security_probe https://<deployment>

The threat model (CLAUDE.md, SECURITY.md) is a public, anonymous, read-only app:
no accounts, passwords, sessions, JWTs, payments, uploads, webhooks or user
records exist, so password resets, IDOR/BOLA, JWT secrets and payment checks
have no surface here. This checks the surfaces that DO exist, from outside:

- nothing private is served: .env, .git, logs/, data/ cache, source, secrets;
- path traversal (plain and encoded) cannot climb out of the static root;
- CORS does not echo a foreign origin;
- no JavaScript source map leaks app internals (Streamlit's own bundle maps,
  if any, are reported, not failed - they are public open-source code);
- the server banner and security headers are reported for the record.

Exit code 1 on any FAIL.
"""

import sys
import urllib.error
import urllib.request

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def get(url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "meleona-security-probe",
                                               **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310 - probe target
            return r.status, dict(r.headers), r.read(200_000)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(20_000)


def main() -> int:
    base = sys.argv[1].rstrip("/")
    _, _, home = get(base + "/")

    # A private file "leaks" only if the response is that file - the app answers
    # unknown paths with its own index.html, so compare against the home page.
    private = {
        "/.env": b"=", "/.git/config": b"[core]", "/.git/HEAD": b"ref:",
        "/logs/meleona.log": b"session", "/main.py": b"import streamlit",
        "/src/ingestion.py": b"def fetch_prices", "/requirements.txt": b"streamlit",
        "/.streamlit/secrets.toml": b"=", "/.streamlit/config.toml": b"[server]",
        "/data/": b"parquet", "/Procfile": b"web:",
    }
    for path, marker in private.items():
        status, _, body = get(base + path)
        leaked = status == 200 and body != home and marker in body
        check(f"not served: {path}", not leaked, f"HTTP {status}")

    for path in ("/static/../main.py", "/static/..%2F..%2Fmain.py", "/%2e%2e/%2e%2e/main.py",
                 "/static/%2e%2e/%2e%2e/requirements.txt", "/app/static/../../main.py"):
        status, _, body = get(base + path)
        check(f"no path traversal: {path}",
              not (status == 200 and body != home and b"import" in body), f"HTTP {status}")

    for path in ("/_stcore/health", "/"):
        _, hdr, _ = get(base + path, {"Origin": "https://evil.example"})
        acao = {k.lower(): v for k, v in hdr.items()}.get("access-control-allow-origin", "")
        check(f"CORS does not trust a foreign origin on {path}",
              acao not in ("*", "https://evil.example"), f"ACAO={acao or '(none)'}")

    maps = [line for line in home.decode("utf-8", "ignore").split('"')
            if line.endswith(".js") and "static/js" in line][:3]
    for js in maps:
        status, _, _ = get(base + "/" + js.lstrip("./") + ".map")
        print(f"INFO  source map {js}.map: HTTP {status}"
              + (" (Streamlit's own open-source bundle)" if status == 200 else ""))

    _, hdr, _ = get(base + "/")
    low = {k.lower(): v for k, v in hdr.items()}
    print(f"INFO  server banner: {low.get('server', '(none)')}")
    for h in ("content-security-policy", "x-frame-options", "x-content-type-options",
              "strict-transport-security", "referrer-policy"):
        print(f"INFO  header {h}: {low.get(h, '(not set)')}")
    print(f"INFO  xsrf cookie on first load: {'_streamlit_xsrf' in low.get('set-cookie', '')}")

    print(f"\n{'all security checks pass' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
