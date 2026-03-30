"""
run.py — One-command launcher for OCR Preprocess Studio
Usage:
    python run.py
    python run.py --host 127.0.0.1 --port 8000
"""
import argparse
import webbrowser
import threading
import time

import uvicorn


def _open_browser(host: str, port: int):
    url = f"http://{host}:{port}"
    time.sleep(1.5)          # give uvicorn a moment to bind
    webbrowser.open(url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Preprocess Studio server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,   help="Bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload",     action="store_true", help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Thread(target=_open_browser, args=(args.host, args.port), daemon=True).start()

    print(f"\n  OCR Preprocess Studio")
    print(f"  ─────────────────────────────────────")
    print(f"  URL  : http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
