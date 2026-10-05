"""Local read-only replay service, stdlib only. Run from repository root."""
from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .replay_v1 import ReplayStore

STATIC = Path(__file__).parent / "static"


def make_server(store, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def send(self, data, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload, status=200):
            self.send(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                host = self.headers.get("Host", "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    self.send_json({"error": "仅允许本机访问"}, 403)
                    return
                url = urlsplit(self.path)
                params = parse_qs(url.query)
                get = lambda key, default="": params.get(key, [default])[0]
                if url.path in ("/", "/app.js", "/style.css"):
                    name, mime = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}[url.path]
                    self.send((STATIC / name).read_bytes(), mime + "; charset=utf-8")
                elif url.path == "/api/catalog":
                    self.send_json(store.catalog())
                elif url.path == "/api/episodes":
                    self.send_json({"episodes": store.listing(get("catalog"), get("policy"), get("seed"), get("area"), get("outcome"), get("q"))})
                elif url.path == "/api/frame":
                    self.send_json(store.frame(get("id"), int(get("step", "0")), get("diagnostic") == "1"))
                elif url.path == "/api/image":
                    data, mime = store.image(get("id"), int(get("step", "0")), get("cell"))
                    self.send(data, mime)
                elif url.path == "/api/export":
                    self.send_json(store.export(get("id")))
                else:
                    self.send_json({"error": "未登记接口"}, 404)
            except (KeyError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)
            except OSError:
                self.send_json({"error": "原件读取失败，请核查本机文件"}, 500)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="GeoNav已保存轨迹回放，零模型调用")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--open-browser", action="store_true", help="核验并启动服务后打开默认浏览器")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    print("正在校验封存轨迹并建立只读目录…", flush=True)
    store = ReplayStore(root)
    print(json.dumps(store.catalog(), ensure_ascii=False), flush=True)
    if args.verify_only:
        return
    server = make_server(store, args.port)
    print(f"平台地址：http://127.0.0.1:{server.server_port}（已保存轨迹回放）", flush=True)
    if args.open_browser:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
