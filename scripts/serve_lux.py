"""Serve a Decision 2.0 package (Decision-2.0-Lux-9B) as /v1/systemone on the loopback interface.

The package ships no server, only ``AutoModel.from_pretrained(..., trust_remote_code=True).system_one``; this wraps
that call unchanged (the runtime fixes the numerics: BF16 autocast with an FP32 head on a GPU). Requests are
answered one at a time.
"""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()

    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.package, trust_remote_code=True, device=args.device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"id": model.model_name}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                state, questions = body["state"], body["questions"]
            except (ValueError, KeyError, TypeError) as failure:
                return self._send(422, {"error": f"bad request: {failure}"})
            try:
                with lock:
                    result = model.system_one(state=state, questions=questions)
            except ValueError as failure:
                return self._send(422, {"error": str(failure)})
            except Exception as failure:  # noqa: BLE001 - reported to the client as a server error
                return self._send(500, {"error": repr(failure)})
            self._send(200, result)

        def log_message(self, *_args) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
