"""Real decision-response and HTTP boundary checks."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from uav_eval.backends import NanoBackend, SystemOneHTTPBackend, normalize_choice


def probabilities(winner="4"):
    values = {str(i): 0.01 for i in range(9)}
    values[winner] = 0.92
    return values


def test_normalizes_nano_and_systemone_choice_shapes():
    answer = {"type": "choice", "choice": "4", "probabilities": probabilities()}
    systemone = {"answers": {"action": answer}}
    nano = {"states": [{"id": "uav", "answers": {"action": answer}}]}
    expected = {i: probabilities()[str(i)] for i in range(9)}
    assert normalize_choice(systemone) == expected
    assert normalize_choice(nano) == expected


@pytest.mark.parametrize("bad", [
    {"0": 1.0},
    {str(i): 0.2 for i in range(9)},
    {**probabilities(), "4": float("nan")},
    {**probabilities(), "4": -0.1},
])
def test_rejects_incomplete_or_invalid_probability_vectors(bad):
    with pytest.raises(ValueError):
        normalize_choice({"answers": {"action": {"probabilities": bad}}})


def test_nano_adapter_preserves_native_state_batch_contract():
    seen = []

    def evaluate(payload):
        seen.append(payload)
        return {"states": [{"id": "uav", "answers": {"action": {
            "type": "choice", "choice": "4", "probabilities": probabilities(),
        }}}]}

    backend = NanoBackend(evaluate, model_id="NanoJev-test")
    payload = {"state": "uav state", "questions": {"action": {"type": "choice"}}}
    assert backend.predict(payload)[4] == 0.92
    assert seen == [{"states": [{"id": "uav", **payload}]}]


def test_http_adapter_posts_systemone_payload_and_reads_probabilities():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            received.append((self.path, json.loads(self.rfile.read(size))))
            body = json.dumps({"answers": {"action": {
                "type": "choice", "choice": "4", "probabilities": probabilities(),
            }}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/systemone"
        backend = SystemOneHTTPBackend(url, model_id="local-checkpoint")
        payload = {"state": "uav state", "questions": {"action": {"type": "choice"}}}
        assert backend.predict(payload)[4] == 0.92
        assert received == [("/v1/systemone", {"model": "local-checkpoint", **payload})]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
