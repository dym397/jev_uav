"""Command-line smoke integration through a real local HTTP boundary."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from benchmark_open_models import main


def test_cli_runs_one_systemone_prompt_arm_and_records_provenance(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert payload["model"] == "test-model"
            assert payload["questions"]["action"]["type"] == "choice"
            probs = {str(i): (1.0 if i == 4 else 0.0) for i in range(9)}
            body = json.dumps({"answers": {"action": {
                "type": "choice", "choice": "4", "probabilities": probs,
            }}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output = tmp_path / "screen"
        main(["--backend", "systemone", "--endpoint",
              f"http://127.0.0.1:{server.server_port}/v1/systemone",
              "--model-id", "test-model", "--split", "smoke",
              "--state-styles", "kinematic_fields", "--question-styles", "balanced",
              "--max-steps", "1", "--output-dir", str(output)])
        report = json.loads((output / "summary.json").read_text())
        metadata = json.loads((output / "run_metadata.json").read_text())
        assert report["arms"][0]["decisions"] == 3
        assert metadata["model_id"] == "test-model"
        assert metadata["state_styles"] == ["kinematic_fields"]
        assert metadata["split"] == "smoke"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
