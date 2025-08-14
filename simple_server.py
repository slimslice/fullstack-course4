import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ai_recruiter.data_loader import load_candidates_csv
from ai_recruiter.ranker import rank_candidates


WEB_DIR = Path(__file__).parent / "web"
DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "candidates.csv"


class RecruiterRequestHandler(BaseHTTPRequestHandler):
	server_version = "AIRecruiterSimple/0.1"

	def _send_common_headers(self, status: int = 200, content_type: str = "application/json; charset=utf-8") -> None:
		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Access-Control-Allow-Origin", "*")
		self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		self.send_header("Access-Control-Allow-Headers", "Content-Type")
		self.end_headers()

	def do_OPTIONS(self):  # noqa: N802
		self._send_common_headers(200)

	def do_GET(self):  # noqa: N802
		parsed = urlparse(self.path)
		if parsed.path in {"/", "/index.html"}:
			index_path = WEB_DIR / "index.html"
			if not index_path.exists():
				self._send_common_headers(404)
				self.wfile.write(b"Not found")
				return
			content = index_path.read_bytes()
			self._send_common_headers(200, "text/html; charset=utf-8")
			self.wfile.write(content)
			return
		if parsed.path == "/healthz":
			self._send_common_headers(200)
			self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
			return
		# Serve static assets under /web/...
		if parsed.path.startswith("/web/"):
			relative = parsed.path[len("/web/"):]
			file_path = (WEB_DIR / relative).resolve()
			try:
				if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.exists():
					raise FileNotFoundError
			except Exception:
				self._send_common_headers(404)
				self.wfile.write(b"Not found")
				return
			ct = "text/plain; charset=utf-8"
			if file_path.suffix == ".css":
				ct = "text/css; charset=utf-8"
			elif file_path.suffix == ".js":
				ct = "application/javascript; charset=utf-8"
			elif file_path.suffix in {".htm", ".html"}:
				ct = "text/html; charset=utf-8"
			self._send_common_headers(200, ct)
			self.wfile.write(file_path.read_bytes())
			return
		self._send_common_headers(404)
		self.wfile.write(b"Not found")

	def do_POST(self):  # noqa: N802
		parsed = urlparse(self.path)
		if parsed.path == "/rank":
			try:
				length = int(self.headers.get("Content-Length", "0"))
				body = self.rfile.read(length) if length > 0 else b"{}"
				payload = json.loads(body.decode("utf-8"))
				job_description = str(payload.get("job_description") or "").strip()
				if not job_description or len(job_description) < 3:
					raise ValueError("job_description is required")
				top_k = int(payload.get("top_k") or 15)
				data_file = payload.get("data_file")
				data_path = Path(data_file) if data_file else DEFAULT_DATA_FILE
				candidates = load_candidates_csv(data_path)
				results = rank_candidates(job_description=job_description, candidates=candidates, top_k=top_k)
				self._send_common_headers(200)
				self.wfile.write(json.dumps({"results": results, "count": len(results)}).encode("utf-8"))
			except Exception as e:
				self._send_common_headers(400)
				self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
			return
		self._send_common_headers(404)
		self.wfile.write(b"Not found")


def run_server(host: str, port: int) -> None:
	server = HTTPServer((host, port), RecruiterRequestHandler)
	print(f"Serving on http://{host}:{port}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


def main(argv: list | None = None) -> None:
	parser = argparse.ArgumentParser(description="AI Recruiter simple HTTP server")
	parser.add_argument("--host", default="0.0.0.0", help="Bind host")
	parser.add_argument("--port", type=int, default=8000, help="Bind port")
	args = parser.parse_args(argv)
	run_server(args.host, args.port)


if __name__ == "__main__":
	main()