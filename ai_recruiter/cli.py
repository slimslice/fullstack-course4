import argparse
import json
from pathlib import Path
from typing import Optional

from .data_loader import load_candidates_csv
from .ranker import rank_candidates


def _read_text_file(path: Path) -> str:
	return path.read_text(encoding="utf-8")


def main(argv: Optional[list] = None) -> None:
	parser = argparse.ArgumentParser(description="AI Recruiter - rank candidates for a job description")
	parser.add_argument("--job", dest="job", type=str, default=None, help="Job description text")
	parser.add_argument("--job-file", dest="job_file", type=str, default=None, help="Path to job description text file")
	parser.add_argument("--data-file", dest="data_file", type=str, default="data/candidates.csv", help="Path to candidates CSV")
	parser.add_argument("--top-k", dest="top_k", type=int, default=15, help="Number of candidates to return")
	parser.add_argument("--format", dest="fmt", choices=["json", "table"], default="table", help="Output format")

	args = parser.parse_args(argv)

	job_text = args.job
	if args.job_file and Path(args.job_file).exists():
		job_text = _read_text_file(Path(args.job_file))
	if not job_text:
		parser.error("Provide --job or --job-file")

	candidates = load_candidates_csv(args.data_file)
	results = rank_candidates(job_description=job_text, candidates=candidates, top_k=args.top_k)

	if args.fmt == "json":
		print(json.dumps(results, indent=2))
		return

	# table
	for row in results:
		cand = row["candidate"]
		matched = ", ".join(row.get("matched_skills", []))
		print(f"{row['rank']:>2}. {cand['name']} — {cand['title']}  | score={row['score']:.4f}  | matched=[{matched}]")


if __name__ == "__main__":
	main()