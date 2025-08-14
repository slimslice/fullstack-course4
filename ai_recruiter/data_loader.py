from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Candidate:
	candidate_id: str
	name: str
	title: str
	years_experience: Optional[float]
	location: Optional[str]
	skills: List[str]
	summary: Optional[str]
	linkedin_url: Optional[str]
	email: Optional[str]

	def to_dict(self):
		return asdict(self)


def _split_skills(raw: str) -> List[str]:
	if not raw:
		return []
	# Split on comma or pipe, trim, lowercase, remove empty
	parts = re.split(r"[|,]", raw)
	clean = []
	for p in parts:
		val = p.strip().lower()
		if not val:
			continue
		clean.append(val)
	return clean


def load_candidates_csv(csv_path: str | Path) -> List[Candidate]:
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"Candidates CSV not found: {path}")
	candidates: List[Candidate] = []
	with path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			candidate = Candidate(
				candidate_id=str(row.get("candidate_id") or ""),
				name=str(row.get("name") or ""),
				title=str(row.get("title") or ""),
				years_experience=_parse_float(row.get("years_experience")),
				location=str(row.get("location") or ""),
				skills=_split_skills(str(row.get("skills") or "")),
				summary=str(row.get("summary") or ""),
				linkedin_url=str(row.get("linkedin_url") or ""),
				email=str(row.get("email") or ""),
			)
			candidates.append(candidate)
	return candidates


def candidate_to_text(candidate: Candidate) -> str:
	parts: List[str] = []
	if candidate.title:
		parts.append(candidate.title)
	if candidate.skills:
		parts.append("Skills: " + ", ".join(candidate.skills))
	if candidate.summary:
		parts.append(candidate.summary)
	if candidate.years_experience is not None:
		parts.append(f"Experience: {candidate.years_experience} years")
	if candidate.location:
		parts.append(f"Location: {candidate.location}")
	return ". ".join([p for p in parts if p])


def _parse_float(val: Optional[str]) -> Optional[float]:
	if val is None:
		return None
	text = str(val).strip()
	if not text:
		return None
	try:
		return float(text)
	except ValueError:
		return None