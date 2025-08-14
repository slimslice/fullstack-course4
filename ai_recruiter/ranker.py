from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .embedding import embed_texts
from .data_loader import Candidate, candidate_to_text


def _tokenize(text: str) -> List[str]:
	if not text:
		return []
	# Alphanumeric word tokens, lowercased
	return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())


def _skill_overlap(job_text: str, candidate_skills: Iterable[str]) -> Tuple[float, List[str]]:
	job_tokens = set(_tokenize(job_text))
	skills_set = set([s.lower() for s in candidate_skills])
	if not job_tokens or not skills_set:
		return 0.0, []
	intersection = [s for s in skills_set if s in job_tokens]
	if not skills_set:
		return 0.0, []
	score = len(intersection) / math.sqrt(len(skills_set))
	return float(score), sorted(intersection)


def _dot(a: List[float], b: List[float]) -> float:
	return float(sum(x * y for x, y in zip(a, b)))


def rank_candidates(
	job_description: str,
	candidates: List[Candidate],
	top_k: int = 15,
	weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
	"""Rank candidates by embedding similarity with a small skill-overlap boost.

	Returns list of dicts with keys: candidate, score, embedding_score, skill_score, matched_skills
	"""
	if not job_description:
		raise ValueError("job_description is required")
	if not candidates:
		return []

	weights = weights or {"embedding": 0.85, "skill_overlap": 0.15}
	w_embed = float(weights.get("embedding", 0.85))
	w_skill = float(weights.get("skill_overlap", 0.15))

	candidate_texts = [candidate_to_text(c) for c in candidates]

	# Get normalized embeddings; cosine similarity reduces to dot product
	emb_job = embed_texts([job_description])[0]
	emb_candidates = embed_texts(candidate_texts)

	similarities = [_dot(vec, emb_job) for vec in emb_candidates]

	ranked: List[Tuple[int, float, float, float, List[str]]] = []
	for idx, candidate in enumerate(candidates):
		emb_score = float(similarities[idx])
		skill_score, matched_skills = _skill_overlap(job_description, candidate.skills)
		total_score = w_embed * emb_score + w_skill * skill_score
		ranked.append((idx, total_score, emb_score, skill_score, matched_skills))

	ranked.sort(key=lambda x: x[1], reverse=True)

	results: List[Dict] = []
	for rank, item in enumerate(ranked[: max(1, top_k)], start=1):
		idx, total, emb_s, sk_s, matched = item
		cand = candidates[idx]
		results.append(
			{
				"rank": rank,
				"score": round(float(total), 6),
				"embedding_score": round(float(emb_s), 6),
				"skill_score": round(float(sk_s), 6),
				"matched_skills": matched,
				"candidate": {
					"candidate_id": cand.candidate_id,
					"name": cand.name,
					"title": cand.title,
					"years_experience": cand.years_experience,
					"location": cand.location,
					"skills": cand.skills,
					"summary": cand.summary,
					"linkedin_url": cand.linkedin_url,
					"email": cand.email,
				},
			}
		)
	return results