import os
import threading
import hashlib
from typing import Iterable, List, Optional


# Optional: lazy import sentence-transformers only if available
_ST_MODEL = None
_ST_LOCK = threading.Lock()


def _try_load_st_model() -> Optional[object]:
	global _ST_MODEL
	if _ST_MODEL is not None:
		return _ST_MODEL
	with _ST_LOCK:
		if _ST_MODEL is not None:
			return _ST_MODEL
		use_hash = os.getenv("USE_HASH_EMBEDDING", "0").strip() in {"1", "true", "yes"}
		if use_hash:
			return None
		try:
			from sentence_transformers import SentenceTransformer  # type: ignore
			model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
			_ST_MODEL = SentenceTransformer(model_name)
		except Exception:
			# Any failure falls back to hash embedder
			_ST_MODEL = None
	return _ST_MODEL


def _l2_normalize(vector: List[float]) -> List[float]:
	norm = sum(v * v for v in vector) ** 0.5
	if norm == 0.0:
		return vector
	return [v / norm for v in vector]


def _tokenize(text: str) -> List[str]:
	buf: List[str] = []
	cur = []
	for ch in text.lower():
		if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in {"+", "#", "."}:
			cur.append(ch)
		else:
			if cur:
				buf.append("".join(cur))
				cur = []
	if cur:
		buf.append("".join(cur))
	return buf


def _hash_embed(text: str, dim: int = 2048, use_bigrams: bool = True) -> List[float]:
	vec = [0.0] * dim
	tokens = _tokenize(text)
	features: List[str] = tokens[:]
	if use_bigrams and len(tokens) >= 2:
		for i in range(len(tokens) - 1):
			features.append(tokens[i] + "_" + tokens[i + 1])
	for tok in features:
		digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
		idx = int(digest, 16) % dim
		vec[idx] += 1.0
	return _l2_normalize(vec)


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
	"""Return L2-normalized embeddings. Uses sentence-transformers if available, else a hashed bag-of-words embedder.

	This function intentionally avoids hard dependency on external packages so the CLI can run in restricted environments.
	"""
	model = _try_load_st_model()
	text_list = list(texts)
	if model is not None:
		# sentence-transformers path
		try:
			embeddings = model.encode(
				text_list,
				batch_size=64,
				convert_to_numpy=False,
				normalize_embeddings=True,
				show_progress_bar=False,
			)
			# Convert to plain python lists for compatibility
			return [list(map(float, e)) for e in embeddings]
		except Exception:
			# Fallback to hash if encoding fails
			pass

	# Hash embedder path
	dim = int(os.getenv("HASH_EMBED_DIM", "2048"))
	return [_hash_embed(t, dim=dim) for t in text_list]