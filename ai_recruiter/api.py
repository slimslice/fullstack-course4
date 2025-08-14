from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_loader import load_candidates_csv
from .ranker import rank_candidates


class RankRequest(BaseModel):
	job_description: str = Field(..., min_length=10)
	top_k: int = Field(15, ge=1, le=50)
	data_file: Optional[str] = Field(None, description="Path to candidates CSV. Defaults to data/candidates.csv")


app = FastAPI(title="AI Recruiter", version="0.1.0")


@app.get("/healthz")
async def healthz():
	return {"status": "ok"}


@app.post("/rank")
async def rank(req: RankRequest):
	data_file = req.data_file or "data/candidates.csv"
	path = Path(data_file)
	if not path.exists():
		raise HTTPException(status_code=400, detail=f"data_file not found: {path}")
	candidates = load_candidates_csv(path)
	results = rank_candidates(req.job_description, candidates, top_k=req.top_k)
	return {"results": results, "count": len(results)}