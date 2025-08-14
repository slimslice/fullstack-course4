# AI Recruiter Prototype

An embeddings-based candidate ranking tool. Provide a job description and receive a ranked list of candidates.

## Quickstart

1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Try the CLI

```bash
python -m ai_recruiter.cli \
  --job-file data/sample_job_descriptions/ml_engineer.txt \
  --data-file data/candidates.csv \
  --top-k 15 \
  --format table
```

3) Run the API

```bash
uvicorn ai_recruiter.api:app --host 0.0.0.0 --port 8000
```

- Health check: GET http://localhost:8000/healthz
- Rank: POST http://localhost:8000/rank

Example JSON body:

```json
{
  "job_description": "We are hiring an ML Engineer with Python, PyTorch, AWS, Docker, MLOps.",
  "top_k": 15
}
```

## How it works

- Uses `sentence-transformers/all-MiniLM-L6-v2` to embed the job description and each candidate profile.
- Scores candidates by cosine similarity plus a small boost for skill overlap.
- Returns top-K with scores and matched skills.

## Project structure

- `ai_recruiter/` package with the core logic
- `data/` sample candidates and job description
- `requirements.txt` Python dependencies

## Customize

- Replace `data/candidates.csv` with your own candidate dataset. Keep headers the same:
  - `candidate_id,name,title,years_experience,location,skills,summary,linkedin_url,email`
  - Separate multiple skills with commas or `|`.
- Tune weights in `ai_recruiter/ranker.py`.
- Switch the embedding model via env var `EMBEDDING_MODEL_NAME` (defaults to `sentence-transformers/all-MiniLM-L6-v2`).

## Notes

- First run downloads the embedding model; allow a minute.
- No external API keys required.
