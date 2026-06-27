# 📊 RedrAI - Candidate Ranking Engine

A highly optimized, hybrid ranking system that evaluates candidate profiles against a Job Description. This engine is built specifically to operate under strict constraints: it evaluates 100k candidates in under a minute on CPU, utilizing precomputed embeddings for the heavy lifting and deterministic rule-based checks for domain relevance.

This is the **INDIA RUNS Hackathon 2026 by Redrob** submission for Team **Pixentropy**. 

## 🏗️ Architecture & Approach

1. **Semantic Similarity**: We use a `sentence-transformers` model (`all-MiniLM-L6-v2`) to compare the narrative text of each candidate's career history against the provided JD. Candidate embeddings are precomputed offline, while the JD is embedded live at runtime.
2. **Skill Matching**: The engine checks exact overlap against the core skills found in the dataset, factoring in proficiency levels, endorsement volume, and usage duration.
3. **Keyword Stuffer Protection**: We aggressively penalize candidates who claim many required skills but do not have a relevant role in their recent career history. 
4. **Behavioral Signals**: We integrate Redrob's behavioral signals to prioritize active, responsive candidates who match location and notice-period requirements.
5. **Honeypot detection**: Flags internally inconsistent profiles (impossible skill-duration claims, experience mismatches, implausible breadth of "expert" skills) and forces them out of the ranking. 
6. **Deterministic Reasoning**: The output reasoning strings are generated without an LLM, pulling directly from computed facts to guarantee zero hallucinations.

## 🚀 Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. (Offline) Precompute Embeddings:**
*Note: This is already done for the provided dataset. Only run this if the `candidates.jsonl` changes.*
```bash
python -m src.precompute_embeddings --jd_file data/raw/job_description.docx
```

**3. (Online) Rank Candidates:**
Generates `submission.csv` in the `output/` directory in under 1 minute.
```bash
python -m src.rank --jd_file data/raw/job_description.txt (or docx file)
```

**4. Validate Output:**
```bash
python validate_submission.py output/submission.csv
```

## 👥 Team Details
### Pixentropy

- Swastik Bose 