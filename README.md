# Self-Consistency Uncertainty for Detecting Hallucinations in Text-to-Image Generation

This repository contains my Spring 2026 Capstone project work on using **self-consistency / uncertainty signals** and **structured faithfulness checks** to detect **visual hallucinations** in text-to-image (T2I) generation. The goal is to study when proxy evaluation signals (e.g., VQA-based checks, CLIP-based alignment) correlate with prompt faithfulness, and to validate these evaluators using caption–image benchmarks.

**Student:** Praneeth Vempati  
**Advisor:** Dr. Ivan Ruchkin  
**Course:** Capstone (Spring 2026)  
**TODO:** (optional) Add a 1–2 sentence statement of your latest scope in plain English.

---

## Repository Structure
├── src/ # Core source code (pipelines, evaluators, utilities)
├── notebooks/ # Jupyter notebooks for experiments and demos
├── scripts/ # CLI scripts for running evaluation, sweeps, exports
├── data/ # Dataset pointers and metadata (do NOT commit large files)
├── docs/ # Documentation (design notes, weekly notes, metric definitions)
├── reports/ # Drafts and final report artifacts (PDF/LaTeX exports)
├── assets/ # Images/figures used in docs and reports
└── README.md # Project overview and navigation guide


### Folder Details
- **src/**: Implementation of the main pipeline (generation + evaluation). This includes:
  - prompt processing / claim extraction
  - faithfulness checks (TIFA/GenEval-style structured checks via VQA)
  - semantic alignment metrics (CLIP-based)
  - benchmarking code (running evaluators on ground-truth images vs generated images)
- **notebooks/**: Interactive notebooks used for prototyping, debugging, and visualizing results.
- **scripts/**: Convenience scripts to run experiments end-to-end and save outputs.
- **data/**: Dataset documentation and pointers. Large datasets should not be committed.
- **docs/**: Notes, definitions, meeting summaries, and references used for the project.
- **reports/**: Report drafts and final report materials.
- **assets/**: Figures and diagrams used in documentation and the report.

---

## How to Navigate the Repo

If you want to understand the project quickly:
1. Start with this README.
2. Look at **notebooks/** for the main experimental workflow.
3. Look at **src/** for the reusable pipeline code.
4. Look at **docs/** for metric definitions and project notes.
5. Look at **reports/** for written deliverables.

---

## Quickstart (Local Setup)

> This is a minimal setup; the project is still evolving.

```bash
# Create environment (example)
python -m venv .venv
source .venv/bin/activate # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install -r requirements.txt
