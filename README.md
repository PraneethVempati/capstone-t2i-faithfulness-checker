# Text-to-Image Hallucination Evaluation and Faithfulness Benchmark

## Project Overview

This project investigates hallucinations in text-to-image (T2I) generation models and studies how structured faithfulness evaluation and uncertainty-based signals can be used to detect semantic errors.

The core goal is to evaluate whether generated images correctly reflect the content of their input prompts, and to validate whether faithfulness scores can be meaningfully backed by benchmark datasets containing caption–image ground truth pairs.

Rather than claiming to solve hallucination detection, this project focuses on:

- Constructing structured faithfulness checks (object existence, attributes, counts).
- Measuring semantic alignment using CLIP-based similarity.
- Validating evaluation scores against ground-truth caption–image datasets.
- Analyzing strengths and limitations of proxy evaluation models.

This repository contains all materials related to implementation, benchmarking, documentation, and reporting.

---

## Repository Structure

The repository is organized as follows:
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── generation.py
│   ├── faithfulness.py
│   ├── metrics.py
│   └── benchmark.py
│
├── notebooks/
│   ├── 01_single_image_evaluation.ipynb
│   └── 02_benchmark_validation.ipynb
│
├── data/
│   └── README.md
│
├── docs/
│   ├── proposal.pdf
│   └── presentation.pdf
│
└── reports/
    └── final_report.pdf



---

## Folder Descriptions

### `src/`

Contains the core implementation code.

- `generation.py`  
  Implements single-image generation using diffusion models.

- `faithfulness.py`  
  Implements structured faithfulness checks using VQA-based verification.

- `metrics.py`  
  Implements auxiliary metrics such as:
  - CLIP prompt–image similarity
  - Caption agreement metrics

- `benchmark.py`  
  Implements validation experiments comparing:
  - Faithfulness scores on ground-truth images
  - Faithfulness scores on generated images from the same captions

---

### `notebooks/`

Contains interactive Jupyter notebooks for experimentation.

- `01_single_image_evaluation.ipynb`  
  Demonstrates single-image generation and faithfulness evaluation.

- `02_benchmark_validation.ipynb`  
  Runs validation experiments on caption–image datasets and analyzes score behavior.

---

### `data/`

Contains dataset-related notes and instructions.

Datasets are not stored directly in the repository due to size constraints. Instead, datasets are loaded via Hugging Face Datasets during experiments.

The `README.md` in this folder describes:
- Dataset choices
- Column mappings (image, caption)
- Benchmark configuration

---

### `docs/`

Contains supporting documentation:
- Project proposal
- Presentation slides
- Additional design notes

---

### `reports/`

Contains final and intermediate reports submitted during the semester.

---

## How to Navigate the Repository

1. Start with this README for project context.
2. Review the `docs/` folder for the project proposal and presentation materials.
3. Explore the `src/` directory for implementation details.
4. Use the notebooks in `notebooks/` to reproduce experiments.
5. Refer to `benchmark.py` to understand how evaluation scores are validated.

---

## Evaluation Philosophy

This project emphasizes **construct validity** and empirical backing of evaluation metrics.

Faithfulness scores are not assumed to be accurate by default. Instead, they are validated by:

- Comparing evaluator behavior on ground-truth caption–image pairs.
- Comparing scores between real and generated images.
- Analyzing correlation and distribution differences.

All evaluation metrics are treated as proxy signals and interpreted conservatively.

---

## Requirements

Core dependencies include:

- PyTorch
- Diffusers
- Transformers
- Datasets
- Gradio
- Matplotlib

See `requirements.txt` for details.

---

## Current Status

The repository currently includes:

- Working single-image evaluation pipeline.
- Structured faithfulness checks.
- CLIP-based semantic alignment metrics.
- Initial benchmark validation framework.

Further work includes:
- Expanding hallucination taxonomy.
- Refining claim extraction.
- Performing large-scale validation experiments.
- Reporting statistical analysis.

---

## Collaborators

Instructor and advisor have been added as collaborators to this repository.
