# Faithfulness-Based Evaluation for Detecting Hallucinations in Text-to-Image Generation

**Author:** Praneeth Vempati | pvempati@ufl.edu | University of Florida  
**Capstone Project** — AI Systems, 2026

---

## Overview

Text-to-image models frequently generate photo-realistic images that violate prompt
semantics — producing wrong colors, missing objects, or broken spatial relations.
These are called **visual hallucinations**.

This project builds a **training-free, model-agnostic evaluation pipeline** that
automatically detects hallucinations by decomposing text prompts into atomic visual
claims and verifying each against the generated image using Visual Question Answering (VQA).

---

## Key Features

- **VQA-based faithfulness scoring** — decomposes prompts into existence, color, count, spatial, and action checks
- **CLIP semantic similarity** — secondary alignment signal
- **Self-consistency uncertainty signal** — generates N images per prompt and measures score variance
- **POPE benchmark** — validates against 9,000 binary hallucination labels
- **OK-VQA benchmark** — validates VQA pipeline on 9,009 real images with human answers
- **Flickr30k large-scale benchmark** — compares GT vs generated across 1,500+ examples
- **ROC-AUC validation** — rigorously treats faithfulness as a binary hallucination classifier
- **Five-tab Gradio interface** — publicly deployed, supports SD v1.5 and SDXL-Turbo

---

## System Architecture

Prompt → Image Generation (SD v1.5 / SDXL-Turbo) → Smart Prompt Parser (objects, colors, counts, spatial, actions) → BLIP-VQA Verification → Faithfulness Score [0–1] → Risk Label + CLIP Score

**Self-Consistency Branch:**

Same Prompt → N independent generations → per-image scores → variance → uncertainty signal


---

## Models Used

| Component | Model |
|-----------|-------|
| Image Generation | SD v1.5 / SDXL-Turbo |
| VQA Verification | BLIP-VQA (Salesforce/blip-vqa-base) |
| Semantic Similarity | CLIP ViT-B/32 |
| Captioning | BLIP Captioner |

---

## Benchmark Results

| Benchmark | Key Metric | Value |
|-----------|-----------|-------|
| Flickr30k (GT vs Generated) | ROC-AUC | 0.5374 |
| Flickr30k | GT Mean Faithfulness | 0.890 |
| Flickr30k | Gen Mean Faithfulness | 0.877 |
| POPE (Binary Hallucination Labels) | VQA Accuracy | 86.0% |
| POPE | ROC-AUC | **0.814** |
| POPE | Avg Precision | 0.739 |

POPE ROC-AUC of **0.814** confirms the faithfulness metric has strong discriminative
power for predicting hallucinations against human-annotated ground-truth labels.

---

## Installation

```bash
git clone https://github.com/PraneethVempati/capstone-t2i-faithfulness-checker.git
cd capstone-t2i-faithfulness-checker
pip install -r requirements.txt
```

---

## Running the App

``` python app.py ```

Or run the notebook:
``` jupyter notebook notebooks/capstone_final.ipynb ```

The Gradio interface will launch with five tabs:

- Single Prompt Evaluation — generate and score any image
- Multi-Image Self-Consistency — variance-based uncertainty signal
- POPE Benchmark — binary hallucination label evaluation
- OK-VQA Benchmark — VQA pipeline validation
- Large-Scale Flickr30k — GT vs Generated comparison

---

## Project Structure

capstone-t2i-faithfulness-checker/
├── app.py                     # Full Gradio application
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── notebooks/
│   └── capstone_final.ipynb  # Final notebook
├── assets/                    # Result plots and diagrams
└── reports/                   # Mid-project review and final report


---

## References

- Hu et al., TIFA: Text-to-Image Faithfulness Evaluation with QA. ICCV 2023.
- Ghosh et al., GenEval: Object-Focused T2I Alignment. NeurIPS 2023.
- Radford et al., CLIP: Learning Transferable Visual Models. ICML 2021.
- Liu et al., Survey on Hallucination in Vision-Language Models. arXiv 2024.
- Li et al., POPE: Evaluating Object Hallucination in LVLMs. EMNLP 2023.

---
