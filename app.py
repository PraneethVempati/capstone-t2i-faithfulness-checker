
import re, traceback, random
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

def _log(msg):
    print(msg, flush=True)

_log(f"Device: {DEVICE}")


PIPE_SD15  = None
PIPE_TURBO = None
CLIP_MODEL = None
CAPTIONER  = None
VQA_MODEL  = None

def get_pipe_sd15():
    global PIPE_SD15
    if PIPE_SD15 is not None: return PIPE_SD15
    _log("Loading SD v1.5 ...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=DTYPE, safety_checker=None,
        requires_safety_checker=False
    ).to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        try: pipe.enable_xformers_memory_efficient_attention()
        except: pass
    PIPE_SD15 = pipe
    return pipe

def get_pipe_turbo():
    global PIPE_TURBO
    if PIPE_TURBO is not None: return PIPE_TURBO
    _log("Loading SDXL-Turbo ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=DTYPE,
        variant="fp16" if DEVICE == "cuda" else None,
    ).to(DEVICE)
    if DEVICE == "cuda": pipe.enable_attention_slicing()
    PIPE_TURBO = pipe
    return pipe

def get_clip():
    global CLIP_MODEL
    if CLIP_MODEL is not None: return CLIP_MODEL
    _log("Loading CLIP ...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    CLIP_MODEL = (model, proc)
    return CLIP_MODEL

def get_captioner():
    global CAPTIONER
    if CAPTIONER is not None: return CAPTIONER
    _log("Loading BLIP captioner ...")
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    model.eval()
    CAPTIONER = (model, proc)
    return CAPTIONER

def get_vqa():
    global VQA_MODEL
    if VQA_MODEL is not None: return VQA_MODEL
    _log("Loading BLIP-VQA ...")
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(DEVICE)
    model.eval()
    VQA_MODEL = (model, proc)
    return VQA_MODEL

def _to_tensor(x):
    if isinstance(x, torch.Tensor): return x
    if hasattr(x, "pooler_output") and isinstance(x.pooler_output, torch.Tensor):
        return x.pooler_output
    if isinstance(x, (tuple, list)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return x[0]
    if hasattr(x, "last_hidden_state") and isinstance(x.last_hidden_state, torch.Tensor):
        return x.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported CLIP output: {type(x)}")

@torch.no_grad()
def clip_image_text_cosine(image: Image.Image, text: str) -> float:
    model, proc = get_clip()
    inputs = proc(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    img_f = _to_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
    txt_f = _to_tensor(model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ))
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return float((img_f * txt_f).sum().item())

@torch.no_grad()
def clip_text_text_cosine(a: str, b: str) -> float:
    model, proc = get_clip()
    inputs = proc(text=[a, b], return_tensors="pt", padding=True).to(DEVICE)
    txt_f = _to_tensor(model.get_text_features(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ))
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return float((txt_f[0] * txt_f[1]).sum().item())

@torch.no_grad()
def blip_caption(image: Image.Image, max_new_tokens: int = 30) -> str:
    model, proc = get_captioner()
    inputs = proc(images=image, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return proc.decode(out[0], skip_special_tokens=True).strip()

@torch.no_grad()
def vqa_ask(image: Image.Image, question: str, max_new_tokens: int = 8) -> str:
    model, proc = get_vqa()
    inputs = proc(image, question, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return proc.decode(out[0], skip_special_tokens=True).strip()


COLORS = {
    "red","blue","green","yellow","orange","purple","pink",
    "black","white","gray","grey","brown","silver","golden"
}
STOPWORDS = set(
    "a an the and or with of in on at to from for is are was "
    "were be being been its their his her".split()
)
SPATIAL_PATTERNS = [
    (r"(\w+)\s+on\s+top\s+of\s+(?:a\s+)?(\w+)",  "on top of"),
    (r"(\w+)\s+next\s+to\s+(?:a\s+)?(\w+)",        "next to"),
    (r"(\w+)\s+in\s+front\s+of\s+(?:a\s+)?(\w+)", "in front of"),
    (r"(\w+)\s+behind\s+(?:a\s+)?(\w+)",           "behind"),
    (r"(\w+)\s+under\s+(?:a\s+)?(\w+)",            "under"),
    (r"(\w+)\s+above\s+(?:a\s+)?(\w+)",            "above"),
    (r"(\w+)\s+beside\s+(?:a\s+)?(\w+)",           "beside"),
    (r"(\w+)\s+near\s+(?:a\s+)?(\w+)",             "near"),
    (r"(\w+)\s+left\s+of\s+(?:a\s+)?(\w+)",        "left of"),
    (r"(\w+)\s+right\s+of\s+(?:a\s+)?(\w+)",       "right of"),
]
COUNT_MAP = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "a":1,"an":1
}
ACTION_VERBS = {
    "wearing","riding","sitting","standing","running","walking",
    "holding","carrying","eating","drinking","playing","sleeping",
    "lying","flying","swimming","jumping","climbing","reading",
    "driving","dancing","cooking","painting"
}

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s)

def _is_yes(ans: str) -> bool:
    a = _norm(ans)
    return a in {"yes","y","true","correct"} or a.startswith("yes ")

def extract_checks(prompt: str, max_checks: int = 8) -> List[Dict[str, Any]]:
    p    = _norm(prompt)
    toks = p.split()
    checks: List[Dict[str, Any]] = []
    seen_q: set = set()

    def add(c):
        if c["q"] not in seen_q and len(checks) < max_checks:
            seen_q.add(c["q"]); checks.append(c)

    for i in range(len(toks) - 1):
        if toks[i] in COLORS and toks[i+1] not in STOPWORDS and toks[i+1].isalpha():
            obj = toks[i+1]
            add({"type":"color", "q":f"What color is the {obj}?", "expect":toks[i]})
            add({"type":"existence", "q":f"Is there a {toks[i]} {obj} in the image?", "expect":"yes"})

    for i in range(len(toks) - 1):
        if toks[i] in COUNT_MAP and toks[i+1] not in STOPWORDS and toks[i+1].isalpha():
            num = COUNT_MAP[toks[i]]; obj = toks[i+1]
            if num > 1:
                add({"type":"count", "q":f"How many {obj} are in the image?", "expect":str(num)})
            add({"type":"existence", "q":f"Is there a {obj} in the image?", "expect":"yes"})

    for pattern, relation in SPATIAL_PATTERNS:
        for m in re.finditer(pattern, p):
            o1, o2 = m.group(1), m.group(2)
            if o1 not in STOPWORDS and o2 not in STOPWORDS:
                add({"type":"spatial", "q":f"Is there a {o1} {relation} the {o2}?", "expect":"yes"})

    for i, t in enumerate(toks):
        if t in ACTION_VERBS and i > 0:
            subj = toks[i-1] if toks[i-1] not in STOPWORDS else (toks[i-2] if i > 1 else "")
            if subj and subj.isalpha():
                add({"type":"action", "q":f"Is someone or something {t} in the image?", "expect":"yes"})

    for t in toks:
        if t.isalpha() and t not in STOPWORDS and t not in COLORS and t not in COUNT_MAP and t not in ACTION_VERBS and len(t) > 2:
            add({"type":"existence", "q":f"Is there a {t} in the image?", "expect":"yes"})

    return checks[:max_checks]

def score_check(check: Dict[str, Any], ans: str) -> int:
    t, exp = check["type"], str(check["expect"]).lower()
    a = _norm(ans)
    if t in ("existence","spatial","action"):
        return 1 if _is_yes(a) else 0
    if t == "color":
        return 1 if re.search(rf"\b{re.escape(exp)}\b", a) else 0
    if t == "count":
        return 1 if re.search(rf"\b{re.escape(exp)}\b", a) else 0
    return 0

@torch.no_grad()
def faithfulness_score(
    prompt: str, image: Image.Image, max_checks: int = 8
) -> Tuple[float, List[List[str]]]:
    checks = extract_checks(prompt, max_checks=max_checks)
    rows, ok = [], 0
    for c in checks:
        ans = vqa_ask(image, c["q"])
        s   = score_check(c, ans)
        ok += s
        rows.append([c["type"], c["q"], str(c["expect"]), ans,
                     "✅ PASS" if s else "❌ FAIL"])
    faith = ok / max(1, len(checks)) if checks else float("nan")
    return faith, [["Type","Question","Expected","Answer","Result"]] + rows


MODEL_OPTIONS = {
    "SD v1.5":    "SD v1.5",
    "SDXL-Turbo": "SDXL-Turbo",
}

@torch.no_grad()
def generate_image(prompt, model_name, seed, steps, guidance, height, width):
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    if model_name == "SDXL-Turbo":
        pipe = get_pipe_turbo()
        return pipe(
            prompt=prompt, num_inference_steps=min(int(steps), 4),
            guidance_scale=0.0, height=int(height), width=int(width), generator=gen
        ).images[0]
    else:
        pipe = get_pipe_sd15()
        return pipe(
            prompt, num_inference_steps=int(steps), guidance_scale=float(guidance),
            height=int(height), width=int(width), generator=gen
        ).images[0]

def to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image): return x.convert("RGB")
    return Image.fromarray(np.array(x)).convert("RGB")

def risk_label(faith: float, clip_score: Optional[float], n_checks: int) -> str:
    if n_checks == 0 or faith != faith: return "⚠️ Unclear — no checks extracted."
    if faith >= 0.85 and (clip_score is None or clip_score >= 0.25): return "🟢 Likely faithful (proxy)"
    if faith >= 0.60: return "🟡 Possible hallucination risk (proxy)"
    return "🔴 Higher hallucination risk (proxy)"

def save_fig(fig, name: str) -> str:
    """
    Saves figure as a large high-DPI PNG.
    Returns the file path so Gradio can offer it as a downloadable image.
    This is how you get a bigger, cleaner version of every plot.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{name}_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


SUGGESTED_PROMPTS = [
    "A red cube on top of a blue sphere on a wooden table",
    "Two dogs wearing sunglasses on a beach",
    "A green apple next to a yellow banana",
    "Three chairs arranged in a living room",
    "A cat sitting under a wooden table",
]

def single_run(
    selected_prompt, custom_prompt, model_name,
    fast_mode, do_clip, do_caption, max_checks,
    seed, steps, guidance, height, width,
    progress=gr.Progress(track_tqdm=False)
):
    try:
        prompt = custom_prompt.strip() or selected_prompt.strip()
        if not prompt:
            return None, "⚠️ Please enter a prompt.", \
                   [["Type","Question","Expected","Answer","Result"]]

        _steps = min(int(steps), 4) if model_name == "SDXL-Turbo" else \
                 min(int(steps), 20) if fast_mode else int(steps)

        progress(0.05, desc="Generating image...")
        img = generate_image(prompt, model_name, seed, _steps, guidance, height, width)

        progress(0.50, desc="Running VQA faithfulness checks...")
        faith, table = faithfulness_score(prompt, img, max_checks=max_checks)

        clip_score = None
        if do_clip:
            progress(0.72, desc="Computing CLIP score...")
            clip_score = clip_image_text_cosine(img, prompt)

        caption, clip_pc = "", None
        if do_caption:
            progress(0.85, desc="Generating BLIP caption...")
            caption = blip_caption(img)
            if do_clip: clip_pc = clip_text_text_cosine(prompt, caption)

        n_checks = max(0, len(table) - 1)
        label    = risk_label(faith, clip_score, n_checks)

        md = f"""### Single-Image Evaluation
**Prompt:** {prompt}
**Model:** {model_name}

| Metric | Value |
|--------|-------|
| **Faithfulness (VQA)** | `{faith:.3f}` ({n_checks} checks) |
| **Risk Assessment** | {label} |"""

        if clip_score is not None: md += f"\n| **CLIP prompt–image** | `{clip_score:.3f}` |"
        if caption:                md += f"\n| **BLIP caption** | *{caption}* |"
        if clip_pc is not None:    md += f"\n| **CLIP prompt–caption** | `{clip_pc:.3f}` |"

        if model_name == "SDXL-Turbo":
            md += "\n\n> ℹ️ SDXL-Turbo: guidance fixed at 0.0, max 4 steps."
        md += "\n\n> ⚠️ Proxy metrics — VQA/CLIP are imperfect evaluators."

        progress(1.0, desc="Done.")
        return img, md, table

    except Exception:
        return None, f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", \
               [["Type","Question","Expected","Answer","Result"]]
    

@torch.no_grad()
def multi_image_consistency_run(
    prompt, model_name, n_images, max_checks,
    steps, guidance, height, width, seed_base,
    do_clip,
    progress=gr.Progress(track_tqdm=False)
):
    try:
        if not prompt.strip():
            return "⚠️ Please enter a prompt.", None, None, []

        _steps = min(int(steps), 4) if model_name == "SDXL-Turbo" else int(steps)
        images, scores, captions, clip_scores = [], [], [], []
        all_checks_tables = []

        for i in range(int(n_images)):
            progress(i / n_images, desc=f"Generating image {i+1}/{n_images}...")
            seed = int(seed_base) + i
            img  = generate_image(prompt, model_name, seed, _steps, guidance, height, width)
            images.append(img)

            progress((i + 0.5) / n_images, desc=f"Scoring image {i+1}/{n_images}...")
            faith, table = faithfulness_score(prompt, img, max_checks=max_checks)
            scores.append(faith)
            all_checks_tables.append(table)

            cap = blip_caption(img)
            captions.append(cap)

            if do_clip:
                clip_scores.append(clip_image_text_cosine(img, prompt))

        scores_arr = np.array([s for s in scores if s == s])
        mean_score = float(scores_arr.mean()) if len(scores_arr) else float("nan")
        std_score  = float(scores_arr.std())  if len(scores_arr) else float("nan")
        variance   = float(scores_arr.var())  if len(scores_arr) else float("nan")
        mean_clip  = float(np.mean(clip_scores)) if clip_scores else None

        if mean_score >= 0.75 and std_score <= 0.15:
            label = "🟢 **Low Variance · High Mean** — Confident and faithful"
            risk  = "LOW hallucination risk"
        elif std_score > 0.20:
            label = "🟡 **High Variance** — Uncertain about this prompt ⚠️"
            risk  = "HIGH hallucination risk"
        else:
            label = "🔴 **Low Variance · Low Mean** — Consistently failing checks"
            risk  = "DEFINITE hallucination ✗"

        score_list = " · ".join([f"`{s:.3f}`" for s in scores])

        md = f"""### Multi-Image Self-Consistency
**Prompt:** {prompt}
**Model:** {model_name} | **Images generated:** {n_images}

| Metric | Value |
|--------|-------|
| **Individual scores** | {score_list} |
| **Mean faithfulness** | `{mean_score:.3f}` |
| **Std deviation** | `{std_score:.3f}` |
| **Variance** | `{variance:.4f}` |"""

        if mean_clip is not None:
            md += f"\n| **Mean CLIP score** | `{mean_clip:.3f}` |"

        md += f"\n\n{label}\n**Risk:** {risk}\n\n#### BLIP Captions Per Image\n"
        for i, cap in enumerate(captions):
            md += f"- Image {i+1}: *{cap}*\n"

        cols_g = min(int(n_images), 3)
        rows_g = (int(n_images) + cols_g - 1) // cols_g
        fig = plt.figure(figsize=(7 * cols_g, 7 * rows_g + 3.5))
        gs  = gridspec.GridSpec(rows_g + 1, cols_g, figure=fig,
                                height_ratios=[4]*rows_g + [2])

        for i, (img, score) in enumerate(zip(images, scores)):
            r, c = divmod(i, cols_g)
            ax   = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            ax.set_title(f"Image {i+1}  |  Faith: {score:.3f}", fontsize=14, fontweight="bold")
            ax.axis("off")

        ax_bar = fig.add_subplot(gs[rows_g, :])
        colors = ["#2ecc71" if s >= 0.75 else "#e67e22" if s >= 0.5 else "#e74c3c"
                  for s in scores]
        bars   = ax_bar.bar(
            [f"Img {i+1}" for i in range(int(n_images))],
            scores, color=colors, edgecolor="k", alpha=0.85
        )
        ax_bar.axhline(mean_score, color="navy", linestyle="--", lw=2,
                       label=f"Mean={mean_score:.3f}")
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_ylabel("Faithfulness Score", fontsize=13)
        ax_bar.set_title(f"Variance={variance:.4f}  |  Std={std_score:.3f}", fontsize=13)
        ax_bar.legend(fontsize=12); ax_bar.grid(True, alpha=0.3, axis="y")
        for bar, s in zip(bars, scores):
            ax_bar.text(bar.get_x()+bar.get_width()/2, s+0.03,
                        f"{s:.3f}", ha="center", fontsize=11, fontweight="bold")

        plt.suptitle(f'Self-Consistency: "{prompt[:70]}"',
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        # Save high-DPI version for download
        plot_path = save_fig(fig, "self_consistency")

        flat_rows = [["Image","Type","Question","Expected","Answer","Result"]]
        for i, table in enumerate(all_checks_tables):
            for row in table[1:]:
                flat_rows.append([f"Img {i+1}"] + row)

        progress(1.0, desc="Done.")
        return md, fig, plot_path, flat_rows

    except Exception:
        return f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", None, None, []
    

def run_pope_benchmark(
    max_rows: int = 200,
    max_checks: int = 6,
    do_clip: bool = True,
    progress=None
):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"pope_benchmark_{timestamp}.csv"

    try:
        if progress: progress(0.02, desc="Loading POPE dataset...")
        ds = load_dataset("lmms-lab/POPE", split="test")

        n = min(max_rows, len(ds))
        ds = ds.select(random.sample(range(len(ds)), n))
        results = []

        for i in range(n):
            try:
                ex       = ds[i]
                question = str(ex["question"]).strip()
                image    = to_pil(ex["image"])
                gt_label = str(ex["answer"]).strip().lower()

                our_answer = vqa_ask(image, question).strip().lower()

                gt_is_yes  = gt_label in {"yes", "y"}
                our_is_yes = our_answer.startswith("yes") or our_answer in {"yes","y"}
                is_correct = int(gt_is_yes == our_is_yes)

                our_faith, _ = faithfulness_score(question, image, max_checks=max_checks)
                is_hallucination = int(not gt_is_yes)

                clip_score = float("nan")
                if do_clip:
                    clip_score = clip_image_text_cosine(image, question)

                results.append({
                    "idx":              i,
                    "question":         question[:200],
                    "gt_answer":        gt_label,
                    "our_answer":       our_answer[:50],
                    "is_correct":       is_correct,
                    "our_faith":        round(our_faith,  4) if our_faith  == our_faith  else None,
                    "clip_score":       round(clip_score, 4) if clip_score == clip_score else None,
                    "is_hallucination": is_hallucination,
                    "category":         ex.get("category", "unknown"),
                })

            except Exception as e:
                _log(f"  Row {i} failed: {str(e)[:80]}")

            if progress:
                progress(0.05 + 0.90*(i+1)/n, desc=f"POPE: {i+1}/{n}")
            elif (i+1) % 50 == 0:
                _log(f"  [{i+1}/{n}] done")

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        _log(f"Saved to {output_csv}")

        summary, fig, plot_path = _build_pope_display(df, n, output_csv)
        return summary, fig, plot_path, df.head(200).values.tolist()

    except Exception:
        return f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", None, None, []


def _build_pope_display(df, n, csv_path):
    valid      = df.dropna(subset=["our_faith","is_hallucination"])
    n_valid    = len(valid)
    accuracy   = df["is_correct"].mean() if "is_correct" in df.columns else float("nan")
    n_hallu    = int(df["is_hallucination"].sum())
    n_faith    = n_valid - n_hallu
    our_v      = valid["our_faith"].values
    hallu_v    = valid["is_hallucination"].values.astype(int)

    roc_auc = ap = float("nan")
    fpr = tpr = None
    if hallu_v.sum() > 0 and (1 - hallu_v).sum() > 0:
        try:
            roc_auc = roc_auc_score(hallu_v, 1 - our_v)
            ap      = average_precision_score(hallu_v, 1 - our_v)
            fpr, tpr, _ = roc_curve(hallu_v, 1 - our_v)
        except Exception as e:
            _log(f"ROC error: {e}")

    if roc_auc == roc_auc and roc_auc >= 0.70:
        auc_txt = f"✅ ROC-AUC = `{roc_auc:.4f}` — strong hallucination detection."
    elif roc_auc == roc_auc and roc_auc >= 0.60:
        auc_txt = f"🟡 ROC-AUC = `{roc_auc:.4f}` — moderate detection."
    elif roc_auc == roc_auc:
        auc_txt = f"🔴 ROC-AUC = `{roc_auc:.4f}` — near-random, needs improvement."
    else:
        auc_txt = "⚠️ ROC-AUC unavailable."

    summary = f"""### POPE Benchmark Results
**Examples:** {n} | **Valid:** {n_valid} | **Saved to:** `{csv_path}`

| Metric | Value |
|--------|-------|
| **VQA Accuracy** | `{accuracy:.4f}` ({accuracy*100:.1f}% correct) |
| **Hallucinated** | `{n_hallu}/{n_valid}` ({100*n_hallu/n_valid:.1f}%) |
| **ROC-AUC** | `{roc_auc:.4f}` |
| **Avg Precision** | `{ap:.4f}` |

{auc_txt}

| | Mean | Std |
|--|------|-----|
| **Faith (faithful)** | `{our_v[hallu_v==0].mean():.3f}` | `{our_v[hallu_v==0].std():.3f}` |
| **Faith (hallucinated)** | `{our_v[hallu_v==1].mean():.3f}` | `{our_v[hallu_v==1].std():.3f}` |
"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"POPE Benchmark — {n_valid} examples", fontsize=16, fontweight="bold")

    hallu_scores    = our_v[hallu_v == 1]
    faithful_scores = our_v[hallu_v == 0]
    axes[0,0].hist(faithful_scores, bins=15, alpha=0.6, color="green",
                   label=f"Faithful (n={n_faith})", edgecolor="k")
    axes[0,0].hist(hallu_scores,    bins=15, alpha=0.6, color="red",
                   label=f"Hallucinated (n={n_hallu})", edgecolor="k")
    axes[0,0].set_xlabel("Our VQA Faithfulness Score", fontsize=13)
    axes[0,0].set_ylabel("Count", fontsize=13)
    axes[0,0].set_title("Score Distribution by Label", fontsize=14)
    axes[0,0].legend(fontsize=12); axes[0,0].grid(True, alpha=0.3)

    # ROC curve
    if fpr is not None:
        axes[0,1].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC={roc_auc:.4f}")
        axes[0,1].plot([0,1],[0,1],"navy","--",alpha=0.6,label="Random (0.5)")
        axes[0,1].set_xlabel("False Positive Rate", fontsize=13)
        axes[0,1].set_ylabel("True Positive Rate", fontsize=13)
        axes[0,1].set_title("ROC Curve — Hallucination Detection", fontsize=14)
        axes[0,1].legend(loc="lower right", fontsize=12); axes[0,1].grid(True, alpha=0.3)
    else:
        axes[0,1].text(0.5,0.5,"ROC unavailable",ha="center",va="center",
                       transform=axes[0,1].transAxes, fontsize=14)

    # Box plot by label
    bp = axes[1,0].boxplot([faithful_scores, hallu_scores],
                            labels=[f"Faithful\n(n={n_faith})", f"Hallucinated\n(n={n_hallu})"],
                            patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#2ecc71")
    if len(bp["boxes"]) > 1: bp["boxes"][1].set_facecolor("#e74c3c")
    axes[1,0].set_ylabel("Our VQA Faithfulness Score", fontsize=13)
    axes[1,0].set_title("Score Distribution by Class", fontsize=14)
    axes[1,0].grid(True, alpha=0.3, axis="y")

    # Category breakdown
    if "category" in df.columns and df["category"].nunique() > 1:
        cats     = df["category"].dropna().unique()
        cat_acc  = [df[df["category"]==c]["is_correct"].mean() for c in cats]
        bar_c    = axes[1,1].bar(cats, cat_acc, color="steelblue", alpha=0.8, edgecolor="k")
        axes[1,1].set_ylabel("VQA Accuracy", fontsize=13)
        axes[1,1].set_title("Accuracy by Category", fontsize=14)
        axes[1,1].set_ylim(0, 1.15)
        axes[1,1].grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bar_c, cat_acc):
            axes[1,1].text(bar.get_x()+bar.get_width()/2, val+0.02,
                           f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    else:
        axes[1,1].scatter(range(len(our_v)), our_v,
                          c=hallu_v, cmap="RdYlGn_r", alpha=0.5, s=20)
        axes[1,1].set_title("Per-example Faith (red=hallucinated)", fontsize=14)
        axes[1,1].set_xlabel("Example index", fontsize=13)
        axes[1,1].set_ylabel("Our VQA Faith Score", fontsize=13)
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_fig(fig, "pope_benchmark")
    return summary, fig, plot_path


def _extract_okvqa_answers(raw_answers) -> List[str]:
    """
    Correctly parses OK-VQA answer formats.
    Handles: list of dicts, list of strings, plain string.
    """
    if raw_answers is None:
        return []

    gt_answers = []

    if isinstance(raw_answers, list):
        for item in raw_answers:
            if isinstance(item, dict):
                ans = item.get("answer", item.get("raw_answer", ""))
                if ans:
                    gt_answers.append(str(ans).strip().lower())
            elif isinstance(item, str):
                if item.strip():
                    gt_answers.append(item.strip().lower())

    elif isinstance(raw_answers, dict):
        nested = raw_answers.get("answers", raw_answers.get("answer", []))
        if isinstance(nested, list):
            for item in nested:
                if isinstance(item, dict):
                    ans = item.get("answer", "")
                    if ans: gt_answers.append(str(ans).strip().lower())
                elif isinstance(item, str):
                    if item.strip(): gt_answers.append(item.strip().lower())
        elif isinstance(nested, str) and nested.strip():
            gt_answers.append(nested.strip().lower())

    elif isinstance(raw_answers, str):
        if raw_answers.strip():
            gt_answers.append(raw_answers.strip().lower())

    return gt_answers


def run_okvqa_benchmark(
    max_rows: int = 200,
    max_checks: int = 6,
    do_clip: bool = True,
    progress=None
):
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"okvqa_benchmark_{timestamp}.csv"

    try:
        if progress: progress(0.02, desc="Loading OK-VQA dataset...")
        ds = load_dataset("Multimodal-Fatima/OK-VQA_train", split="train")

        ex0 = ds[0]
        _log(f"OK-VQA columns: {ds.column_names}")
        _log(f"OK-VQA answers sample: {ex0.get('answers', 'NOT FOUND')}")

        n = min(max_rows, len(ds))
        ds = ds.select(random.sample(range(len(ds)), n))
        results = []

        for i in range(n):
            try:
                ex       = ds[i]
                question = str(ex["question"]).strip()
                image    = to_pil(ex["image"])

                raw_answers = ex.get("answers", ex.get("answer", []))
                gt_answers  = _extract_okvqa_answers(raw_answers)

                if not gt_answers:
                    _log(f"  Row {i}: no GT answers found, raw={raw_answers}")

                our_answer = vqa_ask(image, question).strip().lower()

                is_correct = int(
                    any(
                        gt in our_answer or our_answer in gt
                        for gt in gt_answers
                    )
                ) if gt_answers else 0

                our_faith, _ = faithfulness_score(question, image, max_checks=max_checks)

                clip_score = float("nan")
                if do_clip:
                    clip_score = clip_image_text_cosine(image, question)

                results.append({
                    "idx":        i,
                    "question":   question[:200],
                    "gt_answers": str(gt_answers[:3])[:100],
                    "our_answer": our_answer[:50],
                    "is_correct": is_correct,
                    "our_faith":  round(our_faith,  4) if our_faith  == our_faith  else None,
                    "clip_score": round(clip_score, 4) if clip_score == clip_score else None,
                })

            except Exception as e:
                _log(f"  Row {i} failed: {str(e)[:100]}")

            if progress:
                progress(0.05 + 0.90*(i+1)/n, desc=f"OK-VQA: {i+1}/{n}")
            elif (i+1) % 50 == 0:
                _log(f"  [{i+1}/{n}] done")

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        _log(f"Saved to {output_csv}")

        accuracy  = df["is_correct"].mean() if len(df) else float("nan")
        valid     = df.dropna(subset=["our_faith","is_correct"])
        our_v     = valid["our_faith"].values
        correct_v = valid["is_correct"].values

        try:
            pr, pp   = pearsonr(our_v, correct_v)
            sp, spp  = spearmanr(our_v, correct_v)
        except:
            pr = sp = pp = spp = float("nan")

        n_with_answers = int((df["gt_answers"].apply(lambda x: x != "[]")).sum())

        summary = f"""### OK-VQA Benchmark Results
**Examples:** {n} | **With GT answers:** {n_with_answers} | **Saved to:** `{output_csv}`

| Metric | Value |
|--------|-------|
| **Answer Agreement Rate** | `{accuracy:.4f}` ({accuracy*100:.1f}%) |
| **Pearson r (faith vs correct)** | `{pr:.4f}` (p={pp:.4f}) |
| **Spearman ρ** | `{sp:.4f}` (p={spp:.4f}) |

{'✅ Faithfulness correlates with answer correctness.' if pr > 0.3 else '🔴 Weak correlation — faithfulness score does not strongly predict answer correctness.'}

> **Note:** OK-VQA tests whether our VQA pipeline answers correctly on real images,
> validating that the underlying VQA model is functional on natural scenes.
"""

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"OK-VQA Benchmark — {n} examples", fontsize=16, fontweight="bold")

        corr_scores   = our_v[correct_v == 1]
        incorr_scores = our_v[correct_v == 0]
        axes[0].hist(corr_scores,   bins=15, alpha=0.6, color="green",
                     label=f"Correct (n={len(corr_scores)})", edgecolor="k")
        axes[0].hist(incorr_scores, bins=15, alpha=0.6, color="red",
                     label=f"Incorrect (n={len(incorr_scores)})", edgecolor="k")
        axes[0].set_xlabel("Our Faithfulness Score", fontsize=13)
        axes[0].set_ylabel("Count", fontsize=13)
        axes[0].set_title("Faith Score: Correct vs Incorrect Answers", fontsize=14)
        axes[0].legend(fontsize=12); axes[0].grid(True, alpha=0.3)

        jitter = np.random.normal(0, 0.02, len(our_v))
        axes[1].scatter(our_v, correct_v + jitter, alpha=0.3, s=20, color="steelblue")
        axes[1].set_xlabel("Our Faithfulness Score", fontsize=13)
        axes[1].set_ylabel("Is Correct (jittered)", fontsize=13)
        axes[1].set_title(f"Faith vs Correctness (r={pr:.3f}, ρ={sp:.3f})", fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = save_fig(fig, "okvqa_benchmark")

        return summary, fig, plot_path, df.head(200).values.tolist()

    except Exception:
        return f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", None, None, []
    

def benchmark_run_small(
    dataset_name, split, max_rows, n_examples,
    model_name, fast_mode, do_clip, max_checks,
    seed_base, steps, guidance, height, width,
    progress=gr.Progress(track_tqdm=False)
):
    try:
        if progress: progress(0.02, desc="Loading dataset...")
        ds = load_dataset(dataset_name, split=split)

        cols        = set(ds.column_names)
        image_col   = next((c for c in ["image","img","images"] if c in cols), None)
        caption_col = next((c for c in ["caption","captions","sentences","text"] if c in cols), None)

        if not image_col or not caption_col:
            return (f"❌ Could not find columns. Found: {list(cols)}", None, None, [])

        _steps = min(int(steps), 4)  if model_name == "SDXL-Turbo" else \
                 min(int(steps), 20) if fast_mode else int(steps)

        n       = min(int(n_examples), len(ds), int(max_rows))
        indices = random.sample(range(min(len(ds), int(max_rows))), n)
        gt_scores, gen_scores = [], []
        rows = [["idx","caption","gt_faith","gen_faith","delta","clip_gt","clip_gen"]]

        for i, idx in enumerate(indices):
            ex = ds[idx]
            caption = ex[caption_col]
            if isinstance(caption, list): caption = caption[0]
            caption = str(caption).strip()

            img_gt = to_pil(ex[image_col])
            faith_gt, _ = faithfulness_score(caption, img_gt, max_checks=max_checks)

            seed    = int(seed_base) + idx
            img_gen = generate_image(caption, model_name, seed, _steps, guidance, height, width)
            faith_gen, _ = faithfulness_score(caption, img_gen, max_checks=max_checks)

            c_gt = c_gen = float("nan")
            if do_clip:
                c_gt  = clip_image_text_cosine(img_gt,  caption)
                c_gen = clip_image_text_cosine(img_gen, caption)

            delta = faith_gen - faith_gt
            gt_scores.append(faith_gt); gen_scores.append(faith_gen)
            rows.append([
                idx, caption[:100]+"..." if len(caption)>100 else caption,
                round(faith_gt,3), round(faith_gen,3), round(delta,3),
                round(c_gt,3)  if c_gt  == c_gt  else "N/A",
                round(c_gen,3) if c_gen == c_gen else "N/A",
            ])
            if progress:
                progress(0.08 + 0.88*(i+1)/n, desc=f"Example {i+1}/{n}")

        gt_arr    = np.array(gt_scores)
        gen_arr   = np.array(gen_scores)
        delta_arr = gen_arr - gt_arr
        gt_wins   = int(np.sum(gt_arr > gen_arr))
        gen_wins  = n - gt_wins

        all_scores = np.concatenate([gt_arr, gen_arr])
        all_labels = np.concatenate([np.ones(n), np.zeros(n)])
        roc_auc    = roc_auc_score(all_labels, all_scores)
        fpr, tpr, _ = roc_curve(all_labels, all_scores)

        auc_interp = (
            "✅ Strong discriminative power."  if roc_auc >= 0.70 else
            "🟡 Moderate signal."              if roc_auc >= 0.60 else
            "🔴 Near-random — needs improvement."
        )

        summary = f"""### Live Benchmark Results
**Dataset:** `{dataset_name}` | **Examples:** {n}

| | Mean | Std |
|--|------|-----|
| **Ground-Truth Images** | `{gt_arr.mean():.3f}` | `{gt_arr.std():.3f}` |
| **Generated Images** | `{gen_arr.mean():.3f}` | `{gen_arr.std():.3f}` |
| **Delta (gen − gt)** | `{delta_arr.mean():.3f}` | — |

| Metric | Value |
|--------|-------|
| **ROC-AUC** | `{roc_auc:.4f}` |
| **GT scored higher** | `{gt_wins}/{n}` ({100*gt_wins/n:.1f}%) |

{auc_interp}
"""

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"Live Benchmark — {n} examples | {model_name}", fontsize=16, fontweight="bold")

        axes[0].scatter(gt_scores, gen_scores, alpha=0.7, color="steelblue", edgecolors="k", s=80)
        axes[0].plot([0,1],[0,1],"r--",alpha=0.6,lw=2,label="y=x (equal)")
        axes[0].set_xlabel("GT Faithfulness", fontsize=13); axes[0].set_ylabel("Generated Faithfulness", fontsize=13)
        axes[0].set_title("GT vs Generated", fontsize=14); axes[0].set_xlim(0,1.05); axes[0].set_ylim(0,1.05)
        axes[0].legend(fontsize=12); axes[0].grid(True,alpha=0.3)

        bars = axes[1].bar(["Ground Truth","Generated"],
                           [gt_arr.mean(), gen_arr.mean()],
                           yerr=[gt_arr.std(), gen_arr.std()],
                           capsize=8, color=["#2ecc71","#e74c3c"], alpha=0.8, edgecolor="k", width=0.5)
        axes[1].set_ylabel("Mean Faithfulness", fontsize=13); axes[1].set_title("Mean ± Std", fontsize=14)
        axes[1].set_ylim(0,1.1)
        for bar, mean in zip(bars, [gt_arr.mean(), gen_arr.mean()]):
            axes[1].text(bar.get_x()+bar.get_width()/2, mean+0.03,
                         f"{mean:.3f}", ha="center", fontsize=13, fontweight="bold")
        axes[1].grid(True,alpha=0.3,axis="y")

        axes[2].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {roc_auc:.4f}")
        axes[2].plot([0,1],[0,1],"navy",lw=1.5,linestyle="--",label="Random (AUC=0.50)")
        axes[2].set_xlabel("False Positive Rate", fontsize=13)
        axes[2].set_ylabel("True Positive Rate", fontsize=13)
        axes[2].set_title("ROC Curve", fontsize=14)
        axes[2].legend(loc="lower right",fontsize=12); axes[2].grid(True,alpha=0.3)

        plt.tight_layout()
        plot_path = save_fig(fig, "live_benchmark")

        return summary, fig, plot_path, rows

    except Exception:
        return (f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", None, None, [])


def load_and_display_csv(csv_path: str):
    try:
        df = pd.read_csv(csv_path.strip())
        gt_arr    = df["faith_gt"].dropna().values
        gen_arr   = df["faith_gen"].dropna().values
        delta_arr = gen_arr - gt_arr
        n         = len(df)
        gt_wins   = int(df["gt_wins"].sum()) if "gt_wins" in df.columns else int(np.sum(gt_arr > gen_arr))

        roc_auc = float("nan"); fpr = tpr = None
        try:
            all_scores = np.concatenate([gt_arr, gen_arr])
            all_labels = np.concatenate([np.ones(len(gt_arr)), np.zeros(len(gen_arr))])
            roc_auc    = roc_auc_score(all_labels, all_scores)
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
        except: pass

        summary = f"""### Large-Scale Benchmark (CSV)
**Source:** `{csv_path.strip()}` | **Examples:** {n}

| | Mean | Std |
|--|------|-----|
| **Ground-Truth** | `{gt_arr.mean():.3f}` | `{gt_arr.std():.3f}` |
| **Generated** | `{gen_arr.mean():.3f}` | `{gen_arr.std():.3f}` |
| **Delta** | `{delta_arr.mean():.3f}` | — |
| **ROC-AUC** | `{roc_auc:.4f}` | — |
| **GT scored higher** | `{gt_wins}/{n}` ({100*gt_wins/n:.1f}%) | — |
"""

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f"Large-Scale Benchmark — {n} examples", fontsize=16, fontweight="bold")

        sample = min(2000, n)
        sidx   = np.random.choice(n, sample, replace=False)
        axes[0,0].scatter(gt_arr[sidx], gen_arr[sidx], alpha=0.3, s=20, color="steelblue")
        axes[0,0].plot([0,1],[0,1],"r--",alpha=0.6,lw=2,label="y=x")
        axes[0,0].set_title(f"GT vs Generated (n={sample})", fontsize=14)
        axes[0,0].set_xlabel("GT Faithfulness", fontsize=13); axes[0,0].set_ylabel("Generated", fontsize=13)
        axes[0,0].legend(fontsize=12); axes[0,0].grid(True,alpha=0.3)

        axes[0,1].hist(gt_arr,  bins=20, alpha=0.6, color="green",  label="GT",        edgecolor="k")
        axes[0,1].hist(gen_arr, bins=20, alpha=0.6, color="red",    label="Generated", edgecolor="k")
        axes[0,1].set_title("Score Distribution", fontsize=14)
        axes[0,1].set_xlabel("Faithfulness Score", fontsize=13); axes[0,1].set_ylabel("Count", fontsize=13)
        axes[0,1].legend(fontsize=12); axes[0,1].grid(True,alpha=0.3)

        bars = axes[1,0].bar(["Ground Truth","Generated"],
                             [gt_arr.mean(), gen_arr.mean()],
                             yerr=[gt_arr.std(), gen_arr.std()],
                             capsize=10, color=["#2ecc71","#e74c3c"], alpha=0.8, edgecolor="k")
        axes[1,0].set_ylim(0,1.1); axes[1,0].set_title("Mean ± Std", fontsize=14)
        axes[1,0].set_ylabel("Mean Faithfulness", fontsize=13); axes[1,0].grid(True,alpha=0.3,axis="y")
        for bar, mean in zip(bars,[gt_arr.mean(),gen_arr.mean()]):
            axes[1,0].text(bar.get_x()+bar.get_width()/2, mean+0.03,
                           f"{mean:.3f}", ha="center", fontsize=13, fontweight="bold")

        if fpr is not None:
            axes[1,1].plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC={roc_auc:.4f}")
            axes[1,1].plot([0,1],[0,1],"navy","--",alpha=0.6,lw=1.5,label="Random")
            axes[1,1].set_title("ROC Curve", fontsize=14)
            axes[1,1].legend(loc="lower right",fontsize=12); axes[1,1].grid(True,alpha=0.3)
        else:
            axes[1,1].text(0.5,0.5,"ROC unavailable",ha="center",va="center",
                           transform=axes[1,1].transAxes,fontsize=14)

        plt.tight_layout()
        plot_path = save_fig(fig, "large_benchmark_csv")

        return summary, fig, plot_path, df.head(200).values.tolist()

    except Exception:
        return f"⚠️ Error:\n\n```\n{traceback.format_exc()}\n```", None, None, []
    

with gr.Blocks(title="Image Faithfulness Evaluator") as demo:
    gr.Markdown("""
    # 🖼️ Image Faithfulness Evaluator
    **Tab 1:** Single prompt evaluation.
    **Tab 2:** Multi-image self-consistency.
    **Tab 3:** POPE Benchmark — binary hallucination labels.
    **Tab 4:** OK-VQA Benchmark — answer agreement on real images.
    **Tab 5:** Large-scale Flickr30k benchmark (live or CSV load).

    > **📌 Tip for larger plots:** Click the **⤢ expand icon** in the top-right corner
    > of any plot image to view it fullscreen. Use the **⬇ download button** to save
    > a high-resolution PNG for use in presentations.
    """)

    # ── TAB 1: Single Prompt ─────────────────────────────────
    with gr.Tab("1. Single Prompt Evaluation"):
        with gr.Row():
            with gr.Column(scale=1):
                t1_prompt_pick = gr.Dropdown(
                    choices=SUGGESTED_PROMPTS, value=SUGGESTED_PROMPTS[0],
                    label="Quick prompt suggestions"
                )
                t1_prompt_custom = gr.Textbox(
                    label="Or enter your own prompt (overrides selection)",
                    placeholder="e.g. A red balloon floating in a blue sky"
                )
                t1_model = gr.Radio(
                    choices=list(MODEL_OPTIONS.keys()), value="SD v1.5",
                    label="Generation model",
                    info="SDXL-Turbo = better quality, 1-4 steps, guidance=0.0"
                )
                with gr.Row():
                    t1_fast    = gr.Checkbox(value=False, label="Fast Mode (≤20 steps)")
                    t1_clip    = gr.Checkbox(value=True,  label="CLIP similarity")
                    t1_caption = gr.Checkbox(value=True,  label="BLIP caption")
                t1_max_checks = gr.Slider(1, 12, value=6, step=1, label="Max faithfulness checks")
                with gr.Accordion("Generation settings", open=False):
                    t1_seed     = gr.Slider(0, 10000, value=1234, step=1, label="Seed")
                    t1_steps    = gr.Slider(1, 60,    value=25,   step=1, label="Steps")
                    t1_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
                    t1_height   = gr.Dropdown([512], value=512)
                    t1_width    = gr.Dropdown([512], value=512)
                t1_run = gr.Button("▶ Generate & Evaluate", variant="primary")
            with gr.Column(scale=1):
                t1_img = gr.Image(label="Generated Image", type="pil")
                t1_md  = gr.Markdown()
        t1_table = gr.Dataframe(
            headers=["Type","Question","Expected","Answer","Result"],
            datatype=["str","str","str","str","str"],
            interactive=False, label="VQA Faithfulness Checks"
        )
        t1_run.click(
            fn=single_run,
            inputs=[t1_prompt_pick, t1_prompt_custom, t1_model, t1_fast,
                    t1_clip, t1_caption, t1_max_checks,
                    t1_seed, t1_steps, t1_guidance, t1_height, t1_width],
            outputs=[t1_img, t1_md, t1_table]
        )

    # ── TAB 2: Multi-Image Self-Consistency ──────────────────
    with gr.Tab("2. Multi-Image Self-Consistency"):
        gr.Markdown("""
        Generates **N images from the same prompt** and measures consistency.
        - 🟢 Low variance + high mean → confident and faithful
        - 🟡 High variance → uncertain ⚠️  |  🔴 Low variance + low mean → consistently failing ✗
        """)
        with gr.Row():
            with gr.Column(scale=1):
                mc_prompt     = gr.Textbox(label="Prompt",
                                           value="A red cube on top of a blue sphere on a wooden table")
                mc_model      = gr.Radio(choices=list(MODEL_OPTIONS.keys()), value="SD v1.5", label="Model")
                mc_n_images   = gr.Slider(2, 6, value=3, step=1, label="Number of images")
                mc_max_checks = gr.Slider(1, 12, value=6, step=1, label="Max VQA checks per image")
                mc_clip       = gr.Checkbox(value=True, label="Compute CLIP scores")
                with gr.Accordion("Generation settings", open=False):
                    mc_steps    = gr.Slider(1, 60, value=20, step=1, label="Steps")
                    mc_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
                    mc_seed     = gr.Slider(0, 10000, value=42, step=1, label="Seed base")
                    mc_height   = gr.Dropdown([512], value=512)
                    mc_width    = gr.Dropdown([512], value=512)
                mc_run = gr.Button("▶ Run Self-Consistency", variant="primary")
            with gr.Column(scale=1):
                mc_md   = gr.Markdown()

        # Large plot displayed as image for expand/download
        mc_plot_img  = gr.Image(label="📊 Self-Consistency Plot (click ⤢ to expand, ⬇ to download)",
                                type="filepath")
        mc_plot      = gr.Plot(visible=False)   # kept for compatibility
        mc_table     = gr.Dataframe(
            headers=["Image","Type","Question","Expected","Answer","Result"],
            datatype=["str","str","str","str","str","str"],
            interactive=False, label="Per-image VQA checks"
        )

        def mc_run_wrapper(prompt, model_name, n_images, max_checks,
                           steps, guidance, height, width, seed_base, do_clip,
                           progress=gr.Progress(track_tqdm=False)):
            md, fig, plot_path, rows = multi_image_consistency_run(
                prompt, model_name, n_images, max_checks,
                steps, guidance, height, width, seed_base, do_clip, progress
            )
            return md, plot_path, rows

        mc_run.click(
            fn=mc_run_wrapper,
            inputs=[mc_prompt, mc_model, mc_n_images, mc_max_checks,
                    mc_steps, mc_guidance, mc_height, mc_width, mc_seed, mc_clip],
            outputs=[mc_md, mc_plot_img, mc_table]
        )

    # ── TAB 3: POPE Benchmark ────────────────────────────────
    with gr.Tab("3. POPE Benchmark"):
        gr.Markdown("""
        ### POPE — Binary Hallucination Labels
        Each example: real image + yes/no question → does our VQA predict the label?
        ROC-AUC measures whether faithfulness score predicts hallucinations.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                pope_max_rows   = gr.Slider(50, 500, value=100, step=50, label="Number of examples")
                pope_max_checks = gr.Slider(1, 12, value=6, step=1, label="Max VQA checks")
                pope_clip       = gr.Checkbox(value=True, label="Compute CLIP scores")
                pope_run        = gr.Button("▶ Run POPE Benchmark", variant="primary")
            with gr.Column(scale=1):
                pope_md = gr.Markdown()

        pope_plot_img = gr.Image(label="📊 POPE Plots (click ⤢ to expand, ⬇ to download)",
                                 type="filepath")
        pope_table    = gr.Dataframe(
            headers=["idx","question","gt_answer","our_answer","is_correct",
                     "our_faith","clip_score","is_hallucination","category"],
            interactive=False, label="Per-example results (first 200 rows)"
        )

        def pope_wrapper(max_rows, max_checks, do_clip,
                         progress=gr.Progress(track_tqdm=False)):
            summary, fig, plot_path, table_rows = run_pope_benchmark(
                max_rows, max_checks, do_clip, progress
            )
            return summary, plot_path, table_rows

        pope_run.click(
            fn=pope_wrapper,
            inputs=[pope_max_rows, pope_max_checks, pope_clip],
            outputs=[pope_md, pope_plot_img, pope_table]
        )

    # ── TAB 4: OK-VQA Benchmark ──────────────────────────────
    with gr.Tab("4. OK-VQA Benchmark"):
        gr.Markdown("""
        ### OK-VQA — Answer Agreement on Real Images
        Tests whether our VQA pipeline agrees with human answers on real images.
        High agreement validates that the underlying VQA model is functioning correctly.
        """)
        with gr.Row():
            with gr.Column(scale=1):
                okvqa_max_rows   = gr.Slider(50, 500, value=100, step=50, label="Number of examples")
                okvqa_max_checks = gr.Slider(1, 12, value=6, step=1, label="Max VQA checks")
                okvqa_clip       = gr.Checkbox(value=True, label="Compute CLIP scores")
                okvqa_run        = gr.Button("▶ Run OK-VQA Benchmark", variant="primary")
            with gr.Column(scale=1):
                okvqa_md = gr.Markdown()

        okvqa_plot_img = gr.Image(label="📊 OK-VQA Plots (click ⤢ to expand, ⬇ to download)",
                                  type="filepath")
        okvqa_table    = gr.Dataframe(
            headers=["idx","question","gt_answers","our_answer","is_correct","our_faith","clip_score"],
            interactive=False, label="Per-example results (first 200 rows)"
        )

        def okvqa_wrapper(max_rows, max_checks, do_clip,
                          progress=gr.Progress(track_tqdm=False)):
            summary, fig, plot_path, table_rows = run_okvqa_benchmark(
                max_rows, max_checks, do_clip, progress
            )
            return summary, plot_path, table_rows

        okvqa_run.click(
            fn=okvqa_wrapper,
            inputs=[okvqa_max_rows, okvqa_max_checks, okvqa_clip],
            outputs=[okvqa_md, okvqa_plot_img, okvqa_table]
        )

    # ── TAB 5: Large-Scale Flickr30k ─────────────────────────
    with gr.Tab("5. Large-Scale Benchmark (Flickr30k)"):
        gr.Markdown("""
        **Option A:** Run live benchmark (auto-saves CSV every 50 examples).
        **Option B:** Load a previously saved CSV.
        """)
        with gr.Tabs():

            with gr.Tab("▶ Run Live"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t5a_dataset    = gr.Textbox(label="HF dataset", value="lmms-lab/flickr30k")
                        t5a_split      = gr.Textbox(label="Split", value="test")
                        t5a_max_rows   = gr.Slider(50, 1000, value=200, step=50, label="Max rows to load")
                        t5a_n_examples = gr.Slider(500, 5000, value=10, step=1, label="Examples to evaluate")
                        t5a_model      = gr.Radio(choices=list(MODEL_OPTIONS.keys()), value="SD v1.5", label="Model")
                        with gr.Row():
                            t5a_fast = gr.Checkbox(value=True, label="Fast Mode")
                            t5a_clip = gr.Checkbox(value=True, label="CLIP scores")
                        t5a_checks = gr.Slider(1, 12, value=6, step=1, label="Max VQA checks")
                        with gr.Accordion("Generation settings", open=False):
                            t5a_seed     = gr.Slider(0,10000,value=42,step=1)
                            t5a_steps    = gr.Slider(1,60,value=20,step=1)
                            t5a_guidance = gr.Slider(1.0,15.0,value=7.5,step=0.5)
                            t5a_height   = gr.Dropdown([512],value=512)
                            t5a_width    = gr.Dropdown([512],value=512)
                        t5a_run = gr.Button("▶ Run Live Benchmark", variant="primary")
                    with gr.Column(scale=1):
                        t5a_md = gr.Markdown()

                t5a_plot_img = gr.Image(label="📊 Benchmark Plots (click ⤢ to expand, ⬇ to download)",
                                        type="filepath")
                t5a_table    = gr.Dataframe(
                    headers=["idx","caption","gt_faith","gen_faith","delta","clip_gt","clip_gen"],
                    datatype=["number","str","number","number","number","str","str"],
                    interactive=False, label="Per-example results"
                )

                def t5a_wrapper(dataset_name, split, max_rows, n_examples, model_name,
                                fast_mode, do_clip, max_checks, seed_base, steps, guidance,
                                height, width, progress=gr.Progress(track_tqdm=False)):
                    summary, fig, plot_path, rows = benchmark_run_small(
                        dataset_name, split, max_rows, n_examples, model_name,
                        fast_mode, do_clip, max_checks, seed_base, steps, guidance,
                        height, width, progress
                    )
                    return summary, plot_path, rows

                t5a_run.click(
                    fn=t5a_wrapper,
                    inputs=[t5a_dataset, t5a_split, t5a_max_rows, t5a_n_examples,
                            t5a_model, t5a_fast, t5a_clip, t5a_checks,
                            t5a_seed, t5a_steps, t5a_guidance, t5a_height, t5a_width],
                    outputs=[t5a_md, t5a_plot_img, t5a_table]
                )

            with gr.Tab("📂 Load Saved CSV"):
                t5b_csv_path = gr.Textbox(
                    label="Path to CSV file",
                    placeholder="e.g. benchmark_large_20260406_140000.csv"
                )
                t5b_load = gr.Button("📂 Load & Display", variant="primary")
                with gr.Row():
                    t5b_md = gr.Markdown()

                t5b_plot_img = gr.Image(label="📊 Benchmark Plots (click ⤢ to expand, ⬇ to download)",
                                        type="filepath")
                t5b_table    = gr.Dataframe(interactive=False, label="Results (first 200 rows)")

                def t5b_wrapper(csv_path):
                    summary, fig, plot_path, rows = load_and_display_csv(csv_path)
                    return summary, plot_path, rows

                t5b_load.click(
                    fn=t5b_wrapper,
                    inputs=[t5b_csv_path],
                    outputs=[t5b_md, t5b_plot_img, t5b_table]
                )

demo.queue().launch()
