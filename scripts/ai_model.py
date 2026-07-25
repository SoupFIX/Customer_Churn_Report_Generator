"""
==========================================================================
AI-POWERED CHART ANALYSIS MODULE
==========================================================================
Purpose: Feed generated chart images into a free vision-capable AI model
        (`gemma4:12b`) with a STRUCTURED prompt, so every chart gets a
        consistent, business-relevant written interpretation — turning
        a folder of PNGs into an actual analysis report automatically.

SETUP REQUIRED:
1. Install the SDK:  pip install ollama

WHY A "STRUCTURED PROMPT" MATTERS (core concept of this modssule):
If you just ask a model "describe this chart," you get inconsistent,
rambling answers of varying quality and length. Instead we:
1. Give the model a fixed ROLE ("You are a senior data analyst...")
2. Give it CONTEXT about the dataset (so it doesn't hallucinate meaning)
3. Force a FIXED OUTPUT SCHEMA (same headers every time)
This makes every chart's report comparable, predictable, and easy to
stitch together into one final combined report programmatically.
==========================================================================
"""

import os
import base64
import time
from pathlib import Path
from datetime import datetime
import random
import string
import google.generativeai as genai

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Get a free key at https://aistudio.google.com/app/apikey")
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"
REPORTS_DIR = Path("md files")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

date_part = datetime.now().strftime("report_%Y-%m-%d")
report_path = REPORTS_DIR / f"{date_part}.md"

parent_dir = Path("../charts")
subfolders = [p for p in parent_dir.iterdir() if p.is_dir()]
latest_folder = max(subfolders, key=lambda p: p.stat().st_mtime)

while report_path.exists():
    random_suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
    report_path = REPORTS_DIR / f"{date_part}_{random_suffix}.md"

# --------------------------------------------------------------------
# THE STRUCTURED PROMPT TEMPLATE
# --------------------------------------------------------------------
# WHY put this as a template with placeholders (dataset_context, chart_title)
# rather than a single hardcoded string: every chart needs the SAME
# analytical structure, but different CONTEXT. Separating the fixed
# instructions from the per-chart variables is a core prompt-engineering
# pattern — it keeps your prompt consistent while remaining reusable
# across any chart you throw at it.
# --------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = """You are a senior data analyst reviewing a chart
for a customer churn analysis project. You are given dataset context and a
chart image. Analyze the chart and respond in EXACTLY the structure below —
do not add extra sections, do not skip any section.

DATASET CONTEXT:
{dataset_context}

CHART TITLE: {chart_title}

Respond in this exact markdown structure:

## {chart_title}

**What it shows:**
(1-2 sentences describing literally what's plotted — axes, categories, what
the visual encodes. Be factual, not interpretive here.)

**Key finding:**
(1-2 sentences stating the single most important, specific, quantified
takeaway from this chart. Include actual numbers/percentages visible in
the chart if you can read them.)

**Business implication:**
(1-2 sentences translating the finding into a business action or risk —
e.g. what should a retention team do differently because of this chart.)

**Confidence caveat:**
(1 sentence noting any limitation — e.g. correlation vs causation, sample
size concerns, or what additional data would strengthen this finding.)

Keep the entire response under 200 words. Do not use flowery language.
Be direct and precise, like a report written for a busy VP of Marketing.
At the end provide a correct and most accurate suggestion as per the overall analysis.
"""


# --------------------------------------------------------------------
# DATASET CONTEXT — grounds the model so it doesn't hallucinate
# --------------------------------------------------------------------
# WHY this matters: without context, the model is just guessing at what
# "churn" or "tenure" mean based on the chart alone. Giving it a short,
# accurate description of the dataset dramatically improves grounding
# and reduces hallucinated interpretations.
DATASET_CONTEXT = """
This is the Customer Churn dataset.'Churn' means
the customer has left the company. Key columns: tenure (months as a
customer), monthlycharges (monthly bill in USD), contract (Month-to-month/
One year/Two year), internetservice (DSL/Fiber optic/No), paymentmethod
(Electronic check/Mailed check/Bank transfer/Credit card, automatic).
"""


def encode_image(image_path: str) -> str:
    """
    Converts an image file into a base64-encoded string.

    WHY base64: AI APIs that accept images over HTTP need the raw image
    bytes converted into a text-safe format that can be embedded inside
    a JSON request body. Base64 is the standard way to do this.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_chart(image_path: str, chart_title: str, dataset_context: str = DATASET_CONTEXT) -> str:
    """
    Sends a single chart image + structured prompt to the vision model and returns
    the model's written analysis as a markdown string.
    """
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        dataset_context=dataset_context.strip(),
        chart_title=chart_title,
    )
    # Load image and generate content
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # The generate_content method can take a list of [text, image] parts
        image_part = {
            "mime_type": "image/png",
            "data": open(image_path, "rb").read()
        }
        response = model.generate_content([prompt, image_part])
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")


def analyze_chart_folder(charts_dir: str, output_report_path: str, chart_titles: dict = None):
    """
        Loops through every PNG chart in a folder, sends each one for analysis,
        and stitches all the individual analyses into one combined markdown report.

        chart_titles: optional dict mapping filename -> human-readable title.
            If not given, the filename itself (minus extension) is used.

        WHY loop + stitch rather than one giant multi-image prompt: sending
        charts one at a time keeps each analysis focused and avoids the model
        conflating findings across unrelated charts. It also means if one
        call fails (rate limit, bad image), only that one chart's section
        is missing — not the entire report.
        """
    charts_path = Path(charts_dir)
    png_files = sorted(charts_path.glob("*.png"))

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {charts_dir}")  
    report_sections = []
    report_sections.append("Chart Analysis Report")
    report_sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report_sections.append(f"*Model used: {MODEL_NAME}*\n")

    for i, image_path in enumerate(png_files, 1):
        filename = image_path.stem
        title = (chart_titles or {}).get(image_path.name, filename.replace("_", " ").title())

        print(f"[{i}/{len(png_files)}] Analyzing: {image_path.name} ...................")

        try:
            analysis = analyze_chart(str(image_path), title)
            report_sections.append(analysis)
        except Exception as e:
            print(f"  WARNING: failed to analyze {image_path.name}: {e}")
            report_sections.append(f"## {title}\n\n*Analysis failed: {e}*")

        # WHY sleep: free-tier APIs enforce rate limits (requests per
        # minute). A short pause between calls avoids hitting 429 errors
        # when processing many charts back-to-back.
        time.sleep(2)

    final_report = "\n\n---\n\n".join(report_sections)

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\nReport saved to: {output_report_path}")
    return final_report
# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    analyze_chart_folder(latest_folder,report_path)