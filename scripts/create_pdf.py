import argparse
from markdown_pdf import MarkdownPdf,Section
from pathlib import Path
from datetime import datetime
import random
import string
folder = Path("md files")
files = [f for f in folder.iterdir() if f.is_file()]
latest_file = max(files, key=lambda f: f.stat().st_ctime)

with open(latest_file,"r",encoding="utf-8") as f:
    content = f.read()

REPORTS_DIR = Path("../final reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

date_part = datetime.now().strftime("report_%Y-%m-%d")
report_path = REPORTS_DIR / f"{date_part}.pdf"

while report_path.exists():
    random_suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
    report_path = REPORTS_DIR / f"{date_part}_{random_suffix}.pdf"

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(Section(content))

pdf.save(report_path)
