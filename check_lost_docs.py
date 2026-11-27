"""
check_lost_docs.py - review N2: why are some PDFs never found by the chatbot?

  python check_lost_docs.py                 -> lost_docs_report.txt (send it for checking)

For EVERY PDF in raw_pdfs/ it shows where it stops on the way into the database:
  PDF -> skipped by clean_pdfs.py? -> JSON (doc_type, old/current, sections, words) -> chunks in
  the recursive database. Then it looks for the answer words of the 5 failing questions
  (q016, q017, q026, h13, h15) in the PDF text, the JSON text and the database chunks.
Needs no Ollama, nothing is changed.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

import app

PDF_FOLDER = Path("raw_pdfs")
SKIP_LOG = Path("output/skipped_files.csv")
OUT = Path("lost_docs_report.txt")
PROBES = {                                   # failing question -> words its answer page must have
    "q016 SIM swap": ["sim swap", "sim cloning", "duplicate sim"],
    "q017 screen sharing": ["screen sharing"],
    "h15 vishing": ["vishing"],
    "q026 BSBD minimum balance (LAB)": ["minimum balance", "bsbd"],
    "h13 free withdrawals (LAB)": ["free withdrawals", "four withdrawals"],
}

lines = []
say = lambda text="": lines.append(text) or print(text)


def low(text):
    return re.sub(r"\s+", " ", (text or "").lower())


# 1. what clean_pdfs.py made of each PDF
jsons = {}
for f in sorted(app.JSON_FOLDER.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    jsons[d.get("file_name")] = (f, d)
skipped = {}
if SKIP_LOG.exists():
    with SKIP_LOG.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            skipped[r["file_name"]] = f"{r['reason']} {r.get('detail', '')}".strip()

# 2. what is in the recursive database
chunks, db_text = Counter(), {}
if app.is_built("recursive"):
    data = app.get_vector_db(strategy="recursive").get(include=["metadatas", "documents"])
    for m, t in zip(data["metadatas"], data["documents"]):
        chunks[m.get("source")] += 1
        db_text[m.get("source")] = db_text.get(m.get("source"), "") + " " + low(t)
else:
    say("NOTE: the recursive database is not built or out of date: run python app.py --strategy=recursive")

pdf_text = {}
say(f"{'PDF':<62} {'pages':>5} {'JSON':>4} {'type':<14} {'current':<8} {'sect':>5} {'words':>6} "
    f"{'chunks':>6}  problem")
for pdf in sorted(PDF_FOLDER.glob("*.pdf")):
    with pymupdf.open(str(pdf)) as doc:
        pages = len(doc)
        pdf_text[pdf.name] = [low(p.get_text()) for p in doc]
    f, d = jsons.get(pdf.name, (None, None))
    meta = (d or {}).get("metadata", {})
    sections = len((d or {}).get("sections", []))
    words = sum(len(s.get("text", "").split()) for s in (d or {}).get("sections", []))
    n = chunks.get(pdf.name, 0)
    problem = []
    if pdf.name in skipped and (d is None or not n):    # the log keeps OLD entries too
        problem.append(f"SKIPPED by clean_pdfs.py (last log entry): {skipped[pdf.name]}")
    if d is None:
        problem.append("no JSON")
    elif meta.get("doc_type") == "annual_report" and not app.INCLUDE_ANNUAL_REPORTS:
        problem.append("typed annual_report -> left out of the database")
    if d is not None and meta.get("is_latest") is False:
        problem.append(f"marked OLD (stamp={meta.get('stamp')!r}, superseded_by={meta.get('superseded_by')!r}) "
                       "-> hidden unless the question asks for old versions")
    if d is not None and words < 30 * pages:
        problem.append(f"few words for {pages} pages (text lost in cleaning?)")
    if d is not None and not n and "annual_report" not in str(problem):
        problem.append("JSON exists but NO chunks in the database")
    say(f"{pdf.name[:62]:<62} {pages:>5} {'yes' if d else 'NO':>4} {meta.get('doc_type', '-'):<14} "
        f"{str(meta.get('is_latest', '-')):<8} {sections:>5} {words:>6} {n:>6}  {'; '.join(problem)}")

say()
say("Where are the answer words of the failing questions?")
for probe, terms in PROBES.items():
    say(f"\n  {probe}  ({' / '.join(terms)})")
    for name, pages in pdf_text.items():
        hit_pages = [i for i, t in enumerate(pages, start=1) if any(w in t for w in terms)]
        if not hit_pages:
            continue
        f, d = jsons.get(name, (None, None))
        in_json = d is not None and any(w in low(json.dumps(d, ensure_ascii=False)) for w in terms)
        in_db = any(w in db_text.get(name, "") for w in terms)
        say(f"    {name[:60]:<60} PDF pages {hit_pages[:6]}  JSON: {'yes' if in_json else 'NO'}  "
            f"database: {'yes' if in_db else 'NO'}")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nSaved: {OUT}")
