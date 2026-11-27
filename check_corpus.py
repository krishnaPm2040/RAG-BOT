"""
check_corpus.py - shows which documents fill your vector database.

  python check_corpus.py                      -> recursive database
  python check_corpus.py --strategy=semantic  -> another strategy
  python check_corpus.py --titles             -> ONLY corpus_report.md (no database needed)

Always writes corpus_report.md: title, type, circular number, dates and old/current of every
document, so the titles can be checked on GitHub (output/json/ is not committed).

Prints, for every source file: number of chunks and % of all chunks.
Use it to check that the corpus matches DOMAIN in app.py.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import app

app.apply_cli(sys.argv)
REPORT = Path("corpus_report.md")


def write_titles_report():
    """corpus_report.md: one row per document, read from output/json (no database needed)."""
    try:
        from clean_pdfs import bad_title
    except Exception:
        bad_title = lambda t, w=(): ""
    rows, wrong = [], 0
    for f in sorted(app.JSON_FOLDER.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d.get("metadata", {})
        title = (d.get("title") or "").replace("|", "/")
        why = bad_title(title)
        wrong += bool(why)
        in_index = "no (annual report)" if m.get("doc_type") == "annual_report" and not app.INCLUDE_ANNUAL_REPORTS else "yes"
        rows.append(f"| {d['file_name']} | {title} | {m.get('doc_type') or ''} | {m.get('circular_no') or ''} | "
                    f"{m.get('issue_date') or ''} | {m.get('updated_on') or ''} | "
                    f"{'current' if m.get('is_latest', True) else 'OLD (' + (m.get('stamp') or 'superseded') + ')'} | "
                    f"{in_index} | {'**' + why + '**' if why else 'ok'} |")
    if not rows:
        sys.exit(f"No JSON in {app.JSON_FOLDER}/. Run clean_pdfs.py first.")
    text = "\n".join([
        "# Corpus report", "",
        f"Made by `python check_corpus.py` from `{app.JSON_FOLDER}/` (written by `clean_pdfs.py`).",
        f"{len(rows)} documents; titles that look wrong (`bad_title()`): **{wrong}**.", "",
        "| File | Title | Type | Circular no. | Issue date | Updated on | Version | In index | Title check |",
        "|---|---|---|---|---|---|---|---|---|", *rows, ""])
    REPORT.write_text(text, encoding="utf-8")
    print(f"Saved: {REPORT}  ({len(rows)} documents, {wrong} titles look wrong)")


if "--titles" in sys.argv:
    write_titles_report()
    sys.exit()

strategy = app.CHUNK_STRATEGY
if not app.is_built(strategy):
    sys.exit(f"The '{strategy}' database is not built. Run: python app.py --strategy={strategy}")

db = app.get_vector_db(strategy=strategy)
data = db.get(include=["metadatas", "documents"])
metas = data["metadatas"]
counts = Counter(m.get("source", "?") for m in metas)
total = sum(counts.values())

print(f"\nDomain in app.py: {app.DOMAIN}")
print(f"Database: {app.collection_name(strategy)}  ({total} chunks, {len(counts)} files)\n")
kinds = Counter(m.get("doc_type", "?") for m in metas)
print("By document type:")
for kind, n in kinds.most_common():
    print(f"{n:>7} {n * 100 / total:>6.2f}%  {kind}")
print()
print(f"{'chunks':>7} {'share':>6}  file")
for source, n in counts.most_common():
    print(f"{n:>7} {n * 100 / total:>6.2f}%  {source}")

# Chunks with almost no real text (the heading line does not count)
SHORT = 100                                       # the reviewer measured < 100 characters
real = [doc.split("\n", 1)[1] if "\n" in doc else doc for doc in data["documents"]]
short = [(m.get("source", "?"), r) for m, r in zip(metas, real) if len(r.strip()) < SHORT]
long_heads = [m.get("heading", "") for m in metas if len(m.get("heading", "")) > 150]
print(f"\nChunks with less than {SHORT} characters of real text: {len(short)} "
      f"({len(short) * 100 / total:.1f}%)")
for source, n in Counter(s for s, _ in short).most_common(5):
    print(f"{n:>7}  {source}")
print(f"Heading paths longer than 150 characters: {len(long_heads)}")

# Document titles (a title should never be a letterhead, stamp, date, number or "Annexure")
try:
    from clean_pdfs import bad_title
except Exception:                                   # clean_pdfs.py missing / older version
    bad_title = lambda t, w=(): ""
titles = {m.get("source", "?"): m.get("title", "") for m in metas}
bad = {s: (t, bad_title(t)) for s, t in titles.items() if bad_title(t)}
print(f"\nTitles: {len(titles)} files, {len(bad)} look wrong")
for source, (title, why) in sorted(bad.items()):
    print(f"   {why:<22} {title!r:<40} {source}")
print()
write_titles_report()