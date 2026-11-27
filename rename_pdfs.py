"""
rename_pdfs.py - give the PDFs clean names, because the file name is shown to users as the source.

  1. python rename_pdfs.py           -> writes rename_map.csv (old name, suggested new name)
  2. open rename_map.csv, check / edit the "new" column (leave a row's "new" empty to keep the name)
  3. python rename_pdfs.py --apply   -> renames the PDFs in raw_pdfs/
  4. python clean_pdfs.py            -> old outputs are removed, new names are cleaned
     python app.py --rebuild

Suggestions come from the document title (made by clean_pdfs.py), or from the file name
with obvious problems fixed: "#U2019" -> "'", empty "()", "$", typos like "anval", "FQA".
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

PDF_FOLDER = Path("raw_pdfs")
JSON_FOLDER = Path("output/json")
MAP_FILE = Path("rename_map.csv")
MAX_NAME = 80

TYPOS = {r"\banval\b": "Annual", r"\bforing\b": "Foreign", r"\bissius\b": "Issues",
         r"\bFQA(\d*)\b": r"FAQ\1", r"\bRBi\b": "RBI"}


def safe_name(text):
    """Letters, digits, spaces and - ( ) , . only; plain ASCII; at most MAX_NAME characters."""
    text = text.replace("#U2019", "'").replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.replace("'", "")                                  # Governor's -> Governors
    text = re.sub(r"[\\/:*?\"<>|$#%&{}\[\]]+", " ", text)          # not allowed / ugly in names
    text = re.sub(r"\(\s*\)", " ", text)                           # empty brackets "()"
    text = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))      # "(FAQ )" -> "(FAQ)"
    text = re.sub(r"\)(?=\w)", ") ", text)                         # "(KYC)Direction" -> "(KYC) Direction"
    for pattern, good in TYPOS.items():
        text = re.sub(pattern, good, text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" .-_,")
    if len(text) > MAX_NAME:
        text = text[:MAX_NAME].rsplit(" ", 1)[0]
    return text


def titles_by_file():
    titles = {}
    for f in JSON_FOLDER.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            titles[d["file_name"]] = d.get("title") or ""
        except (OSError, ValueError, KeyError):
            pass
    return titles


def suggest():
    try:
        from clean_pdfs import bad_title
    except Exception:
        bad_title = lambda t, w=(): ""
    titles = titles_by_file()
    used, rows = set(), []
    for pdf in sorted(PDF_FOLDER.glob("*.pdf")):
        title = titles.get(pdf.name, "")
        generic = re.fullmatch(r"(?i)\s*(frequently asked questions|faqs?|press release|notification"
                               r"|master direction|master circular|circular)\s*", title or "")
        base = (safe_name(title) if title and not generic and not bad_title(title) and len(title) <= 120
                else "")
        base = base or safe_name(pdf.stem)
        name, n = base, 2
        while name.lower() in used:                               # two docs with the same title
            name, n = f"{base} ({n})", n + 1
        used.add(name.lower())
        new = name + ".pdf"
        rows.append({"old": pdf.name, "new": "" if new == pdf.name else new, "title": title})
    with MAP_FILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["old", "new", "title"])
        w.writeheader()
        w.writerows(rows)
    changed = [r for r in rows if r["new"]]
    print(f"{len(changed)} of {len(rows)} PDFs get a new name. Check {MAP_FILE}, then run:  "
          f"python rename_pdfs.py --apply\n")
    for r in changed:
        print(f"  {r['old']:<55} -> {r['new']}")


def apply():
    if not MAP_FILE.exists():
        sys.exit(f"{MAP_FILE} not found. Run  python rename_pdfs.py  first.")
    with MAP_FILE.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("new") or "").strip()]
    done = 0
    for r in rows:
        old, new = PDF_FOLDER / r["old"], PDF_FOLDER / r["new"].strip()
        if not new.name.lower().endswith(".pdf"):
            new = new.with_name(new.name + ".pdf")
        if not old.exists():
            print(f"  skip (not found): {old.name}")
        elif new.exists():
            print(f"  skip (name already used): {new.name}")
        else:
            old.rename(new)
            done += 1
            print(f"  {old.name}  ->  {new.name}")
    print(f"\nRenamed {done} PDFs. Now run:  python clean_pdfs.py  and  python app.py --rebuild")


if __name__ == "__main__":
    apply() if "--apply" in sys.argv else suggest()
