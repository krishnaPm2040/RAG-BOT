
"""clean_pdfs.py
=============
Day 1 of the RAG project: turn a folder of PDFs into a CLEAN text corpus.

How each document element is handled
------------------------------------
Basic elements
  1. Titles & headings      -> Markdown (#, ##), heading path saved on every section
  2. Tables                 -> Markdown tables + one sentence per row
  3. Lists / bullets        -> "- item" / "1. item", one list kept together
  4. Graphs & images        -> chart text removed, caption kept as "[Figure 1: ...]"
  5. Index / TOC pages      -> removed
  6. Links & references     -> URLs saved as metadata, "References" kept apart
  7. Headers & footers      -> removed (lines repeating on most pages)
  8. Paragraphs             -> broken lines re-joined, hyphenation fixed

Layout and structure
  9.  Multi-column layout   -> columns detected per page, read column by column
  10. Reading order         -> sidebars, text boxes and margin notes taken out of
                               the main text and saved as separate "note" sections
  11. Footnotes             -> "[^1]" markers kept in text, footnote text attached
                               to the section that references it
  12. Tables across pages   -> a table split over 2+ pages is merged into one,
                               repeated header row removed
  13. Merged cells          -> empty merged cells filled (header: from the left,
                               body: from above); tables nested in tables removed
  14. Nested lists          -> list levels kept from indentation ("  - sub item")
  15. Cross-references      -> "para 3.2", "Annex I" found and linked to the
                               heading they point to
  16. Annexures/appendices  -> sections tagged part="annex: <name>"
  17. PDF bookmarks         -> if the PDF has an outline, it is used for headings
                               (more reliable than guessing from font size)

File safety and versions
  29. Encrypted / password  -> doc.needs_pass checked, file skipped and logged
  30. Corrupt / huge files  -> size + page limit, try/except, and a parse TIMEOUT
                               (each PDF is parsed in its own process, killed if stuck)
  31. Version & date        -> issue_date, circular_no, supersedes, superseded_by
                               stored in JSON "metadata"; superseded_by is filled
                               by a second pass across all documents
  32. Duplicates            -> hash of cleaned text, repeats skipped

Page checks and extra text cleaning (run after parse_pdf, before saving)
  18. Scanned pages         -> found per page; OCR'd with ocrmypdf if it is
                               installed, otherwise listed as "ocr_needed_pages"
  19. Hindi / bilingual     -> every section tagged lang = "en" / "hi" / "mixed"
  20. Broken font encoding  -> pages full of (cid:12) / unknown symbols listed
                               as "ocr_needed_pages"
  21. Special symbols       -> font bullets mapped to "-", "` 500" -> "₹ 500"
  22. Formulas              -> sections with many math symbols flagged
                               has_formula = True (text kept)
  23. Watermarks / stamps   -> rotated or large light repeated text removed;
                               a "SUPERSEDED" stamp sets is_latest = false
  24. Cover / disclaimer    -> cover page and disclaimer pages removed
  25. Form fields           -> filled values added as a "form" section
  26. Annotations           -> saved in "annotations", NOT mixed into the text
  27. Address / signature   -> "To ... Madam / Dear Sir" and
                               "Yours faithfully ... name" blocks removed
  28. Rotated pages         -> rotation removed before reading

Incremental runs
  Only NEW or CHANGED PDFs are cleaned again. Unchanged PDFs are skipped,
  and outputs of PDFs deleted from the input folder are removed.
  Change PIPELINE_VERSION after editing the cleaning code, or run with
  --force, to clean every PDF again.

Outputs, for every PDF:
  output/json/<doc_id>.json    structured sections (for chunking + metadata)
  output/markdown/<doc_id>.md  readable clean text (to check by eye)
plus qa_test_set.csv (in the project folder, only if it does not exist yet): a test-set template,
and  output/skipped_files.csv, every file that was skipped and why.

Install:   python -m pip install pymupdf pdfplumber
Run:       python clean_pdfs.py                     (raw_pdfs/ -> output/)
           python clean_pdfs.py --force             (clean every PDF again)
           (optional: --timeout 300 --max-pages 800 --max-size-mb 100)

Note: this needs PDFs with real text. Scanned PDFs need OCR first,
e.g. `ocrmypdf --skip-text input.pdf output.pdf`.
"""

import argparse
import csv
import json
import multiprocessing as mp
import queue
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pdfplumber  # good at finding tables
import pymupdf     # PyMuPDF: text with font sizes/positions, links, images, bookmarks
import hashlib

# ===========================================================================
# Settings you may want to tune for your documents
# ===========================================================================

HEADER_FOOTER_ZONE = 0.08    # top/bottom 8% of the page = header/footer zone
FRAME_ZONE = 0.15            # booklet page frames (title bar, web address) sit a bit lower: 15%
FRAME_LINE_CHARS = 80        # ...and are short lines
REPEAT_THRESHOLD = 0.5       # repeated on >= 50% of pages -> header/footer
HEADING_SIZE_RATIO = 1.15    # font >= 1.15 x body size -> heading
MAX_HEADING_CHARS = 150      # a longer "heading" is really a paragraph in a big font
FRAME_PAGE_SHARE = 0.6       # a "table" covering 60%+ of the page is a decorative page frame
PROSE_CELL_CHARS = 100       # cells this long on average = paragraphs in boxes, not a data table
MAX_HEADER_CELL = 60         # a real column header is short ("Max balance"), not a sentence

FOOTNOTE_SIZE_RATIO = 0.9    # footnotes use a smaller font than body text
FOOTNOTE_ZONE = 0.6          # ...and sit in the bottom 40% of the page

MIN_GUTTER_WIDTH = 8         # empty vertical strip (points) between columns
BACKGROUND_IMAGE_SHARE = 0.5 # an image covering half the page or more = background, not a chart
TEXTBOX_MIN_WORDS = 12       # a drawn box with this many words = text box,
                             # fewer = chart/diagram (its labels are dropped)
TOC_MATCH_SHARE = 0.5        # use bookmarks only if >= 50% of them are found

# Element 30: limits for corrupt / huge files
MAX_SIZE_MB = 100            # bigger files are skipped
MAX_PAGES = 800              # longer files are skipped
PARSE_TIMEOUT_SEC = 3000      # a PDF that takes longer is stopped and skipped. Was 1000: the
                             # BEAWARE booklet (39 pages of pictures) timed out and was lost (N2)

# Element 31: version metadata is read from the start of the document
METADATA_HEAD_CHARS = 4000   # roughly the first page

# Incremental runs: change this number whenever you change the CLEANING code.
# Every PDF cleaned with an older number is cleaned again once.
PIPELINE_VERSION = "2.4"   # 2.4 = exact supersede match (H3), updated_on (M3)
                           # 2.3 = annual report only by cover/title; FAQ by numbered questions; official titles
                           # 1.2 = page_end, 1.3 = cover IDs, doc types, safe dates, stale output, table symbols
                           # 1.4 = infographics: numbers are not headings, long "headings" are text
                           # 1.5 = page frames / prose boxes are not tables; safer cross-page merge
                           # 1.6 = good titles: no letterhead, stamp, watermark, date, number, "Annexure"
                           # 1.7 = doc types: annual reports detected; "business model Our" is not a manual
                           # 1.8 = supersedes = circular numbers / dated references only; FAQ doc type
                           # 1.9 = fix: the title "Sub:" pattern no longer replaces the letter-block one
                           # 2.0 = text printed on a background picture is kept (was dropped as chart labels)
                           # 2.1 = annual reports also detected from the raw cover page (renamed files)
                           # 2.2 = that check really runs now (raw pages 1-3 + PDF title)

TOC_WORDS = re.compile(r"^\s*(table of contents|contents|index)\s*$", re.I)
REFERENCE_WORDS = re.compile(r"^\s*(references|bibliography|works cited|sources)\s*$", re.I)
TOC_LINE = re.compile(r"(\.{3,}|\s{3,}|…+)\s*\d{1,4}\s*$")       # "Intro ..... 12"

BULLET_CHARS = "•●▪■◦○‣∙·–—-*➢➤►▶✓✔❖"
BULLET_LINE = re.compile(rf"^\s*[{re.escape(BULLET_CHARS)}]\s+")
NUMBERED_LINE = re.compile(r"^\s*(\(?\d{1,3}[.)]|\(?[a-z][.)]|\(?[ivxlc]{1,5}[.)])\s+", re.I)
LIST_ITEM_MD = re.compile(r"^\s*(- |\d{1,3}[.)] )")

CAPTION_LINE = re.compile(r"^\s*(fig(ure)?|chart|graph|image|diagram)\.?\s*\d*[:.\-]?\s*", re.I)
URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")

# Element 11: footnote line starts with a marker: "[^1] text", "1 text", "* text"
FOOTNOTE_START = re.compile(r"^\s*(?:\[\^(\w{1,3})\]|(\d{1,3}|[*†‡§])[.)]?\s)\s*")

# Element 15: "para 3.2", "paragraph 4(a)", "Section 5", "Annex II", "Appendix B"
CROSS_REF = re.compile(
    r"\b(para(?:graph)?|section|clause|chapter|rule|annex(?:ure)?|appendix|schedule)"
    r"\.?\s+(?:no\.?\s*)?([0-9]+(?:\.[0-9]+)*(?:\([a-z0-9]+\))*|[IVXLC]+\b|[A-Z]\b)", re.I)

# Element 15: number at the start of a heading: "3.2 Eligibility", "Chapter 4 - ..."
HEADING_NUMBER = re.compile(r"^\s*(?:chapter|section|part)?\s*([0-9]+(?:\.[0-9]+)*)[.)]?\s", re.I)

# Element 16: annexure headings
ANNEX_HEADING = re.compile(r"^\s*(annex(?:ure)?|appendix|schedule|attachment|exhibit)\b\.?\s*([IVXLC]+|[A-Z]|\d+)?", re.I)

# Element 31: dates and circular numbers
DATE_VALUE = (r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{4}"                         # 02/04/2024
              r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9},?\s+\d{4}"     # 2 April 2024
              r"|[A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})")   # April 02, 2024
LABELLED_DATE = re.compile(r"\b(?:Date|Dated|Issued\s+on|Date\s+of\s+issue)\b\s*[:\-]?\s*" + DATE_VALUE, re.I)
LONE_DATE_LINE = re.compile(r"^\s*" + DATE_VALUE + r"\s*$", re.M)
ANY_DATE = re.compile(DATE_VALUE)
UPDATED_AS_ON = re.compile(r"\(\s*updated\s+(?:as\s+)?(?:on|upto|up\s+to)\s+" + DATE_VALUE, re.I)
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"]

RBI_REF_NO = re.compile(r"\bRBI/(?:[A-Za-z]{2,10}/)*\d{4}-\d{2}/\d+\b")   # RBI/2024-25/12, RBI/DOR/2025-26/151
LABELLED_CIRCULAR_NO = re.compile(
    r"\bCircular\s*(?:No\.?|Number)\s*[:\-]?\s*([A-Z0-9][A-Z0-9./()\-]*)", re.I)
SUPERSEDES = re.compile(
    r"\b(?:supersedes|replaces|(?:stands\s+)?in\s+supersession\s+of)\s+"
    r"((?:No\.|Nos\.|\.(?=\S)|[^.\n]){1,200})", re.I)     # keep "No." and "21.04.048"

# Fix A1: a short page that holds an ID (model no., version, RBI ref) is never dropped
# Fix: "supersedes" keeps ONLY real references, never a sentence fragment
DEPT_REF_NO = re.compile(      # DBR.BP.BC.No.45/21.04.048/2024-25, DOR.CRE.REC.12/03.10.001/2025-26
    r"\b[A-Z]{2,6}(?:\.[A-Za-z]{1,10}){1,6}\.?\s?(?:No\.?\s?)?\d+/[\d.]+/\d{4}-\d{2,4}\b")
DATED_REF = re.compile(        # "circular ... dated July 1, 2019", "Master Direction dated 25.02.2016"
    r"\b(?:circular|master\s+circular|direction|master\s+direction|notification|letter|guidelines|instructions)"
    r"\b[^.;]{0,150}?\bdated\s+" + DATE_VALUE, re.I)

IDENTIFIER = re.compile(
    r"\b(?:[A-Z]{1,6}-[A-Z0-9]{2,}[A-Z0-9-]*"          # AP-X200, SM-G991B
    r"|[A-Z]{2,6}\d{2,}[A-Z0-9]*"                      # XR500, KDL40W
    r"|v\d+(?:\.\d+)+"                                # v2.1
    r"|RBI/(?:[A-Za-z]{2,10}/)*\d{4}-\d{2}/\d+)\b")   # RBI/2024-25/12, RBI/DOR/2025-26/151

# Fix A2: document types. Each type says how to detect it and which fields it must have.
DOC_TYPES = {                  # checked in this order; the first match wins
    "annual_report": {         # "Annual Report 2025-26", "Annual Report FY2026", "annual report for the year 2025"
        "detect": re.compile(r"\b(?:integrated\s+)?annual\s+report\b\s*(?:for\s+the\s+(?:financial\s+)?year\s*)?"
                             r"(?:FY\s*)?[\d]{2,4}(?:\s*[-–/]\s*\d{2,4})?"
                             r"|\b(?:audited|unaudited)\s+(?:standalone\s+|consolidated\s+)?financial\s+results\b",
                             re.I),
        "required": [],
    },
    "faq": {                   # FAQs are not circulars: no circular number / date needed
        "detect": re.compile(r"\bFrequently\s+Asked\s+Questions\b|\bFAQs?\b", re.I),
        "required": [],
    },
    "rbi_circular": {
        "detect": re.compile(r"Reserve Bank of India|\bRBI/(?:[A-Za-z]{2,10}/)*\d{4}-\d{2}/\d+"
                             r"|Master Direction", re.I),
        "required": ["circular_no", "issue_date"],
    },
    "manual": {
        # "User Manual", or a LABELLED model code WITH a digit: "Model No: AP-X200", "Model: X200".
        # ("business model Our ..." is NOT a manual: no label, no digit.)
        "detect": re.compile(r"\b(?:user|owner'?s|installation|service|operating|instruction)\s+"
                             r"(?:manual|guide)\b"
                             r"|\bmodel\s*(?:no\.?|number|#|[:\-])\s*[:\-]?\s*[A-Z]{0,6}[-/]?[A-Z]*\d[A-Z0-9\-/]*",
                             re.I),
        "required": ["model"],
    },
}
MODEL_NO = re.compile(r"\bModel\s*(?:No\.?|Number)?\s*[:\-]?\s*"
                      r"((?=[A-Z0-9/-]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+)+|[A-Z]{1,6}\d{2,}[A-Z0-9]*)", re.I)
                                                # a model code always has a digit
DOC_VERSION = re.compile(r"\b(?:v|ver\.?|version|rev(?:ision)?\.?)\s*(\d+(?:\.\d+)*)\b", re.I)
PRODUCT_NAME = re.compile(r"^[#\s]*(.{3,80}?)\s*[|\-:]?\s*\b(?:user|owner'?s|installation|service|"
                          r"operating|instruction)\s+(?:manual|guide)\b", re.I | re.M)
BRAND = re.compile(r"(?:©|\(c\)|copyright)\s*(?:\d{4}(?:\s*-\s*\d{4})?\s*,?\s*)?"
                   r"([A-Z][A-Za-z0-9&.\- ]{1,40}?)(?=\s*(?:[.,]|all rights|$))", re.I | re.M)

# Fix A3: footnote lines in the Markdown look like "[^1]: as per circular dated ..."
FOOTNOTE_TEXT_LINE = re.compile(r"^\s*\[\^[^\]]+\]:.*$", re.M)


# ===========================================================================
# Step 0: small helpers
# ===========================================================================

def normalise_text(text: str) -> str:
    """Element 8: fix the small character problems PDFs create."""
    text = unicodedata.normalize("NFKC", text)          # ligatures: "ﬁ" -> "fi"
    text = (text.replace("“", '"').replace("”", '"')
                .replace("’", "'").replace("‘", "'"))
    text = re.sub("[​‌‍­﻿]", "", text)  # invisible chars
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def digits_to_hash(text: str) -> str:
    """'Page 3 of 20' -> 'page # of #', so changing page numbers still match."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def norm_key(text: str) -> str:
    """Lowercase letters/digits only, to compare headings with bookmarks."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def inside(inner, outer, tol=2) -> bool:
    """True if box `inner` (x0, y0, x1, y1) lies inside box `outer`."""
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and
            inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def overlap_share(a, b) -> float:
    """Share of box a's area that overlaps box b."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    return max(w, 0) * max(h, 0) / area


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0
    return values[min(len(values) - 1, int(len(values) * p))]


# ===========================================================================
# Step 1: read every line with font size, boldness, position
# ===========================================================================

def read_lines(doc):
    """Return a list of pages; each page is a list of line dicts:
       {text, size, bold, bbox, page}

    Element 11: small raised numbers (superscripts) are footnote markers.
    They are written as "[^1]" so the link to the footnote is not lost."""
    pages = []
    for page_no, page in enumerate(doc, start=1):
        lines = []
        for b_no, block in enumerate(page.get_text("dict")["blocks"]):
            if block["type"] != 0:                     # 1 = image block
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                max_size = max(s["size"] for s in spans)
                baseline = max(s["origin"][1] for s in spans)
                parts, normal_sizes = [], []
                for s in spans:
                    t = s["text"]
                    # superscript = PyMuPDF flag bit 1, or small + raised text
                    raised = s["origin"][1] < baseline - 1 and s["size"] < 0.8 * max_size
                    if (s["flags"] & 1 or raised) and re.fullmatch(r"\s*[\d*†‡§]{1,3}\s*", t):
                        parts.append(f"[^{t.strip()}] " if not parts else f"[^{t.strip()}]")
                    else:
                        parts.append(t)
                        normal_sizes.append(s["size"])
                text = normalise_text("".join(parts))
                size = round(max(normal_sizes or [max_size]), 1)
                bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
                lines.append({"text": text, "size": size, "bold": bold,
                              "bbox": tuple(line["bbox"]), "page": page_no,
                              "block": f"{page_no}-{b_no}"})   # PDF text block id
        pages.append(merge_lone_markers(lines))
    return pages


LONE_MARKER = re.compile(rf"^\s*([{re.escape(BULLET_CHARS)}]|\(?\d{{1,3}}[.)]|\(?[a-z][.)]|\(?[ivxlc]{{1,5}}[.)])\s*$", re.I)


def merge_lone_markers(lines):
    """Elements 3 and 14: many PDFs draw the bullet ("•", "(a)", "1.") as a
    separate piece of text to the left of the item. Join it with the text on
    the same baseline, keeping the bullet's x position (needed for nesting)."""
    used, out = set(), []
    for i, ln in enumerate(lines):
        if i in used:
            continue
        if LONE_MARKER.match(ln["text"]):
            mid = (ln["bbox"][1] + ln["bbox"][3]) / 2
            for j, other in enumerate(lines):
                if j == i or j in used:
                    continue
                o_mid = (other["bbox"][1] + other["bbox"][3]) / 2
                gap = other["bbox"][0] - ln["bbox"][2]
                if abs(o_mid - mid) < 3 and 0 <= gap < 40:
                    used.add(j)
                    ln = {**other, "text": ln["text"].strip() + " " + other["text"],
                          "bbox": (ln["bbox"][0], other["bbox"][1], other["bbox"][2], other["bbox"][3])}
                    break
        out.append(ln)
    return out


# ===========================================================================
# Step 2 (Element 7): headers and footers
# ===========================================================================

def in_header_footer_zone(bbox, page_height, zone=HEADER_FOOTER_ZONE) -> bool:
    return bbox[3] < page_height * zone or bbox[1] > page_height * (1 - zone)


def in_frame_zone(line, page_height) -> bool:
    """Short line in the top/bottom 15%: may be a repeated page frame ("Office of RBI Ombudsman")."""
    return len(line["text"]) <= FRAME_LINE_CHARS and in_header_footer_zone(line["bbox"], page_height, FRAME_ZONE)


def find_headers_footers(doc, pages):
    """Text in the top/bottom zone that repeats on most pages = header/footer."""
    counter = Counter()
    for page, lines in zip(doc, pages):
        counter.update({digits_to_hash(ln["text"]) for ln in lines
                        if in_frame_zone(ln, page.rect.height)})
    min_pages = max(2, int(len(pages) * REPEAT_THRESHOLD))
    return {key for key, n in counter.items() if n >= min_pages}


def is_header_footer(line, page_height, repeated) -> bool:
    if not in_frame_zone(line, page_height):
        return False
    key = digits_to_hash(line["text"])
    if in_header_footer_zone(line["bbox"], page_height) and \
            re.fullmatch(r"[\-\s]*(page\s*)?#(\s*of\s*#)?[\-\s]*", key):   # bare page numbers
        return True
    return key in repeated


# ===========================================================================
# Step 3 (Element 5): contents / index pages
# ===========================================================================

def is_toc_page(lines) -> bool:
    if not lines:
        return False
    if any(TOC_WORDS.match(ln["text"]) for ln in lines[:5]):
        return True
    toc_like = sum(1 for ln in lines if TOC_LINE.search(ln["text"]))
    return toc_like >= 5 and toc_like / len(lines) > 0.5


# ===========================================================================
# Step 4 (Elements 1 and 17): headings from font size OR from PDF bookmarks
# ===========================================================================

def font_statistics(pages):
    """Body size = most common font size (weighted by amount of text).
    Bigger sizes become heading levels 1, 2, 3."""
    sizes = Counter()
    for lines in pages:
        for ln in lines:
            sizes[ln["size"]] += len(ln["text"])
    if not sizes:
        return 10, {}
    body = sizes.most_common(1)[0][0]
    big = sorted({s for s in sizes if s >= body * HEADING_SIZE_RATIO}, reverse=True)
    return body, {s: min(i + 1, 3) for i, s in enumerate(big)}


# Infographics: "38%", "~14%", "₹6,933 crores", "2.3x", "Rs 450 cr" are big numbers, NOT headings
NUMBER_ONLY = re.compile(
    r"^\s*[^\w\s]*\s*(?:rs\.?|inr|usd|us\$)?\s*[^\w\s]*\s*\d[\d.,]*\s*"
    r"(?:%|per\s*cent|crores?|cr\.?|lakhs?|lacs?|bn|billion|mn|million|thousand|k|x|times|bps)?\s*[)+]*\s*$",
    re.I)


def looks_like_heading_text(text) -> bool:
    """A heading needs real words: at least one word of 3+ letters, and not just a number."""
    return bool(re.search(r"[A-Za-z]{3,}", text)) and not NUMBER_ONLY.match(text)


def sentence_like(text) -> bool:
    """'In 2021, under the oversight of the Board, the bank expanded ...' is a sentence:
    headings almost never have 10+ small (lower-case) words."""
    return len(re.findall(r"\b[a-z][a-z]+\b", text)) >= 10


def font_heading_level(line, levels) -> int:
    text = line["text"]
    if len(text) > 120 or (text.endswith((".", ",", ";")) and len(text) > 60):
        return 0
    if not looks_like_heading_text(text):            # "38%", "₹6,933 crores" -> normal text
        return 0
    if line["size"] in levels:
        return levels[line["size"]]
    if line["bold"] and len(text) < 80 and not BULLET_LINE.match(text) and not FOOTNOTE_START.match(text):
        return 3                                     # short bold line, e.g. "3.2 Eligibility"
    return 0


class BookmarkHeadings:
    """Element 17: the PDF outline (bookmarks) lists every heading with its
    level and page. When present, it is far more reliable than font sizes.

    We match each text line against the bookmark titles on that page (or the
    page before/after, since bookmark pages are sometimes off by one). Long
    headings can wrap onto 2-3 lines, so a line that is the START of a title
    is joined with the following lines until the full title is matched."""

    def __init__(self, doc):
        self.by_page = defaultdict(list)
        self.total = 0
        for level, title, page in doc.get_toc(simple=True):
            if title.strip() and page > 0:
                self.by_page[page].append({"key": norm_key(title), "level": level,
                                           "title": normalise_text(title), "used": False})
                self.total += 1
        self.matched = 0

    def available(self) -> bool:
        return self.total >= 2

    def match(self, items, i, page_no):
        """Try to match items[i] (and maybe the next lines) to a bookmark.
        Returns (level, title, number_of_lines_used) or None."""
        first = items[i]
        if first["kind"] != "line":
            return None
        key = norm_key(first["text"])
        if len(key) < 3:
            return None
        for p in (page_no, page_no - 1, page_no + 1):
            for entry in self.by_page.get(p, []):
                if entry["used"] or not entry["key"].startswith(key):
                    continue
                combined, used = key, 1
                # heading wrapped onto several lines: keep adding lines
                while combined != entry["key"] and i + used < len(items) and used < 4:
                    nxt = items[i + used]
                    if nxt["kind"] != "line":
                        break
                    candidate = combined + norm_key(nxt["text"])
                    if not entry["key"].startswith(candidate):
                        break
                    combined, used = candidate, used + 1
                if combined == entry["key"]:
                    entry["used"] = True
                    self.matched += 1
                    return entry["level"], entry["title"], used
        return None


# ===========================================================================
# Step 5 (Elements 2, 12, 13): tables
# ===========================================================================

def clean_table(rows):
    """Element 13: merged cells.

    pdfplumber returns None for cells that are covered by a merged cell.
      - in the header row a merged cell usually spans columns -> fill from LEFT
      - in body rows it usually spans rows (e.g. a category) -> fill from ABOVE
    Real empty cells come back as "" and are left empty."""
    rows = [list(r) for r in rows if r and any(c not in (None, "") for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    header_was_merged = any(c is None for c in rows[0][1:])
    for r_i, row in enumerate(rows):
        for c_i, cell in enumerate(row):
            if cell is not None:
                continue
            if r_i == 0:
                row[c_i] = row[c_i - 1] if c_i > 0 else ""
            elif rows[r_i - 1][c_i] not in (None, ""):
                row[c_i] = rows[r_i - 1][c_i]
            else:
                row[c_i] = row[c_i - 1] if c_i > 0 else ""
    rows = [[normalise_text(str(c or "")).replace("\n", " ").replace("|", "/") for c in r] for r in rows]

    # Two header rows: "Limits" spans 2 columns, and the row below names them
    # ("Balance", "Credits per year"). Combine into one header row:
    # "Limits - Balance", "Limits - Credits per year".
    if len(rows) >= 3 and header_was_merged and not any(re.search(r"\d", c) for c in rows[1]):
        rows[0] = [a if a == b or not b else (f"{a} - {b}" if a else b) for a, b in zip(rows[0], rows[1])]
        del rows[1]
    return rows


def table_to_markdown(rows) -> str:
    header, body = rows[0], rows[1:]
    md = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def table_rows_as_text(rows):
    """Element 2: one sentence per row, e.g.
    'Account type: Small account; Max balance: Rs 50,000'.
    Embeddings match these far better than a Markdown grid."""
    header = rows[0]
    if any(len(h) > MAX_HEADER_CELL for h in header):          # not a real header: no labels
        return ["; ".join(v for v in r if v) for r in rows]
    return ["; ".join(f"{h}: {v}" if h and h != v else v for h, v in zip(header, r) if v)
            for r in rows[1:]]


def is_data_table(rows, bbox, page_w, page_h) -> bool:
    """False for things pdfplumber calls a table but are not:
      - a page FRAME (border around the whole page, e.g. booklets with a title bar)
      - PROSE in boxes (cells full of sentences, not short values)
      - a header row that is a sentence or a URL"""
    x0, top, x1, bottom = bbox
    if (x1 - x0) * (bottom - top) >= FRAME_PAGE_SHARE * page_w * page_h:
        return False                                          # covers most of the page: a frame
    cells = [c for r in rows for c in r if c.strip()]
    if not cells or sum(len(c) for c in cells) / len(cells) > PROSE_CELL_CHARS:
        return False                                          # paragraphs, not values
    header = [c for c in rows[0] if c.strip()]
    if any(len(c) > MAX_HEADER_CELL or re.search(r"https?://|www\.", c) for c in header):
        return False                                          # header is a sentence / web address
    return True


def extract_tables(pdf_path):
    """Return {page_number: [{"bbox", "rows"}, ...]}."""
    tables = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            found = []
            for t in page.find_tables():
                rows = clean_table(t.extract())
                # 1 row or 1 column is not a real table (often a framed text box)
                if len(rows) >= 2 and len(rows[0]) >= 2 and is_data_table(
                        rows, t.bbox, float(page.width), float(page.height)):
                    found.append({"bbox": tuple(t.bbox), "rows": rows})
            # Element 13: a table found INSIDE another table is a nested table;
            # its text is already in the outer table's cell, so drop it
            found = [t for t in found
                     if not any(o is not t and inside(t["bbox"], o["bbox"]) and o["bbox"] != t["bbox"]
                                for o in found)]
            if found:
                tables[page_no] = found
    return tables
def add_page_end(sections):
    """Every section gets page_end = the last page it covers.
    Text: from its lines. Tables: from the pages they were merged over."""
    for s in sections:
        pages = s.get("pages") or [s["page"]]
        s["page_end"] = max([s.get("page_end") or s["page"], *pages])
    return sections

def merge_tables_across_pages(sections):
    """Element 12: a long table continues on the next page. pdfplumber sees
    two separate tables. We merge them when:
      - table A is the last content on page N,
      - table B is the first content on page N+1,
      - both have the same number of columns.
    If B starts with the same header row as A (repeated header), it is dropped."""
    content = ("text", "table", "heading")
    merged = []
    for s in sections:
        if s["type"] == "table" and merged:
            # previous content section (skip notes)
            prev = next((m for m in reversed(merged) if m["type"] in content), None)
            first_on_page = not any(m["page"] == s["page"] and m["type"] in content
                                    for m in merged)
            same_header = s["rows"][0] == prev["rows"][0] if prev and prev["type"] == "table" else False
            looks_continued = (prev and prev["type"] == "table"
                               and not any(re.search(r"[A-Za-z]{3,}", c) and not re.search(r"\d", c)
                                           for c in s["rows"][0]))   # B starts with data, not a header
            if (prev and prev["type"] == "table" and first_on_page
                    and prev["pages"][-1] == s["page"] - 1
                    and len(prev["rows"][0]) == len(s["rows"][0])
                    and (same_header or looks_continued)):
                rows = s["rows"]
                if rows[0] == prev["rows"][0]:
                    rows = rows[1:]                  # repeated header row
                prev["rows"].extend(rows)
                prev["pages"].append(s["page"])
                continue
        merged.append(s)
    for s in merged:                                  # rebuild text after merging
        if s["type"] == "table":
            s["text"] = table_to_markdown(s["rows"])
            s["rows_text"] = table_rows_as_text(s["rows"])
    return merged


# ===========================================================================
# Step 6 (Elements 4, 6, 10): images, charts, text boxes, links
# ===========================================================================

def find_boxes(page, lines, table_boxes):
    """Find drawn boxes and images, and decide what each one is.

    Element 4:  images and charts -> the text inside (axis labels) is dropped
    Element 10: a framed box full of sentences is a TEXT BOX / sidebar ->
                its text is kept, but OUTSIDE the main paragraph flow.
    Returns (chart_boxes, textbox_boxes)."""
    page_area = page.rect.width * page.rect.height
    images = [tuple(i["bbox"]) for i in page.get_image_info()]
    try:
        drawings = [tuple(r) for r in page.cluster_drawings()]
    except AttributeError:                             # older PyMuPDF
        drawings = []

    def big_enough(b):
        w, h = b[2] - b[0], b[3] - b[1]
        return w > 50 and h > 20 and w * h < 0.8 * page_area

    def prose_inside(b):
        inner = [ln for ln in lines if inside(ln["bbox"], b)]
        words = sum(len(ln["text"].split()) for ln in inner)
        avg_len = sum(len(ln["text"]) for ln in inner) / max(len(inner), 1)
        return words >= TEXTBOX_MIN_WORDS and avg_len >= 20

    # An image is a CHART only when the text on it is chart labels. A background picture
    # (big, or with real sentences printed on top, e.g. the BEAWARE booklet pages) is not:
    # its text is normal content and must be kept.
    charts = [b for b in images if big_enough(b)
              and (b[2] - b[0]) * (b[3] - b[1]) < BACKGROUND_IMAGE_SHARE * page_area
              and not prose_inside(b)]
    textboxes = []
    for b in drawings:
        if not big_enough(b) or any(overlap_share(b, t) > 0.5 for t in table_boxes):
            continue
        inner = [ln for ln in lines if inside(ln["bbox"], b)]
        words = sum(len(ln["text"].split()) for ln in inner)
        avg_len = sum(len(ln["text"]) for ln in inner) / max(len(inner), 1)
        if words >= TEXTBOX_MIN_WORDS and avg_len >= 20:
            textboxes.append(b)
        elif (b[3] - b[1]) > 50:
            charts.append(b)
    return charts, textboxes


def page_links(page):
    """Element 6: clickable links with their visible text."""
    by_url = {}
    words = page.get_text("words")
    for link in page.get_links():
        uri = link.get("uri")
        if uri:
            r = link["from"]
            anchor = [w[4] for w in words
                      if r.x0 <= (w[0] + w[2]) / 2 <= r.x1 and r.y0 <= (w[1] + w[3]) / 2 <= r.y1]
            by_url.setdefault(uri, []).append(normalise_text(" ".join(anchor)))
    return [{"url": u, "text": " ".join(t for t in texts if t), "page": page.number + 1}
            for u, texts in by_url.items()]


# ===========================================================================
# Step 7 (Element 11): footnotes
# ===========================================================================

def split_footnotes(lines, page_height, body_size):
    """Footnotes = small-font lines at the bottom of the page, BELOW the last
    normal body line, where the first one starts with a marker (1, *, [^1]).
    Returns (remaining_lines, footnotes) with footnotes = [{"marker", "text"}]."""
    body_bottom = max((ln["bbox"][3] for ln in lines if abs(ln["size"] - body_size) < 0.6), default=0)
    candidates = [ln for ln in lines
                  if ln["size"] <= body_size * FOOTNOTE_SIZE_RATIO
                  and ln["bbox"][1] > page_height * FOOTNOTE_ZONE
                  and ln["bbox"][1] >= body_bottom - 1]
    candidates.sort(key=lambda ln: ln["bbox"][1])
    # footnotes must start with a marker line
    while candidates and not FOOTNOTE_START.match(candidates[0]["text"]):
        candidates.pop(0)
    if not candidates:
        return lines, []

    notes = []
    for ln in candidates:
        m = FOOTNOTE_START.match(ln["text"])
        if m:
            notes.append({"marker": m.group(1) or m.group(2), "text": ln["text"][m.end():]})
        else:                                          # wrapped footnote line
            notes[-1]["text"] += " " + ln["text"]
    used = {id(ln) for ln in candidates}
    return [ln for ln in lines if id(ln) not in used], notes


# ===========================================================================
# Step 8 (Elements 9 and 10): columns and reading order
# ===========================================================================

def text_area(pages, body_size):
    """Left/right edges of the main text, from normal body lines (whole doc)."""
    body = [ln for lines in pages for ln in lines if abs(ln["size"] - body_size) < 0.6]
    if not body:
        return 0, 10_000
    return percentile([ln["bbox"][0] for ln in body], 0.05), percentile([ln["bbox"][2] for ln in body], 0.95)


def find_gutters(lines, left, right):
    """Element 9: find empty vertical strips between columns ("gutters").

    We mark which x positions are covered by NARROW lines (lines shorter than
    ~half the text width). A strip in the middle that no narrow line touches,
    with enough lines on both sides and few lines crossing it, is a gutter."""
    width = right - left
    if width <= 0:
        return []
    narrow = [ln for ln in lines if (ln["bbox"][2] - ln["bbox"][0]) < 0.55 * width]
    if len(narrow) < 6:
        return []
    covered = [0] * (int(width) + 2)
    for ln in narrow:
        for x in range(max(0, int(ln["bbox"][0] - left)), min(len(covered), int(ln["bbox"][2] - left) + 1)):
            covered[x] += 1

    gutters, start = [], None
    for x, v in enumerate(covered + [1]):
        if v == 0 and start is None:
            start = x
        elif v != 0 and start is not None:
            if x - start >= MIN_GUTTER_WIDTH and start > 0.2 * width and x < 0.8 * width:
                gutters.append(left + (start + x) / 2)
            start = None

    good = []
    for g in gutters:
        left_n = sum(1 for ln in narrow if ln["bbox"][2] <= g)
        right_n = sum(1 for ln in narrow if ln["bbox"][0] >= g)
        crossing = sum(1 for ln in lines if ln["bbox"][0] < g - 2 and ln["bbox"][2] > g + 2)
        if left_n >= 3 and right_n >= 3 and crossing < 0.3 * (left_n + right_n):
            good.append(g)
    return good


def column_of(bbox, gutters) -> int:
    """Column number of a box, or -1 if it spans across a gutter (a title or
    wide table that covers both columns)."""
    for g in gutters:
        if bbox[0] < g - 2 and bbox[2] > g + 2:
            return -1
    return sum(1 for g in gutters if bbox[0] >= g)


def reading_order(items, gutters):
    """Element 9: order items the way a person reads them.

    Wide items (col -1) split the page into horizontal bands. Inside each
    band we read column 0 top-to-bottom, then column 1, and so on.
        [ wide title           ]
        [ col 0 ]   [ col 1 ]       <- band 1: left column first, then right
        [ wide table           ]
        [ col 0 ]   [ col 1 ]       <- band 2
    """
    for it in items:
        it["col"] = column_of(it["bbox"], gutters)
    items = sorted(items, key=lambda it: (round(it["bbox"][1]), it["bbox"][0]))
    ordered, band = [], []
    for it in items:
        if it["col"] == -1:
            ordered += sorted(band, key=lambda b: (b["col"], round(b["bbox"][1]), b["bbox"][0]))
            band = []
            ordered.append(it)
        else:
            band.append(it)
    ordered += sorted(band, key=lambda b: (b["col"], round(b["bbox"][1]), b["bbox"][0]))
    return ordered


# ===========================================================================
# Step 9 (Elements 3, 8, 14): lines -> paragraphs and (nested) lists
# ===========================================================================

def typical_line_gap(pages, body_size):
    """Normal vertical gap between two lines of the same paragraph. A gap
    clearly bigger than this means a new paragraph starts."""
    gaps = []
    for lines in pages:
        for a, b in zip(lines, lines[1:]):
            g = b["bbox"][1] - a["bbox"][3]
            # only lines from the same PDF text block (= same paragraph),
            # so table rows and list items don't distort the number
            # (gap can be slightly negative: line boxes overlap when spacing is tight)
            if a["block"] == b["block"] and abs(a["size"] - body_size) < 0.6 and -body_size * 0.5 <= g < body_size * 2:
                gaps.append(g)
    return statistics.median(gaps) if gaps else 2


def lines_to_blocks(lines, line_gap):
    """Join PDF lines into paragraphs and list items.

    New block when:
      - a bullet or number starts a new list item,
      - the vertical gap is clearly bigger than normal line spacing,
      - a caption line starts or ends.
    Element 8:  a page/column break mid-sentence does NOT start a new paragraph.
    Element 14: list level comes from indentation. Each deeper indent adds
                two spaces, so Markdown shows "- item" / "  - sub item"."""
    blocks, current, prev = [], "", None
    indent_stack = []                               # x positions of open list levels

    for ln in lines:
        text, x0 = ln["text"], ln["bbox"][0]
        is_bullet = bool(BULLET_LINE.match(text))
        is_number = bool(NUMBERED_LINE.match(text))
        starts_item = is_bullet or is_number
        jumped = prev is not None and (ln["page"] != prev["page"] or ln.get("col") != prev.get("col")
                                       or ln["bbox"][1] < prev["bbox"][1] - 1)
        big_gap = prev is not None and not jumped and \
            (ln["bbox"][1] - prev["bbox"][3]) > line_gap + prev["size"] * 0.4
        if jumped and current and current.rstrip().endswith((".", ":", "?", "!")):
            big_gap = True                          # ended a sentence -> new paragraph
        # Element 14: a normal line that starts LEFT of the list's bullets is
        # not part of the last item; the list has ended
        if indent_stack and not starts_item and not jumped and x0 < indent_stack[0] - 3:
            big_gap = True
        is_caption = ln.get("caption") or (prev is not None and prev.get("caption"))

        if starts_item or big_gap or is_caption or not current:
            if current:
                blocks.append(current)
            if starts_item:
                # Element 14: work out the nesting level from the indent
                while indent_stack and x0 < indent_stack[-1] - 3:
                    indent_stack.pop()
                if not indent_stack or x0 > indent_stack[-1] + 3:
                    indent_stack.append(x0)
                level = len(indent_stack) - 1
                if is_bullet:
                    text = BULLET_LINE.sub("- ", text)                 # "• x" -> "- x"
                elif not re.match(r"^\s*\d{1,3}[.)]\s", text):
                    text = "- " + text.strip()                         # "(a) x" -> "- (a) x"
                text = "  " * level + text.strip()
            else:
                indent_stack = []                   # normal paragraph ends the list
            current = text
        else:
            if current.endswith("-") and not current.endswith(" -"):
                current = current[:-1] + text       # "regu-" + "lation"
            else:
                current += " " + text
        prev = ln
    if current:
        blocks.append(current)
    return blocks


def join_blocks(blocks) -> str:
    """List items on consecutive lines, everything else separated by a blank line."""
    out = ""
    for b in blocks:
        prev_item = bool(out) and bool(LIST_ITEM_MD.match(out.rsplit("\n", 1)[-1]))
        sep = "" if not out else "\n" if (LIST_ITEM_MD.match(b) and prev_item) else "\n\n"
        out += sep + b
    return out


# ===========================================================================
# Step 10 (Elements 15, 16): section numbers, cross-references, annexures
# ===========================================================================

def add_cross_references(sections):
    """Element 15: find "para 3.2" / "Annex I" in the text and link each one
    to the section it points to, so a chunk that says "as per para 3.2" can
    later pull in para 3.2 as extra context."""
    number_index, annex_index = {}, {}
    for s in sections:
        if s["type"] != "heading":
            continue
        last = s["heading_path"].split(" > ")[-1]
        m = ANNEX_HEADING.match(last)
        if m and m.group(2):
            annex_index.setdefault(m.group(2).upper(), s["heading_path"])
        elif s.get("section_no"):
            number_index.setdefault(s["section_no"], s["heading_path"])

    for s in sections:
        if s["type"] not in ("text", "note"):
            continue
        refs, seen = [], set()
        for m in CROSS_REF.finditer(s["text"]):
            kind, target = m.group(1).lower(), m.group(2)
            if (kind, target) in seen:
                continue
            seen.add((kind, target))
            if kind.startswith(("annex", "appendix", "schedule")):
                resolved = annex_index.get(target.upper())
            else:
                base = re.sub(r"\(.*", "", target)       # "4(a)" -> "4"
                resolved = number_index.get(target) or number_index.get(base)
            refs.append({"text": m.group(0), "kind": kind, "target": target,
                         "resolved_heading": resolved})
        if refs:
            s["cross_refs"] = refs
    return sections


# ===========================================================================
# Step 11: parse one PDF
# ===========================================================================

def parse_pdf(pdf_path: Path, use_bookmarks=True):
    doc = pymupdf.open(pdf_path)
    pages = read_lines(doc)
    repeated = find_headers_footers(doc, pages)
    body_size, levels = font_statistics(pages)
    line_gap = typical_line_gap(pages, body_size)
    area_left, area_right = text_area(pages, body_size)
    tables = extract_tables(pdf_path)

    bookmarks = BookmarkHeadings(doc)
    use_bookmarks = use_bookmarks and bookmarks.available()

    title = normalise_text((doc.metadata or {}).get("title") or "")
    if (title.lower() in ("untitled", "anonymous", "(anonymous)", "microsoft word", "none")
            or title.lower().endswith((".doc", ".docx", ".pdf"))):
        title = ""                          # placeholder titles from PDF tools
    links, removed = [], Counter()
    sections, heading_path, buffer = [], [], []
    pending_notes = []                  # Element 10: sidebars waiting to be written
    page_footnotes = defaultdict(list)  # Element 11
    in_references, references = False, []
    part = "main"                       # Element 16
    multi_column_pages = 0

    def new_section(**kw):
        kw.setdefault("heading_path", " > ".join(heading_path) or title or pdf_path.stem)
        kw["part"] = part
        sections.append(kw)

    def flush():
        nonlocal buffer
        if buffer:
            blocks = lines_to_blocks(buffer, line_gap)
            if in_references:
                references.extend(blocks)
            else:
                new_section(type="text", text=join_blocks(blocks), page=buffer[0]["page"],
                            page_end=max(ln["page"] for ln in buffer))   # last page it reaches
        buffer = []

    def flush_notes():
        """Element 10: write sidebars/text boxes after the section they sit in."""
        for note in pending_notes:
            new_section(type="note", text="> **Note:** " + note["text"], page=note["page"],
                        note_kind=note["kind"])
        pending_notes.clear()

    for page, lines in zip(doc, pages):
        page_no, height = page.number + 1, page.rect.height
        links.extend(page_links(page))

        if is_toc_page(lines):                                     # Element 5
            removed["toc_or_index_pages"] += 1
            continue

        # Element 7: drop headers and footers
        kept = []
        for ln in lines:
            if is_header_footer(ln, height, repeated):
                removed["header_footer_lines"] += 1
            else:
                kept.append(ln)

        # Elements 2, 4, 10: tables, charts, text boxes
        page_tables = tables.get(page_no, [])
        table_boxes = [t["bbox"] for t in page_tables]
        charts, textboxes = find_boxes(page, kept, table_boxes)

        items, box_lines, margin = [], defaultdict(list), []
        for t in page_tables:
            items.append({"kind": "table", "bbox": t["bbox"], "rows": t["rows"]})
        flow = []
        for ln in kept:
            if any(inside(ln["bbox"], b) for b in table_boxes):
                continue                                           # inside a table
            box = next((b for b in textboxes if inside(ln["bbox"], b)), None)
            if box:
                box_lines[box].append(ln)                          # inside a text box
                continue
            if any(inside(ln["bbox"], b) for b in charts) and not CAPTION_LINE.match(ln["text"]):
                removed["text_inside_images"] += 1                 # chart labels
                continue
            # Element 10: margin notes sit outside the main text area
            if ln["bbox"][2] < area_left - 2 or ln["bbox"][0] > area_right + 2:
                margin.append(ln)
                continue
            flow.append(ln)

        # Notes are kept as single items in the reading order, so they end up
        # right after the section they sit next to (not mixed into sentences).
        for box, blines in box_lines.items():
            blines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
            items.append({"kind": "note", "note_kind": "text_box", "bbox": box,
                          "text": " ".join(lines_to_blocks(blines, line_gap))})
            removed["text_boxes_moved"] += 1
        # Consecutive margin lines ("Amended" / "Jan 2025") form one note
        margin.sort(key=lambda ln: (ln["bbox"][0] > area_right, ln["bbox"][1]))
        for ln in margin:
            last = items[-1] if items and items[-1].get("note_kind") == "margin_note" else None
            if last and -3 <= ln["bbox"][1] - last["bbox"][3] < ln["size"] * 1.5 \
                    and abs(ln["bbox"][0] - last["bbox"][0]) < 20:
                last["text"] += " " + ln["text"]
                last["bbox"] = (min(last["bbox"][0], ln["bbox"][0]), last["bbox"][1],
                                max(last["bbox"][2], ln["bbox"][2]), ln["bbox"][3])
            else:
                items.append({"kind": "note", "note_kind": "margin_note", "bbox": ln["bbox"],
                              "text": ln["text"]})
                removed["margin_notes_moved"] += 1

        # Element 11: footnotes at the bottom of the page
        flow, notes = split_footnotes(flow, height, body_size)
        page_footnotes[page_no].extend(notes)

        # Element 9: columns + reading order
        gutters = find_gutters(flow, area_left, area_right)
        if gutters:
            multi_column_pages += 1
        for ln in flow:
            ln["kind"] = "line"
        items = reading_order(items + flow, gutters)
        for it in items:
            if it["kind"] == "line":
                it["col"] = it.get("col", 0)

        i = 0
        while i < len(items):
            it = items[i]

            if it["kind"] == "note":                               # Element 10
                pending_notes.append({"text": it["text"], "page": page_no, "kind": it["note_kind"]})
                i += 1
                continue

            if it["kind"] == "table":                              # Elements 2, 12
                flush()
                new_section(type="table", rows=it["rows"], page=page_no, pages=[page_no],
                            text=table_to_markdown(it["rows"]))
                i += 1
                continue

            ln = it
            if CAPTION_LINE.match(ln["text"]) and len(ln["text"]) < 200:   # Element 4
                buffer.append({**ln, "text": f"[{ln['text']}]", "caption": True})
                i += 1
                continue

            # Element 17 (bookmarks) or Element 1 (font size): is it a heading?
            heading = None
            if use_bookmarks:
                hit = bookmarks.match(items, i, page_no)
                if hit:
                    heading = hit                                  # (level, text, n_lines)
                elif not title and page_no == 1 and levels.get(ln["size"]) == 1:
                    heading = (1, ln["text"], 1)                   # document title
            else:
                lvl = font_heading_level(ln, levels)
                if lvl:
                    # Element 1: a long heading wraps onto 2+ lines. Join the
                    # next lines if they have the same font and sit right below.
                    text, used, last = ln["text"], 1, ln
                    while i + used < len(items):
                        nxt = items[i + used]
                        if (nxt["kind"] != "line" or nxt["size"] != ln["size"] or nxt["bold"] != ln["bold"]
                                or not 0 - ln["size"] <= nxt["bbox"][1] - last["bbox"][3] < ln["size"] * 0.6
                                or font_heading_level(nxt, levels) != lvl
                                or HEADING_NUMBER.match(nxt["text"])):   # "1. Intro" = new heading
                            break
                        text, used, last = text + " " + nxt["text"], used + 1, nxt
                    if len(text) <= MAX_HEADING_CHARS and not sentence_like(text):
                        heading = (lvl, text, used)
                    else:                                          # a paragraph in a big font: text
                        buffer.extend(items[i:i + used])
                        i += used
                        continue

            if heading:
                level, text, used = heading
                flush()
                flush_notes()
                in_references = bool(REFERENCE_WORDS.match(text))
                if not in_references:
                    annex = ANNEX_HEADING.match(text)              # Element 16
                    if annex:
                        part = "annex: " + text
                        level = 1                                  # annex starts a new top level
                    if not title and level == 1:
                        title = text
                    heading_path = heading_path[:level - 1] + [text]
                    num = HEADING_NUMBER.match(text)               # Element 15
                    new_section(type="heading", text="#" * min(level, 6) + " " + text,
                                page=page_no, section_no=num.group(1) if num else None)
                i += used
                continue

            buffer.append(ln)                                      # Elements 3, 8, 14
            i += 1

    flush()
    flush_notes()

    # If too few bookmarks matched the text, the outline is unreliable:
    # parse again using font sizes instead
    if use_bookmarks and bookmarks.matched < TOC_MATCH_SHARE * bookmarks.total:
        return parse_pdf(pdf_path, use_bookmarks=False)

    sections = [s for s in sections if s["text"].strip()]
    sections = merge_tables_across_pages(sections)                 # Element 12

    # Element 11: attach each footnote to the section that cites "[^n]"
    for page_no, notes in page_footnotes.items():
        for note in notes:
            marker = f"[^{note['marker']}]"
            target = next((s for s in sections if s["type"] in ("text", "note") and marker in s["text"]
                           and page_no - 1 <= s["page"] <= page_no), None)
            if target is None:                      # no marker found -> last text on that page
                target = next((s for s in reversed(sections)
                               if s["type"] == "text" and s["page"] <= page_no), None)
            if target is not None:
                target.setdefault("footnotes", []).append({**note, "page": page_no})
                removed["footnotes_attached"] += 1

    sections = add_cross_references(sections)                      # Element 15

    known = {l["url"] for l in links}                              # Element 6
    for s in sections:
        for url in URL_PATTERN.findall(s["text"]):
            if url not in known:
                links.append({"url": url, "text": "", "page": s["page"]})
                known.add(url)

    return {
        "doc_id": re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_"),
        "file_name": pdf_path.name,
        "title": title or pdf_path.stem,
        "num_pages": len(doc),
        "headings_from": "bookmarks" if use_bookmarks else "font_size",
        "multi_column_pages": multi_column_pages,
        "sections": sections,
        "links": links,
        "references": references,
        "cleaning_stats": dict(removed),
    }


# ===========================================================================
# Step 12: outputs
# ===========================================================================

def to_markdown(parsed) -> str:
    """Readable Markdown per document, to check the cleaning by eye."""
    out = [f"<!-- source: {parsed['file_name']} | pages: {parsed['num_pages']} "
           f"| headings from: {parsed['headings_from']} -->"]
    current_part = "main"
    for s in parsed["sections"]:
        if s["part"] != current_part:
            out.append("---")                       # visual break before annexures
            current_part = s["part"]
        text = s["text"]
        if s.get("footnotes"):
            text += "\n\n" + "\n".join(f"[^{f['marker']}]: {f['text']}" for f in s["footnotes"])
        out.append(text)
    if parsed["references"]:
        out.append("## References\n\n" + "\n".join(f"- {r}" for r in parsed["references"]))
    if parsed["links"]:
        out.append("## Links\n\n" + "\n".join(
            f"- [{l['text'] or l['url']}]({l['url']}) (p.{l['page']})" for l in parsed["links"]))
    return "\n\n".join(out) + "\n"


def write_qa_template(path: Path):
    """Template for the 30 hand-written test questions
    (~12 paragraph, ~8 table, ~5 list, ~5 multi-section or no-answer)."""
    if path.exists():
        return                                      # never overwrite your questions
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "expected_answer", "source_doc_id",
                    "page", "answer_type", "notes"])
        w.writerow(["q001", "What is the example question?", "Example answer copied from the doc",
                    "example_doc", 3, "paragraph | table | list | multi_section | no_answer",
                    "Delete this example row"])


# ===========================================================================
# Element 32: duplicates
# Check for duplicates by hashing the cleaned text (ignoring whitespace,
# Unicode formatting, and letter case).
# ===========================================================================

def cleaned_text_hash(parsed):
    """
    Generate a unique hash from a PDF's cleaned content.

    Ignores differences in whitespace, Unicode formatting,
    and letter case when comparing documents.
    """

    parts = []

    # Collect cleaned section text
    for section in parsed.get("sections", []):
        text = section.get("text", "")

        if text:
            parts.append(str(text))

        # Include attached footnotes
        for note in section.get("footnotes", []):
            marker = note.get("marker", "")
            note_text = note.get("text", "")
            parts.append(f"Footnote {marker}: {note_text}")

    # Include references
    parts.extend(
        str(reference)
        for reference in parsed.get("references", [])
        if reference
    )

    # Include hyperlinks without their page numbers
    for link in parsed.get("links", []):
        url = link.get("url", "")
        link_text = link.get("text", "")

        if url or link_text:
            parts.append(f"{link_text} {url}")

    # Combine the extracted content
    text = "\n".join(parts)

    # Normalize formatting differences
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()

    # Don't treat empty documents as duplicates
    if not text:
        return None

    # Generate SHA-256 hash
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ===========================================================================
# Skipped-file log (Elements 29, 30, 32)
# ===========================================================================

def log_skip(log_path: Path, pdf_path: Path, reason: str, detail: str = ""):
    """Append one row to output/skipped_files.csv so every skipped file
    can be checked later (password, corrupt, too big, timeout, duplicate)."""
    new_file = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "file_name", "reason", "detail"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), pdf_path.name, reason, detail])


# ===========================================================================
# Element 29: password-protected PDFs
# ===========================================================================

def check_password_pdf(pdf_path: Path) -> bool:
    """Return True ONLY if the PDF is password-protected.

    A file that cannot be opened at all is NOT counted here (returns False);
    validate_pdf_file() catches it next and counts it as failed/corrupt."""
    doc = None

    try:
        doc = pymupdf.open(str(pdf_path))

        if doc.needs_pass or doc.is_encrypted:
            print(f"PASSWORD-PROTECTED: {pdf_path.name}")
            return True

        return False

    except Exception:
        return False                    # corrupt -> handled by validate_pdf_file

    finally:
        if doc is not None:
            doc.close()


# ===========================================================================
# Element 30: corrupt / huge files (size + page limit + readability)
# ===========================================================================

def validate_pdf_file(
    pdf_path: Path,
    max_size_mb: int = MAX_SIZE_MB,
    max_pages: int = MAX_PAGES
):
    """Validate PDF size, page count, and readability.
    Returns (True, "") if OK, else (False, "reason: detail")."""

    doc = None

    try:
        if not pdf_path.exists() or not pdf_path.is_file():
            print(f"FILE NOT FOUND: {pdf_path}")
            return False, "not_found"

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

        if file_size_mb == 0:
            print(f"EMPTY FILE: {pdf_path.name}")
            return False, "empty_file"

        if file_size_mb > max_size_mb:
            print(
                f"FILE TOO LARGE: {pdf_path.name} "
                f"({file_size_mb:.2f} MB > {max_size_mb} MB)"
            )
            return False, f"too_large: {file_size_mb:.2f} MB"

        doc = pymupdf.open(str(pdf_path))


        if not doc.is_pdf:
            print(f"INVALID PDF: {pdf_path.name}")
            return False, "not_a_pdf"

        if doc.page_count == 0:
            print(f"EMPTY PDF: {pdf_path.name}")
            return False, "zero_pages"

        if doc.page_count > max_pages:
            print(
                f"PAGE LIMIT EXCEEDED: {pdf_path.name} "
                f"({doc.page_count} pages > {max_pages})"
            )
            return False, f"too_many_pages: {doc.page_count}"

        # Try to actually read the first page: catches files that open
        # but have broken page content
        doc[0].get_text("text")

        return True, ""

    except Exception as e:
        print(f"CORRUPT OR UNREADABLE PDF: {pdf_path.name}: {e}")
        return False, f"corrupt: {type(e).__name__}: {e}"

    finally:
        if doc is not None:
            doc.close()


# ===========================================================================
# Element 30: timeout. Each PDF is parsed in a separate process; if it takes
# longer than the limit (or the process crashes), it is killed and skipped,
# so one bad file can never hang the whole run.
# ===========================================================================

def _parse_worker(pdf_path_str, result_queue):
    """Runs inside the child process:
    fix rotation / OCR (18, 28) -> parse_pdf (1-17) -> extra cleaning (18-27)."""
    work_dir = Path(tempfile.mkdtemp(prefix="pdfclean_"))
    try:
        path, prep = prepare_pdf(Path(pdf_path_str), work_dir)        # 18, 28
        parsed = parse_pdf(path)                                      # 1-17
        parsed = extra_cleaning(parsed, path, prep)                   # 18-27
        result_queue.put(("ok", parsed))
    except Exception as e:
        result_queue.put(("error", f"{type(e).__name__}: {e}"))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)   # delete the temporary copy


def parse_pdf_with_timeout(pdf_path: Path, timeout_sec: int = PARSE_TIMEOUT_SEC):
    """Call parse_pdf() with a time limit. Raises TimeoutError if too slow,
    RuntimeError if parsing failed or the worker process crashed."""
    ctx = mp.get_context("spawn")          # same behaviour on Windows / Linux / macOS
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_parse_worker, args=(str(pdf_path), result_queue), daemon=True)
    proc.start()

    deadline = time.monotonic() + timeout_sec
    result = None
    try:
        while time.monotonic() < deadline:
            try:
                result = result_queue.get(timeout=1)   # read BEFORE join (avoids deadlock)
                break
            except queue.Empty:
                if not proc.is_alive():                # crashed without a result
                    try:
                        result = result_queue.get(timeout=1)
                        break
                    except queue.Empty:
                        raise RuntimeError(f"parser process crashed (exit code {proc.exitcode})")
        if result is None:
            raise TimeoutError(f"timed out after {timeout_sec}s")
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        result_queue.close()

    status, payload = result
    if status == "error":
        raise RuntimeError(payload)
    return payload


# ===========================================================================
# Element 31: document version & date
# ===========================================================================

def to_iso_date(raw):
    """'April 02, 2024' / '2nd April 2024' / '02/04/2024' -> '2024-04-02'.
    Numeric dates are read as day/month/year (Indian format).
    Returns None if the text is not a real date."""
    if not raw:
        return None
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw.strip(), flags=re.I)
    text = re.sub(r"\s*,\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?i)\bsept\b", "Sep", text)          # "Sept" -> "Sep"
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


REFERENCE_CONTEXT = re.compile(
    r"(circular|notification|letter|direction|vide|supersed|replac|amend|dated|referred|earlier)",
    re.I)


def is_reference_date(head: str, start: int) -> bool:
    """True if the date belongs to ANOTHER document, e.g.
    '... supersedes circular RBI/2019-20/45 dated July 1, 2019'.
    Looks at the text just before the date, within the same sentence."""
    before = head[max(0, start - 80):start]
    before = re.split(r"[.\n]\s", before)[-1]          # same sentence only
    return bool(REFERENCE_CONTEXT.search(before))


def find_issue_date(head: str):
    """Issue date of THIS document, most reliable first:
       1. a labelled date at the start of a line: 'Date: 02/04/2024'
       2. the first date on page 1 that is NOT a reference to another
          circular ('... circular X dated July 1, 2019' is skipped)
    No safe date -> None. A wrong date is worse than an empty one.
    Footnote lines are ignored: their dates belong to other documents."""
    head = FOOTNOTE_TEXT_LINE.sub("", head)                      # Fix A3: skip footnotes
    for m in LABELLED_DATE.finditer(head):
        line_start = head.rfind("\n", 0, m.start()) + 1
        if head[line_start:m.start()].strip(" #*>-|") == "" and to_iso_date(m.group(1)):
            return m.group(1).strip()
    for m in ANY_DATE.finditer(head):
        if to_iso_date(m.group(1)) and not is_reference_date(head, m.start()):
            return m.group(1).strip()
    return None                                                  # Fix A3: no guessing


def find_circular_no(head: str):
    """Circular number of THIS document (from the first page only, so that
    an older circular mentioned in the body is not picked up):
       1. RBI reference number: 'RBI/2024-25/12'
       2. labelled: 'Circular No. DBR.BP.BC.45/21.04.048/2024-25'"""
    m = RBI_REF_NO.search(head)
    if m:
        return m.group(0)
    m = LABELLED_CIRCULAR_NO.search(head)
    if m:
        return m.group(1).rstrip(".,;:)")
    return None


ANNUAL_COVER = re.compile(r"\b(?:integrated\s+)?annual\s+report\b", re.I)
RESULTS_LETTER = re.compile(r"\b(?:audited|unaudited)\s+(?:standalone\s+|consolidated\s+)?financial\s+results\b"
                            r"|\boutcome\s+of\s+(?:the\s+)?board\s+meeting\b", re.I)


def is_annual_report_pdf(pdf_path) -> bool:
    """A company's annual report or results? (2.3)
    YES : "ANNUAL REPORT 2025-26" on the COVER (top of page 1) or in the PDF's own title,
          or "audited financial results" in the first 3 pages (results sent to the stock exchange).
    NO  : "ANNUAL REPORT 2024-25" only as a running page footer. That is a CHAPTER of RBI's own
          Annual Report ("Regulation, Supervision and Financial Stability"): it is RBI content
          and stays in the database (review: 'Annual report.pdf' was wrongly left out)."""
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            first = doc[0].get_text() if len(doc) else ""
            raw = " ".join(doc[i].get_text() for i in range(min(3, len(doc))))
            title = (doc.metadata or {}).get("title") or ""
    except Exception:
        return False
    cover = first.strip()[:600]                    # the cover = the first lines of page 1
    return bool(ANNUAL_COVER.search(cover) or ANNUAL_COVER.search(title) or RESULTS_LETTER.search(raw))


NUMBERED_QUESTION = re.compile(
    r"^\s*(?:Q\.?\s*)?\d{1,3}[.)]\s+(?:What|How|Can|Is|Are|Who|Whom|Where|When|Why|Will|Do|Does|"
    r"Should|Which|Whether|Has|Have|May)\b[^\n]*", re.M | re.I)


def detect_doc_type(head: str) -> str:
    """Fix A2: first type in DOC_TYPES whose pattern is found on page 1.
    2.3: a page of numbered questions ("1. What is Digital Rupee?") is an FAQ even when the
    word "FAQ" is not printed (FAQ1, FAQ2, FAQ3 were typed as circulars)."""
    if not RBI_REF_NO.search(head) and len(NUMBERED_QUESTION.findall(head)) >= 2:
        return "faq"
    for doc_type, rule in DOC_TYPES.items():
        if doc_type == "faq" and RBI_REF_NO.search(head):
            continue                      # a circular that only MENTIONS FAQs stays a circular
        if rule["detect"].search(head):
            return doc_type
    return "general"


def extract_manual_metadata(head: str) -> dict:
    """Fix A2: model, product, brand and version of a user manual (page 1 only)."""
    model = MODEL_NO.search(head)
    if not model:                                   # "AP-X200 | User Manual" without "Model"
        product_line = PRODUCT_NAME.search(head)
        model = IDENTIFIER.search(product_line.group(1)) if product_line else None
        model_no = model.group(0) if model else None
    else:
        model_no = model.group(1)
    version = DOC_VERSION.search(head)
    product = PRODUCT_NAME.search(head)
    brand = BRAND.search(head)
    product_name = product.group(1).strip(" |-:#") if product else None
    if product_name and model_no:                   # "Model AP-X200" -> no product name
        product_name = re.sub(r"(?i)(\bmodel\s*(no\.?|number)?\s*[:\-]?\s*)?" + re.escape(model_no),
                              "", product_name).strip(" |-:") or None
    return {
        "model": model_no,
        "product": product_name,
        "brand": brand.group(1).strip() if brand else None,
        "doc_version": version.group(1) if version else None,
    }


def raw_cover_text(pdf_path, pages=3) -> str:
    """Plain text of the first pages + PDF metadata title, before any cleaning."""
    try:
        doc = pymupdf.open(pdf_path)
        try:
            text = " ".join(doc[i].get_text() for i in range(min(pages, len(doc))))
            return (doc.metadata or {}).get("title", "") + " " + text
        finally:
            doc.close()
    except Exception:
        return ""


def extract_version_metadata(text: str) -> dict:
    """Extract circular version metadata from the cleaned document text.

    issue_date     ISO date (YYYY-MM-DD), sortable: newest version wins
    issue_date_raw the date exactly as printed in the PDF
    circular_no    this document's own circular / reference number
    supersedes     the OLDER circular(s) that THIS document replaces
                   ("This circular supersedes ...", "in supersession of ...")
    superseded_by  the NEWER circular that replaces this one. A document never
                   names its own replacement, so this starts as None and is
                   filled by link_superseded() after all PDFs are processed.
    is_latest      False once a newer circular supersedes this one
    """
    head = text[:METADATA_HEAD_CHARS]
    doc_type = detect_doc_type(head)                              # Fix A2

    issue_date_raw = find_issue_date(head)
    circular_no = find_circular_no(head) if doc_type == "rbi_circular" else None
    manual = (extract_manual_metadata(head) if doc_type == "manual"
              else {"model": None, "product": None, "brand": None, "doc_version": None})

    supersedes = find_supersedes(text)     # whole text: often in the last paragraph

    # review M3: a consolidated master direction keeps its first issue date ("July 5, 2016") but
    # is "(Updated as on October 01, 2026)". The LATEST update date is stored too.
    updates = [to_iso_date(m.group(1)) for m in UPDATED_AS_ON.finditer(head)]
    updated_on = max((u for u in updates if u), default=None)

    return {
        "doc_type": doc_type,
        **manual,                          # model, product, brand, doc_version
        "issue_date": to_iso_date(issue_date_raw),
        "issue_date_raw": issue_date_raw,
        "updated_on": updated_on,
        "circular_no": circular_no,
        "supersedes": supersedes,
        "superseded_by": None,
        "is_latest": True,
    }


def find_supersedes(text: str):
    """References after 'supersedes / replaces / in supersession of' -- ONLY:
       - RBI numbers:        RBI/2019-20/45
       - department numbers: DBR.BP.BC.No.45/21.04.048/2024-25
       - dated references:   'circular ... dated July 1, 2019' -> 'dated 2019-07-01'
    A fragment like 'earlier orders' gives nothing. Returns '; '-joined refs or None."""
    refs = []
    for m in SUPERSEDES.finditer(text):
        part = m.group(1)
        found = RBI_REF_NO.findall(part) + DEPT_REF_NO.findall(part)
        if not found:
            for d in DATED_REF.finditer(m.group(0)):
                iso = to_iso_date(d.group(1))
                if iso:
                    found.append(f"dated {iso}")
        refs += [r for r in found if r not in refs]
    return "; ".join(refs) or None


def norm_ref(text: str) -> str:
    """Compare circular numbers ignoring case, spaces and punctuation."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def link_superseded(json_dir: Path) -> int:
    """Second pass over ALL json files:
    if document B says it supersedes circular X, and document A has
    circular_no X, then A.superseded_by = B's circular_no and A.is_latest = False.
    Re-computed from scratch on every run, so it stays correct when new
    circulars are added later. Returns the number of links found."""
    docs = {}
    for path in json_dir.glob("*.json"):
        try:
            docs[path] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not read {path.name}: {e}")

    # remember the links as they are now, to save only files that change
    before = {path: (d.get("metadata", {}).get("superseded_by"),
                     d.get("metadata", {}).get("is_latest"))
              for path, d in docs.items()}

    # reset, so old links from previous runs don't stay behind
    for d in docs.values():
        meta = d.setdefault("metadata", {})
        meta["superseded_by"] = None
        meta["is_latest"] = not meta.get("stamp")      # Element 23: stamped = not latest

    links = 0
    for new_path, new_doc in docs.items():
        new_meta = new_doc["metadata"]
        # review H3: every reference on its own, compared EXACTLY. (Before: a substring test on
        # all of them joined, so "RBI/2024-25/1" = "rbi2024251" was found inside "rbi20242512".)
        refs = [r.strip() for r in (new_meta.get("supersedes") or "").split(";")]
        targets = {norm_ref(r) for r in refs if r and not r.startswith("dated ")}
        if not refs or not any(refs):
            continue
        dates = set(re.findall(r"dated (\d{4}-\d{2}-\d{2})", new_meta.get("supersedes") or ""))
        for old_path, old_doc in docs.items():
            if old_path == new_path:
                continue
            no = norm_ref(old_doc["metadata"].get("circular_no"))
            by_number = len(no) >= 4 and no in targets         # review H3: EXACT match
            by_date = (old_doc["metadata"].get("issue_date") in dates
                       and old_doc["metadata"].get("doc_type") == new_meta.get("doc_type"))
            if not (by_number or by_date):
                continue
            old_meta = docs[old_path]["metadata"]
            # safety: the "old" one must not be newer than the one replacing it
            if (old_meta.get("issue_date") and new_meta.get("issue_date")
                    and old_meta["issue_date"] > new_meta["issue_date"]):
                continue
            old_meta["superseded_by"] = new_meta.get("circular_no") or new_doc.get("doc_id")
            old_meta["is_latest"] = False
            links += 1
            print(f"SUPERSEDED: {docs[old_path].get('file_name')} "
                  f"-> replaced by {new_doc.get('file_name')}")

    # save only the files whose superseded_by / is_latest really changed
    for path, d in docs.items():
        if (d["metadata"]["superseded_by"], d["metadata"]["is_latest"]) == before[path]:
            continue                       # nothing changed -> no write
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        if d["metadata"].get("superseded_by"):
            print(f"SUPERSEDED_BY ADDED SUCCESSFULLY -> {path.name} "
                  f"(superseded_by = {d['metadata']['superseded_by']})")
    return links


def print_final_metadata(json_dir: Path):
    """Print the FINAL version metadata of every document, after
    link_superseded() has filled superseded_by / is_latest."""
    rows = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        m = d.get("metadata", {})
        rows.append((d.get("file_name", path.name),
                     m.get("circular_no") or "-",
                     m.get("issue_date") or "-",
                     "yes" if m.get("is_latest", True) else "NO",
                     m.get("superseded_by") or "-"))
    if not rows:
        return

    print("\nFinal metadata:")
    header = ("File", "Circular No", "Issue date", "Latest", "Superseded by")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(5)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    print(line.format(*header))
    print(line.format(*("-" * w for w in widths)))
    for r in rows:
        print(line.format(*r))


# ===========================================================================

def file_sha256(path: Path) -> str:
    """Fingerprint of the PDF FILE itself (its bytes). Fast: the file is read
    in 1 MB pieces, so even a big file takes well under a second.
    Same fingerprint = same file, nothing changed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def load_previous_outputs(json_dir: Path):
    """Read every JSON saved by earlier runs.
    Returns {file_name: {"json": path, "md": path, "file_hash", "pipeline_version",
                         "content_hash"}}"""
    previous = {}
    for json_path in json_dir.glob("*.json"):
        try:
            d = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not read {json_path.name}: {e}")
            continue
        previous[d.get("file_name", json_path.name)] = {
            "json": json_path,
            "md": json_path.parent.parent / "markdown" / f"{json_path.stem}.md",
            "file_hash": d.get("file_hash"),
            "pipeline_version": d.get("pipeline_version"),
            "content_hash": d.get("content_hash") or cleaned_text_hash(d),
        }
    return previous


def remove_outputs(entry):
    """Delete the old JSON and Markdown of one PDF."""
    for key in ("json", "md"):
        try:
            entry[key].unlink()
        except FileNotFoundError:
            pass


# ===========================================================================
# Elements 18-28: page checks and extra text cleaning
# ===========================================================================

SALUTATION = re.compile(r"\b(?:madam\s*/\s*(?:dear\s+)?sir|(?:dear\s+)?sir\s*/\s*madam|dear\s+(?:sir|madam|sirs))\b[,:]?", re.I)
CLOSING = re.compile(r"\byours\s+(?:faithfully|sincerely|truly)\b", re.I)
SUBJECT_LINE = re.compile(r"^\s*(?:sub(?:ject)?\s*[:.\-]|re\s*:)", re.I)
BOILERPLATE = re.compile(r"\b(?:disclaimer|intentionally left blank|all rights reserved)\b", re.I)
STAMP_WORDS = re.compile(r"\b(?:superseded|withdrawn|repealed|cancelled)\b", re.I)
GARBAGE = re.compile(r"\(cid:\d+\)|\ufffd")           # broken font output
PRIVATE_USE = re.compile("[-]")              # font-specific symbols
DEVANAGARI = re.compile("[ऀ-ॿ]")               # Hindi letters
MATH = re.compile(r"[∑∫√∞±×÷≠≈∂∆∏πΣ]|\b[a-z]\s*=\s*[a-z0-9(]|\^\d")
SYMBOL_MAP = {"": "-", "": "-", "": "-", "": "-",   # Symbol /
              "": "-", "": "-", "": "-", "": " "}   # Wingdings bullets
RUPEE_BACKTICK = re.compile(r"`(?=\s?\d)")               # old Rupee font: "` 500" = "₹ 500"


def is_scanned_page(page) -> bool:
    """Element 18: almost no text, and an image covers most of the page."""
    words = len(page.get_text("text").split())
    area = abs(page.rect) or 1
    image_area = sum(abs(pymupdf.Rect(i["bbox"]) & page.rect) for i in page.get_image_info())
    return words < 20 and image_area > 0.5 * area


TURN_FOR_DIRECTION = {(0, -1): 90, (0, 1): 270, (-1, 0): 180}   # text dir -> fix


def sideways_turn(page) -> int:
    """Element 28: if MOST of the text on the page runs sideways or upside
    down, return the rotation (90 / 180 / 270) that makes it upright, else 0."""
    chars = Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            direction = (round(line["dir"][0]), round(line["dir"][1]))
            chars[direction] += sum(len(sp["text"].strip()) for sp in line["spans"])
    total = sum(chars.values())
    if total < 20:
        return 0
    direction, n = chars.most_common(1)[0]
    return TURN_FOR_DIRECTION.get(direction, 0) if n > 0.5 * total else 0


def prepare_pdf(pdf_path: Path, work_dir: Path):
    """Elements 18 and 28, BEFORE parsing. Works on a temporary copy, the
    original PDF is never changed.
      28: rotated pages -> rotation removed, and pages whose text is
          sideways are turned upright, so text and tables read normally
      18: scanned pages -> OCR with ocrmypdf (only pages without text),
          if ocrmypdf is installed
    Returns (path_to_parse, info)."""
    info = {"rotated_pages": [], "ocr_done": False}
    doc = pymupdf.open(str(pdf_path))
    try:
        for page in doc:
            changed = False
            if page.rotation:                        # page has a /Rotate setting
                page.remove_rotation()
                changed = True
            turn = sideways_turn(page)               # text itself is sideways
            if turn:
                page.set_rotation(turn)
                page.remove_rotation()
                changed = True
            if changed:
                info["rotated_pages"].append(page.number + 1)
        scanned = [p.number + 1 for p in doc if is_scanned_page(p)]
        path = pdf_path
        if info["rotated_pages"]:
            path = work_dir / pdf_path.name          # same file name -> same doc_id
            doc.save(str(path))
    finally:
        doc.close()

    if scanned and not missing_ocr_tools():
        out = work_dir / ("ocr_" + pdf_path.name)
        run = subprocess.run(ocr_command() + ["--skip-text", "--quiet", str(path), str(out)],
                             capture_output=True, text=True)
        if run.returncode != 0:
            print(f"          OCR failed for {pdf_path.name}: {(run.stderr or '').strip()[:300]}")
        if run.returncode == 0:
            final = work_dir / pdf_path.name
            out.replace(final)
            path, info["ocr_done"] = final, True
    return path, info


def ocr_command():
    """'ocrmypdf' on PATH, or 'python -m ocrmypdf' when it is only pip-installed in the venv."""
    if shutil.which("ocrmypdf"):
        return ["ocrmypdf"]
    import importlib.util
    if importlib.util.find_spec("ocrmypdf"):
        import sys
        return [sys.executable, "-m", "ocrmypdf"]
    return []


def missing_ocr_tools():
    """Which OCR tools are not installed: ocrmypdf (pip), Tesseract and Ghostscript (system)."""
    missing = []
    if not ocr_command():
        missing.append("ocrmypdf  (pip install ocrmypdf)")
    if not shutil.which("tesseract"):
        missing.append("Tesseract (see README: OCR setup)")
    if not any(shutil.which(g) for g in ("gs", "gswin64c", "gswin32c")):
        missing.append("Ghostscript (see README: OCR setup)")
    return missing


def inspect_pages(doc):
    """Elements 18, 20, 23, 24, 25, 26: look at every page once.
    Nothing is removed here; it only reports what it finds."""
    report = {"scanned_pages": [], "bad_encoding_pages": [], "blank_pages": [],
              "low_text_pages": [], "boilerplate_pages": [],
              "watermarks": [], "stamp": None, "forms": [], "annotations": []}
    marks = defaultdict(set)                                   # text -> pages
    for page in doc:
        n = page.number + 1
        text = page.get_text("text")
        words = len(text.split())

        if is_scanned_page(page):                              # 18
            report["scanned_pages"].append(n)
        elif words == 0:
            report["blank_pages"].append(n)                    # 24
        elif words < 30:
            report["low_text_pages"].append(n)                 # 24

        chars = sum(1 for c in text if not c.isspace())        # 20
        bad = len(GARBAGE.findall(text)) + sum(1 for c in PRIVATE_USE.findall(text)
                                               if c not in SYMBOL_MAP)
        if chars >= 50 and bad > 0.05 * chars:
            report["bad_encoding_pages"].append(n)

        if words < 150 and BOILERPLATE.search(text):           # 24
            report["boilerplate_pages"].append(n)

        found, rotated_chars, all_chars = [], 0, 0             # 23
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                rotated = abs(line["dir"][1]) > 0.1             # not left-to-right
                for span in line["spans"]:
                    t = span["text"].strip()
                    all_chars += len(t)
                    rotated_chars += len(t) if rotated else 0
                    c = span["color"]
                    light = min((c >> 16) & 255, (c >> 8) & 255, c & 255) >= 190
                    if len(t) >= 3 and (rotated or (light and span["size"] >= 20)):
                        found.append((t, rotated))
        for t, rotated in found:
            # rotated text is a watermark only if it is a SMALL part of the
            # page; a page that is mostly rotated text is real content
            if not rotated or rotated_chars < 0.3 * max(all_chars, 1):
                marks[t].add(n)

        for w in page.widgets() or []:                         # 25
            if w.field_value not in (None, "", "Off"):
                report["forms"].append({"page": n, "field": w.field_name,
                                        "value": str(w.field_value)})

        for a in page.annots() or []:                          # 26
            note = (a.info or {}).get("content", "").strip()
            if note:
                report["annotations"].append({"page": n, "type": a.type[1], "text": note})

    need = 1 if len(doc) == 1 else 2               # watermark = repeated on 2+ pages
    report["watermarks"] = sorted(t for t, pages in marks.items() if len(pages) >= need)
    report["stamp"] = next((t for t in report["watermarks"] if STAMP_WORDS.search(t)), None)
    return report


def remove_letter_blocks(sections):
    """Element 27: remove the letter parts of a circular that are not rules:
      start: "To, All Scheduled Banks ..." up to "Madam / Dear Sir,"
      end:   "Yours faithfully" and the short name / designation lines after it
    A "Subject:" line and all headings are kept."""
    stats = Counter()
    sal = next((i for i, s in enumerate(sections)
                if s["type"] == "text" and s["page"] <= 2 and SALUTATION.search(s["text"])), None)
    if sal is not None:
        keep = []
        for i, s in enumerate(sections):
            if i < sal and s["type"] == "text" and s["page"] <= 2 \
                    and len(s["text"].split()) < 80 and not SUBJECT_LINE.match(s["text"]):
                stats["address_blocks_removed"] += 1          # address / ref lines
                continue
            if i == sal:
                s["text"] = SALUTATION.sub("", s["text"]).strip()
            keep.append(s)
        sections = keep

    close = next((i for i in range(len(sections) - 1, -1, -1)
                  if sections[i]["type"] == "text" and CLOSING.search(sections[i]["text"])), None)
    if close is not None:
        s = sections[close]
        s["text"] = s["text"][:CLOSING.search(s["text"]).start()].strip()
        end = close + 1
        while (end < len(sections) and sections[end]["type"] == "text"
               and sections[end]["part"] == s["part"]
               and sections[end]["page"] <= s["page"] + 1
               and len(sections[end]["text"].split()) < 40):
            end += 1                                          # name, designation, seal
        stats["signature_blocks_removed"] += 1 + (end - close - 1)
        sections = sections[:close + 1] + sections[end:]
    return [s for s in sections if s["text"].strip()], stats


def fix_section_text(sections, watermarks):
    """Elements 19, 21, 22, 23 on every section:
      23: remove watermark / stamp text
      21: map font bullets to "-" and "` 500" to "₹ 500"
      19: tag lang = "en" / "hi" / "mixed" (both languages are kept)
      22: flag has_formula = True when the text has many math symbols"""
    stats = Counter()
    wm = [re.compile(r"\s*".join(map(re.escape, w.split()))) for w in watermarks]

    def fix(text):
        for pat in wm:
            text, n = pat.subn("", text)
            stats["watermark_text_removed"] += n
        for bad, good in SYMBOL_MAP.items():
            if bad in text:
                stats["symbols_fixed"] += text.count(bad)
                text = text.replace(bad, good)
        text, n = RUPEE_BACKTICK.subn("₹", text)
        stats["symbols_fixed"] += n
        stats["unknown_symbols"] += text.count("\ufffd")
        return re.sub(r"(?<=\S)[ \t]{2,}", " ", text).strip()   # keep list indents

    for s in sections:
        s["text"] = fix(s["text"])
        if s["type"] == "heading" and not s["text"].strip("# "):
            s["text"] = ""                       # heading was only a watermark
        if wm and s.get("heading_path"):
            parts = [h for h in s["heading_path"].split(" > ")]
            for pat in wm:
                parts = [pat.sub("", h).strip() for h in parts]
            s["heading_path"] = " > ".join(h for h in parts if h)
        if s.get("rows"):                        # Fix A4b: clean every table cell,
            s["rows"] = [[fix(c) for c in row] for row in s["rows"]]
            s["text"] = table_to_markdown(s["rows"])     # then rebuild both views
            s["rows_text"] = table_rows_as_text(s["rows"])
        elif s.get("rows_text"):
            s["rows_text"] = [fix(r) for r in s["rows_text"]]
        letters = sum(c.isalpha() for c in s["text"])
        if letters:
            hindi = sum(1 for c in s["text"]
                        if c.isalpha() and DEVANAGARI.match(c)) / letters
            s["lang"] = "hi" if hindi > 0.6 else "mixed" if hindi > 0.1 else "en"
        if len(MATH.findall(s["text"])) >= 3:
            s["has_formula"] = True
            stats["formula_sections"] += 1
    return [s for s in sections if s["text"].strip()], stats


# ---------------------------------------------------------------------------
# Document title: never a letterhead, stamp, watermark, date, number or label
# ---------------------------------------------------------------------------

SUBJECT_TITLE = re.compile(r"^\s*(?:sub(?:ject)?|re)\s*[:.\-–]\s*(.{8,200})$", re.I | re.M)
                               # own name: SUBJECT_LINE (above) is used by remove_letter_blocks
LETTERHEAD = re.compile(       # "Reserve Bank of India", "RBI - Central Office - Mumbai", "www.rbi.org.in"
    r"^\s*(?:reserve\s+bank\s+of\s+india|rbi|www\.rbi\.org\.in)\b"
    r"(?:[\s,\-–/]*(?:central\s+office|head\s+office|mumbai|new\s+delhi|india"
    r"|(?:[A-Za-z&]+\s+){0,4}department(?:\s+of\s+[A-Za-z&\s]+)?|website[:\s]*\S*|\S+\.(?:in|org)\S*))*\s*$", re.I)
LABEL_ONLY = re.compile(
    r"^\s*(?:annex(?:ure)?|appendix|schedule|attachment|exhibit|enclosure|table\s+of\s+contents"
    r"|contents|index|preface|foreword|notification|circular|press\s+release|draft|page)"
    r"\b[\s\-–.:]*(?:[IVXLC]+|[A-Z]|\d+)?[\s?.:]*$", re.I)


def bad_title(text, watermarks=()) -> str:
    """Why this text cannot be a title ('' = it is fine)."""
    text = (text or "").strip(" #*>-–:|\t")
    letters = [c for c in text if c.isalpha()]
    if len(re.findall(r"[A-Za-z]{3,}", text)) < 2 and not LABEL_ONLY.match(text):
        if not letters or sum(1 for c in letters if c.isascii()) / len(letters) < 0.7:
            return "not English / garbled"                    # Hindi letterhead, broken font
        return "too few words"                                # "~14%", "schedule?"
    if letters and sum(1 for c in letters if c.isascii()) / len(letters) < 0.7:
        return "not English / garbled"
    if GARBAGE.search(text) or PRIVATE_USE.search(text):
        return "broken characters"
    if STAMP_WORDS.fullmatch(text.strip(" .!")) or (STAMP_WORDS.search(text) and len(text.split()) <= 3):
        return "stamp"                                        # "Withdrawn"
    if any(w and w.lower() in text.lower() and len(text) <= len(w) + 10 for w in watermarks):
        return "watermark"
    if ANY_DATE.search(text) and len(re.findall(r"[A-Za-z]{3,}", re.sub(DATE_VALUE, "", text))) < 2:
        return "date"                                         # "June 05, 2026"
    if LABEL_ONLY.match(text):
        return "label"                                        # "ANNEXURE- I", "Schedule?"
    if LETTERHEAD.match(text):
        return "letterhead"                                   # same on every RBI document
    if text.endswith("?") and len(text.split()) <= 4:
        return "question fragment"
    if len(text) > 200:
        return "too long"
    # 2.3: the new wrong titles the review found
    if text[:1].islower():
        return "sentence fragment"                            # "section (2) of section 5 of ..."
    if re.match(r"^chapter\s+[ivxlc\d]+\b[\s\-–:.]*(?:preliminary|introduction|general|definitions|"
                r"short\s+title\b.*)?\s*$", text, re.I):
        return "chapter heading"                              # "Chapter I – Preliminary"
    if (RBI_REF_NO.search(text) or re.fullmatch(r"[A-Z]{2,6}(?:[./ -][A-Za-z0-9]+)*[/.]\d[\w./-]*", text)) \
            and len(re.findall(r"[A-Za-z]{3,}", RBI_REF_NO.sub("", text))) < 2:
        return "circular number"                              # "RBI/DBR/2015-16/19"
    if re.match(r"^[A-Z][.)]?\s+[A-Z][a-z]", text) and not re.match(r"^[AI]\s", text):
        return "sub-section heading"                          # "B Scope of counterparty credit risk"
    if re.match(r"^\d{1,3}[.)]\s", text):
        return "numbered item"                                # "1. What is RB-IOS, 2026?"
    if re.match(r"^note\b", text, re.I):
        return "note"                                         # "Note: The respondent company ..."
    if "_" in text or re.fullmatch(r"RBI\s*(?:\([^)]*\)\s*)*Direction|RBI(?:\s+\w+)?", text):
        return "file name"                                    # "RBI (FAQ1) Direction", "RBI Issues"
    return ""


# ---- 2.3: the OFFICIAL title from page 1 (review: new wrong titles such as
# "section (2) of section 5 of ...", "Chapter I – Preliminary", "B Scope of ...", "RBI/DBR/2015-16/19")
TITLE_DIRECTION = re.compile(r"^(?:Master\s+Direction\s*[-–]\s*)?(?:Reserve\s+Bank\s+of\s+India|RBI)\s*\(", re.I)
TITLE_DIRECTION_END = re.compile(r"\bDirections?\b[^()]*?\b(?:19|20)\d{2}\b", re.I)
SALUTATION = re.compile(r"^(?:dear\s+)?(?:sir|madam)\b[\s/,]*(?:(?:dear\s+)?(?:sir|madam)\b)?[\s,.]*$", re.I)
DATE_ONLY = re.compile(r"^\s*(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
                       r"(?:\d{1,2},?\s+)?\d{4}\s*$", re.I)
BODY_START = re.compile(r"^(?:please|the\s|today|good\s+morning|in\s|on\s|we\s|as\s|with\s|this\s|"
                        r"declaration|reference|\d+\.\s|\(\w{1,4}\)\s)", re.I)
UPDATED_NOTE = re.compile(r"\s*\((?:updated|amended)\s+(?:as\s+)?(?:on|upto|up\s+to)[^)]*\)+", re.I)
CONNECTOR_END = re.compile(r"(?:\b(?:of|for|on|and|in|the|to|by|with|under|from)|[-–,(/])\s*$", re.I)


def page_one_lines(pdf_path, max_lines=80):
    """[(text, size, bold)] of page 1, in reading order, empty lines dropped."""
    out = []
    try:
        with pymupdf.open(str(pdf_path)) as doc:
            if not len(doc):
                return out
            for block in doc[0].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = [sp for sp in line["spans"] if sp["text"].strip()]
                    if not spans:
                        continue
                    text = re.sub(r"\s+", " ", "".join(sp["text"] for sp in spans)).strip()
                    bold = any(sp["flags"] & 16 or "bold" in sp["font"].lower() for sp in spans)
                    out.append((text, round(max(sp["size"] for sp in spans), 1), bold))
                    if len(out) >= max_lines:
                        return out
    except Exception:
        pass
    return out


def _run(lines, i, limit=3, stop_at_body=True):
    """Join line i with the lines after it that belong to the same title: same bold style and
    font size, or (no bold used) a line that ends in "of", "for", "-" ... and continues."""
    text, size, bold = lines[i]
    parts = [text]
    for t, sz, b in lines[i + 1:i + limit]:
        same_style = bold and b and abs(sz - size) <= 0.6
        continues = not bold and CONNECTOR_END.search(parts[-1]) and not BODY_START.match(t)
        if (not (same_style or continues) or (stop_at_body and BODY_START.match(t)) or DATE_ONLY.match(t)
                or NUMBERED_QUESTION.match(t)):
            break
        parts.append(t)
    return " ".join(parts)


def _clean_title(text):
    text = UPDATED_NOTE.sub("", text)
    text = re.sub(r"(?<=[a-z])\((?=[A-Z])", " (", text)             # "India(Know" -> "India (Know"
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-–")
    if text.isupper() and len(text.split()) >= 3:            # "REGULATION, SUPERVISION AND ..."
        small = {"and", "of", "the", "for", "to", "in", "on", "with", "by", "a", "an", "be"}
        words = [w.lower() if w.lower() in small else (w if w in {"RBI", "KYC", "NBFC", "MSME", "UCB"} else w.capitalize())
                 for w in text.split()]
        words[0] = words[0][:1].upper() + words[0][1:]
        text = " ".join(words)
    return text[:200]


def official_title(pdf_path) -> str:
    """The title RBI itself prints on page 1, found by the place it always has:
      1. 'Reserve Bank of India (X) Directions, 2025'  (+ the next line when it wraps)
      2. the subject line right after 'Dear Sir / Madam,' (letters, master circulars)
      3. the line after the date in a press release / statement
      4. the 'Frequently Asked Questions ...' heading of an FAQ (+ the survey name above it)
      5. a first line in CAPITALS ('FORM OF COMPLAINT ...', chapter of a report)
    '' when none is found (choose_title then tries its other candidates)."""
    raw = page_one_lines(pdf_path)
    for i, (t, _, _) in enumerate(raw):                                # 1 (raw lines: "Directions, 2026"
        if TITLE_DIRECTION.match(t):                                   #    alone on a line is kept)
            parts = [t]
            for nxt, _, _ in raw[i + 1:i + 3]:
                if TITLE_DIRECTION_END.search(" ".join(parts)) or BODY_START.match(nxt) \
                        or re.match(r"^reserve\s+bank\s+(?:had|has)\b", nxt, re.I):
                    break
                parts.append(nxt)
            return _clean_title(" ".join(parts))

    keep = [i for i, (t, _, _) in enumerate(raw)                      # lines that may START a title
            if not bad_title(t) or SALUTATION.match(t) or DATE_ONLY.match(t)]
    texts = [raw[i][0] for i in keep]

    for n, i in enumerate(keep[:-1]):                                  # 2
        if SALUTATION.match(raw[i][0]):
            nxt = keep[n + 1]
            if not BODY_START.match(raw[nxt][0]) and len(raw[nxt][0]) <= 160:
                return _clean_title(_run(raw, nxt))

    for n, i in enumerate(keep[:15]):                                  # 3
        if DATE_ONLY.match(raw[i][0]) and n + 1 < len(keep) and not BODY_START.match(raw[keep[n + 1]][0]):
            title = _run(raw, keep[n + 1])
            if not bad_title(title):
                return _clean_title(title)

    for i, (t, _, bold) in enumerate(raw[:30]):                       # 4
        if re.search(r"frequently\s+asked\s+questions", t, re.I):
            title = _run(raw, i, stop_at_body=False) if not t.rstrip().endswith(")") else t
            title = re.sub(r"\s+(?:General\s+(?:Instructions|Information))$", "", title, flags=re.I)
            bare = re.fullmatch(r"frequently\s+asked\s+questions\s*(?:\(FAQs?\))?", t.strip(), re.I)
            if bare and title.strip() != t.strip():                    # "FAQs" + "Integrated Ombudsman Scheme"
                return _clean_title(title[len(t):].strip() + " – Frequently Asked Questions")
            before, j = [], i - 1
            while j >= 0 and raw[j][2] == bold and not re.search(
                    r"reserve\s+bank|department|division|mumbai|^rbi\b", raw[j][0], re.I):
                before.insert(0, raw[j][0])
                j -= 1
            return _clean_title(" ".join(before) + " – " + title if before else title)

    first_line = raw[0][0] if raw else ""
    questions = [i for i, (t, _, _) in enumerate(raw[:40]) if NUMBERED_QUESTION.match(t)]
    if len(questions) >= 2 and (NUMBERED_QUESTION.match(first_line) or re.match(r"^[A-H][.)]\s", first_line)):
        first = questions[0]                                           # 6: FAQ with NO printed title
        lettered = re.match(r"^[A-H][.)]\s+(.{8,120})$", raw[first - 1][0]) if first else None
        topic = lettered.group(1) if lettered else re.sub(r"^\s*(?:Q\.?\s*)?\d{1,3}[.)]\s*", "", raw[first][0])
        return _clean_title("Frequently Asked Questions: " + topic)

    for t in texts[:6]:                                                # 5
        letters = [c for c in t if c.isalpha()]
        if (len(t.split()) >= 3 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.8
                and not LETTERHEAD.match(t)):
            return _clean_title(t)
    return ""


def title_from_file_name(path: Path) -> str:
    """'RBI (Payments Banks)Direction' -> 'RBI (Payments Banks) Direction'."""
    name = re.sub(r"[_]+|(?<=\w)-(?=\w)", " ", path.stem)
    name = re.sub(r"\)(?=\w)", ") ", name)
    return re.sub(r"\s{2,}", " ", name).strip()


def choose_title(parsed, sections, pdf_path: Path, watermarks):
    """First GOOD candidate: subject line, title found by the parser, headings, file name."""
    candidates = [official_title(pdf_path)]                     # 2.3: RBI's own title on page 1
    candidates += [m.group(1) for m in SUBJECT_TITLE.finditer(parsed.get("original_head", ""))][:1]
    candidates.append(parsed.get("title", ""))
    candidates += [s["text"].lstrip("# ").strip() for s in sections if s["type"] == "heading"][:15]
    for c in candidates:
        c = re.sub(r"\s+", " ", c or "").strip(" #*>-–:|")
        if c and not bad_title(c, watermarks):
            return c[:200]
    return title_from_file_name(pdf_path)


def extra_cleaning(parsed, pdf_path: Path, prep: dict):
    """Elements 18-27: one call after parse_pdf(). Order:
      1. keep the first page text for the version metadata (element 31)
      2. remove cover / disclaimer pages            (24)
      3. remove address and signature blocks        (27)
      4. watermarks, symbols, language, formulas    (23, 21, 19, 22)
      5. add filled form fields as a section        (25)
      6. save page checks and annotations           (18, 20, 26, 28)"""
    doc = pymupdf.open(str(pdf_path))
    try:
        report = inspect_pages(doc)
    finally:
        doc.close()

    # 1. the address / date lines are removed below, but element 31 needs them
    parsed["original_head"] = to_markdown(parsed)[:METADATA_HEAD_CHARS]

    sections, stats = parsed["sections"], Counter()

    # 2. cover page = a first page with very few words (not a scan)
    drop = set(report["boilerplate_pages"])
    if parsed["num_pages"] > 1 and 1 in report["low_text_pages"]:
        drop.add(1)
    # Fix A1: never drop a short page that holds a model no. / version / RBI ref
    page_text = defaultdict(str)
    for s in sections:
        page_text[s["page"]] += " " + s["text"]
    protected = {p for p in drop if IDENTIFIER.search(page_text[p])}
    drop -= protected
    if protected:
        stats["cover_pages_kept_for_ids"] = len(protected)
    kept = [s for s in sections if s["page"] not in drop]
    stats["cover_or_disclaimer_sections_removed"] = len(sections) - len(kept)

    # 3 and 4
    kept, st = remove_letter_blocks(kept)
    stats.update(st)
    kept, st = fix_section_text(kept, report["watermarks"])
    stats.update(st)
    old_title = parsed["title"]
    parsed["title"] = choose_title(parsed, kept, pdf_path, report["watermarks"])
    if parsed["title"] != old_title:
        stats["title_replaced"] += 1
    for s in kept:
        parts = [p for p in (s.get("heading_path") or "").split(" > ") if p]
        parts = [p for p in parts if not bad_title(p, report["watermarks"])]   # no stamp/date/letterhead
        s["heading_path"] = " > ".join(parts) or parsed["title"]

    # 5. form fields, one section per page
    by_page = defaultdict(list)
    for f in report["forms"]:
        by_page[f["page"]].append(f"- {f['field']}: {f['value']}")
    for page_no, rows in sorted(by_page.items()):
        kept.append({"type": "form", "text": "Form fields:\n" + "\n".join(rows),
                     "page": page_no, "heading_path": parsed["title"], "part": "forms"})

    # 6. page checks (nothing here goes into the text)
    parsed["sections"] = kept
    parsed["page_quality"] = {
        "ocr_needed_pages": [] if prep["ocr_done"] else
                            sorted(set(report["scanned_pages"]) | set(report["bad_encoding_pages"])),
        "scanned_pages": report["scanned_pages"],
        "bad_encoding_pages": report["bad_encoding_pages"],
        "blank_pages": report["blank_pages"],
        "rotated_pages": prep["rotated_pages"],
        "ocr_done": prep["ocr_done"],
        "watermarks": report["watermarks"],
        "languages": sorted({s["lang"] for s in kept if s.get("lang")}),
    }
    parsed["annotations"] = report["annotations"]
    parsed["stamp"] = report["stamp"]
    parsed["cleaning_stats"].update({k: v for k, v in stats.items() if v})
    return parsed


def main():
    ap = argparse.ArgumentParser(
        description="Clean PDFs into text for RAG"
    )
    ap.add_argument(
        "--input", default="raw_pdfs",
        help="folder with PDF files"
    )
    ap.add_argument(
        "--output", default="output",
        help="folder for results"
    )
    ap.add_argument("--timeout", type=int, default=PARSE_TIMEOUT_SEC,
                    help="seconds allowed per PDF before it is skipped")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES,
                    help="skip PDFs with more pages than this")
    ap.add_argument("--max-size-mb", type=int, default=MAX_SIZE_MB,
                    help="skip PDFs bigger than this (MB)")
    ap.add_argument("--force", action="store_true",
                    help="clean ALL PDFs again, even unchanged ones")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)

    (out_dir / "json").mkdir(parents=True, exist_ok=True)
    (out_dir / "markdown").mkdir(parents=True, exist_ok=True)
    skip_log = out_dir / "skipped_files.csv"
    json_dir = out_dir / "json"

    pdfs = sorted(in_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {in_dir}/")
        return

    # 1. Initialize counters
    ok = 0
    new_count = 0
    changed_count = 0
    unchanged_count = 0
    removed_count = 0
    password_count = 0
    failed_count = 0
    timeout_count = 0
    timed_out = []                   # review N2: names, so a lost file is never silent
    duplicate_count = 0

    # 2. Load what earlier runs saved
    previous = load_previous_outputs(json_dir)

    # Element 32: content fingerprints of files already saved
    seen_content_hashes = {}
    # File fingerprints of files already saved (catches exact copies fast)
    seen_file_hashes = {}
    current_names = {p.name for p in pdfs}
    for name, entry in previous.items():
        if name not in current_names:
            continue                         # deleted PDF: cannot be the "original"
        if entry["content_hash"]:
            seen_content_hashes.setdefault(entry["content_hash"], name)
        if entry["file_hash"]:
            seen_file_hashes.setdefault(entry["file_hash"], name)

    if args.force:
        print("FORCE: every PDF will be cleaned again")

    # 3. Process each PDF
    for pdf in pdfs:
        old = previous.get(pdf.name)

        # Incremental: FIRST check if this PDF is unchanged.
        # Only the file bytes are read; the PDF is not opened or cleaned.
        file_hash = None
        try:
            if pdf.stat().st_size <= args.max_size_mb * 1024 * 1024:
                file_hash = file_sha256(pdf)
        except OSError:
            pass                             # validate_pdf_file reports it

        if (old is not None and not args.force and file_hash
                and old["pipeline_version"] == PIPELINE_VERSION
                and old["file_hash"] == file_hash):
            unchanged_count += 1
            print(f"UNCHANGED: {pdf.name} (skipped, already cleaned)")
            continue

        if old is None:
            status = "NEW"
        elif args.force:
            status = "FORCED"
        elif old["pipeline_version"] != PIPELINE_VERSION:
            status = "NEW CODE"              # cleaned with older cleaning rules
        else:
            status = "CHANGED"

        # Element 29: password protection
        if check_password_pdf(pdf):
            password_count += 1
            log_skip(skip_log, pdf, "password_protected")
            if old:
                remove_outputs(old)          # Fix A4a: no stale output left behind
            continue

        # Element 30: file size, page count, and PDF readability
        valid, reason = validate_pdf_file(pdf, args.max_size_mb, args.max_pages)
        if not valid:
            failed_count += 1
            log_skip(skip_log, pdf, reason.split(":")[0], reason)
            if old:
                remove_outputs(old)          # Fix A4a: no stale output left behind
            continue

        # Element 32 (fast): exact same file as another one already saved
        twin = seen_file_hashes.get(file_hash)
        if twin and twin != pdf.name:
            print(f"SKIPPED DUPLICATE: {pdf.name} (same file as {twin})")
            duplicate_count += 1
            log_skip(skip_log, pdf, "duplicate", f"same file as {twin}")
            if old:
                remove_outputs(old)          # its old output is now a duplicate
            continue

        try:
            # Element 30: parse with a time limit
            started = time.monotonic()
            parsed = parse_pdf_with_timeout(pdf, args.timeout)
            seconds = time.monotonic() - started

            # Element 32: duplicate cleaned content
            # (same file name = this file's own earlier output -> not a duplicate)
            content_hash = cleaned_text_hash(parsed)

            if (content_hash and content_hash in seen_content_hashes
                    and seen_content_hashes[content_hash] != pdf.name):
                original = seen_content_hashes[content_hash]
                print(
                    f"SKIPPED DUPLICATE: {pdf.name} "
                    f"(same cleaned content as {original})"
                )
                duplicate_count += 1
                log_skip(skip_log, pdf, "duplicate", f"same as {original}")
                if old:
                    remove_outputs(old)
                continue

            # Element 31: document version/date metadata
            cleaned_text = to_markdown(parsed)
            head = parsed.pop("original_head", "")      # first page before element 27
            parsed["metadata"] = extract_version_metadata(head + "\n" + cleaned_text)
            # 2.4: some PDFs lose page 1 in parsing (a table-of-contents page, a cover). Fill the
            # fields that are still empty from the RAW first page (circular no., dates).
            meta = parsed["metadata"]
            if not (meta.get("issue_date") and meta.get("updated_on") and
                    (meta.get("circular_no") or meta.get("doc_type") != "rbi_circular")):
                raw_meta = extract_version_metadata(raw_cover_text(pdf, pages=1))
                for field in ("issue_date", "issue_date_raw", "updated_on", "circular_no"):
                    if not meta.get(field) and raw_meta.get(field):
                        if field == "circular_no" and meta.get("doc_type") != "rbi_circular":
                            continue
                        meta[field] = raw_meta[field]
            # The cover page (often only "Annual Report 2025-26" in big letters) can be removed
            # by the cleaning above, so also check the RAW first 3 pages and the PDF's own title.
            if parsed["metadata"].get("doc_type") != "annual_report" and is_annual_report_pdf(pdf):
                parsed["metadata"]["doc_type"] = "annual_report"
                print("          type    : annual_report (found on the raw cover pages)")
            elif parsed["metadata"].get("doc_type") == "annual_report" and not is_annual_report_pdf(pdf):
                parsed["metadata"]["doc_type"] = "general"   # only a page footer said "Annual Report"
                print("          type    : general (RBI Annual Report chapter, kept)")
            if parsed.get("stamp"):                      # Element 23: "SUPERSEDED" stamp
                parsed["metadata"]["stamp"] = parsed["stamp"]
                parsed["metadata"]["is_latest"] = False

            # Fingerprints for the next run
            parsed["file_hash"] = file_hash
            parsed["pipeline_version"] = PIPELINE_VERSION
            seen_file_hashes[file_hash] = pdf.name
            if content_hash:
                seen_content_hashes[content_hash] = pdf.name
                parsed["content_hash"] = content_hash

        except TimeoutError as e:
            timeout_count += 1
            timed_out.append(pdf.name)
            print(f"TIMEOUT {pdf.name}: {e}")
            log_skip(skip_log, pdf, "timeout", str(e))
            if old:
                remove_outputs(old)          # Fix A4a: no stale output left behind
            continue

        except Exception as e:
            failed_count += 1
            print(f"FAILED {pdf.name}: {e}")
            log_skip(skip_log, pdf, "parse_failed", str(e))
            if old:
                remove_outputs(old)          # Fix A4a: no stale output left behind
            continue

        # 4. Save JSON and Markdown
        ok += 1
        if status == "NEW":
            new_count += 1
        else:
            changed_count += 1
        doc_id = parsed["doc_id"]
        add_page_end(parsed["sections"])                 # page range for every section

        (out_dir / "json" / f"{doc_id}.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        (out_dir / "markdown" / f"{doc_id}.md").write_text(
            cleaned_text,                            # built once above
            encoding="utf-8"
        )

        # Confirm metadata was really saved into the JSON file
        json_path = out_dir / "json" / f"{doc_id}.json"
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        metadata_saved = "metadata" in saved and "content_hash" in saved

        words = sum(
            len(s["text"].split()) for s in parsed["sections"]
        )
        n_tables = sum(
            1 for s in parsed["sections"] if s["type"] == "table"
        )

        warn = (
            "  <- few words, scanned? run OCR"
            if words < 50 * parsed["num_pages"] else ""
        )

        print(
            f"{status + ':':<10}{pdf.name}: {parsed['num_pages']} pages, "
            f"{len(parsed['sections'])} sections, "
            f"{n_tables} tables, "
            f"{parsed['multi_column_pages']} multi-column pages, "
            f"headings from {parsed['headings_from']}, "
            f"{words} words, {seconds:.1f}s{warn}"
        )
        print(f"          cleaning: {parsed['cleaning_stats']}")
        q = parsed.get("page_quality", {})
        if q.get("ocr_needed_pages"):
            missing = missing_ocr_tools()
            print(f"          WARNING: pages {q['ocr_needed_pages']} need OCR (scanned or broken text)"
                  + (f" - missing: {', '.join(missing)}" if missing else " - OCR ran but did not fix them"))
        if q.get("ocr_done"):
            print("          OCR done on scanned pages")
        if q.get("rotated_pages"):
            print(f"          rotated pages fixed: {q['rotated_pages']}")
        if q.get("languages") and q["languages"] != ["en"]:
            print(f"          languages: {q['languages']}")
        if parsed.get("annotations"):
            print(f"          annotations saved: {len(parsed['annotations'])}")
        if parsed.get("stamp"):
            print(f"          stamp found: '{parsed['stamp']}' -> is_latest = False")
        meta = parsed["metadata"]
        if meta["doc_type"] == "manual":
            print(f"          manual  : model={meta['model']}, product={meta['product']}, "
                  f"brand={meta['brand']}, version={meta['doc_version']}, "
                  f"issue_date={meta['issue_date']}")
        else:
            print(f"          version : circular_no={meta['circular_no']}, "
                  f"issue_date={meta['issue_date']} (raw: {meta['issue_date_raw']}), "
                  f"supersedes={meta['supersedes']}")
        required = DOC_TYPES.get(meta["doc_type"], {}).get("required", [])
        missing = [f for f in required if not meta.get(f)]          # Fix A2
        if missing:
            print(f"          WARNING: {meta['doc_type']}: {', '.join(missing)} not found "
                  f"- check page 1 of this PDF")
        if metadata_saved:
            print(f"          METADATA ADDED SUCCESSFULLY -> {json_path.name}")
        else:
            print(f"          WARNING: metadata NOT found in {json_path.name}")

    # 5. PDFs deleted from the input folder -> remove their old outputs
    for name, entry in previous.items():
        if name not in current_names:
            remove_outputs(entry)
            removed_count += 1
            print(f"REMOVED: {name} (no longer in {in_dir}/, old output deleted)")

    # 6. Element 31: link old circulars to the newer ones that replace them
    #    (always checks ALL saved files: a new circular can replace an old,
    #     unchanged one)
    superseded_links = link_superseded(json_dir)
    print_final_metadata(json_dir)

    # 7. Create QA test-set template in the PROJECT folder (next to eval.py), not in output/:
    #    output/ is not committed to git, and the reviewer needs this file in the repo
    qa_file = Path("qa_test_set.csv")
    write_qa_template(qa_file)

    # 8. Print summary
    print(f"\nDone: {len(pdfs)} PDFs checked -> {out_dir}/")
    print(f"New: {new_count}   Changed: {changed_count}   "
          f"Unchanged (skipped): {unchanged_count}   Removed: {removed_count}")
    print(f"Password-protected PDFs: {password_count}")
    print(f"Failed PDFs (corrupt / too large / too many pages / parse error): {failed_count}")
    print(f"Timed-out PDFs: {timeout_count}")
    for name in timed_out:
        print(f"   NOT in the database: {name}  (run again with --timeout {args.timeout * 3})")
    print(f"Duplicate PDFs skipped: {duplicate_count}")
    print(f"Superseded links found: {superseded_links}")
    if password_count or failed_count or timeout_count or duplicate_count:
        print(f"Skipped files logged in {skip_log}")
    if ok:
        print(
            f"Next: check {out_dir}/markdown/, then write 30 questions "
            f"in {qa_file.resolve()}"
        )


if __name__ == "__main__":
    main()