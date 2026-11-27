"""
eval.py - test the RAG chatbot (Day 4) and every improvement (Day 5) on the SAME questions.

Test files (tab-separated, header row):
  id | question | expected_answer | source_doc_id | file_name | page | answer_type | notes
  - qa_test_set.csv : the main set (40 answerable + no_answer questions)
  - qa_heldout.csv  : the HELD-OUT set. Never used to tune anything: judge the extras on it.
  - page can be "2, 5" (hit if ANY of those pages is retrieved)
  - answer_type "no_answer" = out of scope: the bot must say "I could not find this ..."

One setup:
  python eval.py --strategy=recursive                          baseline (vectors only)
  python eval.py --strategy=recursive --hybrid                 + BM25
  python eval.py --strategy=recursive --hybrid --rerank        + re-ranker
  python eval.py --strategy=recursive --hybrid --rerank --rewrite
  ablation (review H1 / M5):  add --synonyms (on)  --no-stemming  --no-docfocus

Everything (the evidence for the review):
  python eval.py --bm25-pages             plain BM25 over whole pages: the floor to beat
  python eval.py --all --setups           3 strategies x (base, +hybrid, +rerank, +rewrite)
                                          + 3 ablation rows on recursive+hybrid: without
                                          DOC_FOCUS, without stemming, with synonyms
  python eval.py --report                 eval_report.md + the results table in README.md

Options for any line:
  --test=qa_heldout.csv   another question file          --k=5   k used for EVERY setup
  --retrieval-only        no answers: fast, scores only retrieval
  --ragas                 also faithfulness + answer relevance (separate venv, see README)
  --ids                   print the doc_id of every file in the database
  --resume                skip a run that already FINISHED with the same settings (same setup,
                          test file, database, retrieval-only / answers): after a stop, start the
                          same commands again with --resume and they go on where they stopped
  --grade                 hand-grade the answers that have no numbers (review N6), e.g.
                          python eval.py --grade --strategy=recursive --hybrid

Fair comparison: every setup is scored at the SAME k (Hit@1, Hit@3, Hit@k, MRR@k). The
chatbot itself answers from app.TOP_K = 5 passages.

Every result CSV is in test_result/: each run adds one row to test_result/eval_results.csv (with
all its settings) and writes test_result/eval_details_<test file>_<setup>.csv (every question: hit,
rank, answer, key facts, seconds). The earlier version of a file that is written again is moved to
test_result/old/<date_time>/, so test_result/ holds the latest results and nothing is lost.
"""

import csv
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import app

app.apply_cli(sys.argv)                          # review L5: app.py does not read sys.argv itself

EVAL_K = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--k=")), 5)
TEST_FILE = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--test=")),
                      "qa_test_set.csv"))
RETRIEVAL_ONLY = "--retrieval-only" in sys.argv
RESULT_DIR = Path("test_result")                 # every result CSV goes into this folder
RESULTS_FILE = RESULT_DIR / "eval_results.csv"
OLD_DIR = RESULT_DIR / "old" / datetime.now().strftime("%Y-%m-%d_%H%M")   # earlier versions, per run
REPORT_FILE = Path("eval_report.md")
README = Path("README.md")

NOT_FOUND = re.compile(r"could not find|not (?:available|mentioned|found|present) in the (?:documents|context)"
                       r"|no information|does not (?:contain|mention)", re.I)
COLUMNS = ["id", "question", "expected_answer", "source_doc_id", "file_name",
           "page", "answer_type", "notes"]


# ---------------------------------------------------------------------------
# 1. Load the test questions
# ---------------------------------------------------------------------------

def norm_name(text):
    """'Governor's Statement.pdf' and 'governor_s_statement' compare equal."""
    text = re.sub(r"\.pdf$", "", (text or "").strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_pages(text):
    """'2, 5' -> {2, 5};  '3' -> {3};  '' -> set()"""
    return {int(p) for p in re.findall(r"\d+", str(text or ""))}


def check_not_damaged(path, raw):
    """Excel 'Save as CSV' puts every line in quotes and cuts it at commas: stop clearly."""
    lines = [l for l in raw.splitlines() if l.strip()]
    wrapped = sum(1 for l in lines if l.startswith('"') and "\t" in l)
    if lines and wrapped > len(lines) / 2:
        sys.exit(f"{path} looks DAMAGED (saved by Excel: every line is in quotes and cut at commas).\n"
                 f"Put the original file back. To look at it, use Notepad or VS Code.")


def fix_encoding(raw):
    """'â‚¹' / 'â€“' (UTF-8 read as Windows-1252) -> the real characters."""
    if "â" not in raw and "Ã" not in raw:
        return raw
    try:
        return raw.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw


def read_csv_rows(path):
    data = path.read_bytes()
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raw = data.decode("cp1252", errors="replace")
    raw = fix_encoding(raw)
    check_not_damaged(path, raw)
    delimiter = "\t" if raw.count("\t") > raw.count(",") / 3 else ","
    rows = [r for r in csv.reader(raw.splitlines(), delimiter=delimiter) if any(c.strip() for c in r)]
    if rows and rows[0] and rows[0][0].strip().lower() == "id":
        header = [h.strip().lower() for h in rows[0]]
        aliases = {"source": "file_name", "answer": "expected_answer", "type": "answer_type"}
        header = [aliases.get(h, h) for h in header]
        return [dict(zip(header, r)) for r in rows[1:]]
    return [dict(zip(COLUMNS, r)) for r in rows]


def load_cases(path=None):
    path = Path(path or TEST_FILE)
    if not path.exists():
        sys.exit(f"No test file {path}. Put it next to eval.py, or use --test=yourfile.csv")
    cases = []
    for n, r in enumerate(read_csv_rows(path), start=1):
        question = (r.get("question") or "").strip()
        if not question:
            continue
        cases.append({
            "id": (r.get("id") or f"q{n:03d}").strip(),
            "question": question,
            "answer": (r.get("expected_answer") or "").strip(),
            "file": (r.get("file_name") or "").strip(),
            "doc_id": (r.get("source_doc_id") or "").strip(),
            "pages": parse_pages(r.get("page")),
            "type": (r.get("answer_type") or "").strip().lower() or "paragraph",
        })
    print(f"Test questions: {len(cases)} from {path} "
          f"({sum(c['type'] == 'no_answer' for c in cases)} no_answer)")
    return cases


# ---------------------------------------------------------------------------
# 2. Checks
# ---------------------------------------------------------------------------

def same_doc(meta, case):
    """Right document? By doc_id (survives renaming the PDF), else by file name."""
    doc_id = (meta.get("doc_id") or "").lower()
    if doc_id and case["doc_id"].lower() == doc_id:
        return True
    return norm_name(meta.get("source")) in {norm_name(case["file"]), norm_name(case["doc_id"])} - {""}


def is_hit(meta, case):
    """Right document AND one of the expected pages (a chunk covers page .. page_end)."""
    try:
        page = int(meta.get("page"))
        page_end = int(meta.get("page_end") or page)
    except (TypeError, ValueError):
        return False
    return same_doc(meta, case) and any(page <= p <= page_end for p in case["pages"])


NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
NUMBER_WORDS = {w: str(n) for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                     "eighty": "80", "ninety": "90", "hundred": "100", "half-yearly": "6", "twice": "2"})
NUMBER_WORD = re.compile(r"\b(" + "|".join(map(re.escape, NUMBER_WORDS)) + r")\b", re.I)
EMAIL_OR_URL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+", re.I)


def key_facts(text):
    """The checkable facts of an expected answer (review M7): its numbers, amounts, dates and
    e-mail / web addresses. 'Rs 20,000' -> '20000', '11.5 per cent' -> '11.5'."""
    facts = {f.lower().rstrip(".,)") for f in EMAIL_OR_URL.findall(text or "")}
    clean = EMAIL_OR_URL.sub(" ", text or "")
    for n in NUMBER.findall(clean):
        n = n.replace(",", "").rstrip(".")
        if n:
            facts.add(n)
    facts |= {NUMBER_WORDS[w.lower()] for w in NUMBER_WORD.findall(clean)}   # "three" = "3" (N6)
    return facts


def fact_score(expected, answer):
    """(found, total) of the expected answer's key facts in the bot's answer. total 0 = no facts."""
    facts = key_facts(expected)
    if not facts:
        return 0, 0
    have = key_facts(answer)
    low = (answer or "").lower()
    found = sum(1 for f in facts if f in have or ("@" in f and f in low))
    return found, len(facts)


# ---------------------------------------------------------------------------
# 3. Setups
# ---------------------------------------------------------------------------

def current_setup():
    return {"strategy": app.CHUNK_STRATEGY, "hybrid": app.USE_HYBRID, "rerank": app.USE_RERANK,
            "rewrite": app.USE_REWRITE, "synonyms": app.USE_SYNONYMS, "stemming": app.USE_STEMMING,
            "doc_focus": app.DOC_FOCUS}


def use_setup(s):
    app.CHUNK_STRATEGY, app.USE_HYBRID, app.USE_RERANK, app.USE_REWRITE = (
        s["strategy"], s["hybrid"], s["rerank"], s["rewrite"])
    app.USE_SYNONYMS, app.USE_STEMMING, app.DOC_FOCUS = s["synonyms"], s["stemming"], s["doc_focus"]


def label_of(s):
    parts = [s["strategy"]] + [n for n in ("hybrid", "rerank", "rewrite") if s[n]]
    parts += ["synonyms"] if s["synonyms"] else []
    parts += [f"-{n}" for n, key in (("stemming", "stemming"), ("docfocus", "doc_focus")) if not s[key]]
    if app.SEARCH_SCOPE != "all":
        parts.append(app.SEARCH_SCOPE)
    if app.MIN_RELEVANCE:
        parts.append(f"cutoff{app.MIN_RELEVANCE}")
    return "+".join(parts).replace("+-", "-")


def test_md5(path):
    """Fingerprint of the test file's TEXT: Windows (CRLF) and Linux (LF) line endings give the same
    md5, so a git checkout does not look like a new version of the test set (review N1)."""
    path = Path(path)
    return hashlib.md5(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:8] if path.exists() else ""


def run_settings(s, vector_db=None, chunks=""):
    md5 = test_md5(TEST_FILE)
    order = ("strategy", "hybrid", "rerank", "rewrite", "synonyms", "stemming", "doc_focus")
    return {**{k: s[k] for k in order}, "scope": app.SEARCH_SCOPE, "eval_k": EVAL_K, "answer_k": app.TOP_K,
            "retrieval_only": RETRIEVAL_ONLY,
            "chunk_size": app.CHUNK_SIZE, "chunk_overlap": app.CHUNK_OVERLAP,
            "db_version": app.DB_VERSION,
            "chunks_in_db": vector_db._collection.count() if vector_db is not None else chunks,
            "embed_model": app.EMBED_MODEL, "chat_model": f"ollama:{app.CHAT_MODEL}",
            "min_relevance": app.MIN_RELEVANCE,
            "include_annual_reports": app.INCLUDE_ANNUAL_REPORTS,
            "test_file": TEST_FILE.name, "test_file_md5": md5}


# ---------------------------------------------------------------------------
# 4. Run every question
# ---------------------------------------------------------------------------

def run_questions(vector_db, cases):
    details = []
    for n, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        docs = app.retrieve(vector_db, case["question"], top_k=EVAL_K)     # scored at the SAME k
        t1 = time.perf_counter()
        used = docs[:app.TOP_K]                                            # what the bot answers from
        old_used = False
        if RETRIEVAL_ONLY:
            answer = ""
        else:                                  # the same path as the chatbot, incl. the N4 fallback
            answer, used, old_used = app.answer_with_fallback(vector_db, case["question"], used,
                                                              top_k=app.TOP_K)
        t2 = time.perf_counter()
        if old_used:                           # N4: the bot answered from the second (old-version)
            docs = used                        # search, so that is what is scored
        row = {"case": case, "docs": docs, "used": used, "answer": answer, "old_fallback": old_used,
               "sec_retrieval": t1 - t0, "sec_answer": t2 - t1}
        try:
            row["score"] = app.top_relevance(vector_db, case["question"])  # for the cut-off advice
        except Exception:
            row["score"] = None

        if case["type"] == "no_answer":
            row["refused"] = None if RETRIEVAL_ONLY else bool(NOT_FOUND.search(answer))
            status = ("(no answer generated)" if row["refused"] is None
                      else "OK (refused)" if row["refused"] else "WRONG (answered)")
        else:
            row["rank"] = next((i for i, d in enumerate(docs, start=1) if is_hit(d.metadata, case)), None)
            row["doc_rank"] = next((i for i, d in enumerate(docs, start=1) if same_doc(d.metadata, case)), None)
            row["facts"] = fact_score(case["answer"], answer) if not RETRIEVAL_ONLY else (0, 0)
            status = (f"HIT @{row['rank']}" if row["rank"] else "DOC ONLY" if row["doc_rank"] else "MISS")
            if not RETRIEVAL_ONLY and not row["rank"] and not row["doc_rank"] and not app.is_refusal(answer):
                status += " but ANSWERED"      # review N3: a fluent answer from the wrong passages
        if old_used:
            status += " (old version)"
        print(f"  [{n}/{len(cases)}] {case['id']:<5} {case['type']:<13} {status:<18} "
              f"{row['sec_retrieval'] + row['sec_answer']:5.1f}s  {case['question'][:50]}")
        details.append(row)
    return details


def suggest_cutoff(details):
    """MIN_RELEVANCE advice: real questions should score ABOVE it, no_answer ones BELOW.
    Tuned on the DEV half only (questions 1, 3, 5, ...) and checked on the other half."""
    dev = details[0::2]
    real = [d["score"] for d in dev if d["case"]["type"] != "no_answer" and d.get("score") is not None]
    none = [d["score"] for d in dev if d["case"]["type"] == "no_answer" and d.get("score") is not None]
    if not real or not none:
        return None
    lowest_real, highest_none = min(real), max(none)
    if highest_none < lowest_real:
        cut = (highest_none + lowest_real) / 2
        held = details[1::2]
        ok = sum(1 for d in held if d.get("score") is not None
                 and (d["score"] >= cut) == (d["case"]["type"] != "no_answer"))
        return (f"  Suggested MIN_RELEVANCE  : {cut:.2f}  (dev half: no_answer max {highest_none:.2f} "
                f"< real min {lowest_real:.2f})\n"
                f"  ...on the other half     : {ok}/{len(held)} questions on the right side of {cut:.2f}")
    return (f"  MIN_RELEVANCE advice     : no clean cut on the dev half (no_answer max {highest_none:.2f}, "
            f"real min {lowest_real:.2f}). Keep 0 and let the model refuse.")


def score(details):
    answerable = [d for d in details if d["case"]["type"] != "no_answer"]
    no_answer = [d for d in details if d["case"]["type"] == "no_answer"]
    s = {"answerable": len(answerable), "no_answer": len(no_answer)}
    if answerable:
        for k in (1, 3):
            s[f"hit@{k}"] = sum(1 for d in answerable if d["rank"] and d["rank"] <= k) / len(answerable)
        s["hit_rate"] = sum(1 for d in answerable if d["rank"]) / len(answerable)
        s["mrr"] = sum(1 / d["rank"] for d in answerable if d["rank"]) / len(answerable)
        s["doc_hit_rate"] = sum(1 for d in answerable if d["doc_rank"]) / len(answerable)
        with_facts = [d for d in answerable if d["facts"][1]]
        if with_facts and not RETRIEVAL_ONLY:
            s["answer_correct"] = sum(d["facts"][0] / d["facts"][1] for d in with_facts) / len(with_facts)
        s["by_type"] = {t: (sum(1 for d in answerable if d["case"]["type"] == t and d["rank"]),
                            sum(1 for d in answerable if d["case"]["type"] == t))
                        for t in sorted({d["case"]["type"] for d in answerable})}
    if no_answer and not RETRIEVAL_ONLY:
        s["no_answer_acc"] = sum(1 for d in no_answer if d["refused"]) / len(no_answer)
    if answerable and not RETRIEVAL_ONLY:
        missed = [d for d in answerable if not d["rank"] and not d["doc_rank"]]
        answered = [d for d in missed if not app.is_refusal(d["answer"])]
        s["answered_on_miss"] = f"{len(answered)}/{len(missed)}"     # review N3
        s["old_fallback"] = sum(1 for d in details if d.get("old_fallback"))  # review N4
    s["sec_per_q"] = sum(d["sec_retrieval"] + d["sec_answer"] for d in details) / max(len(details), 1)
    return s


# ---------------------------------------------------------------------------
# 5. RAGAS: faithfulness + answer relevance (separate venv)
# ---------------------------------------------------------------------------

def score_ragas(details):
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    samples = [{"user_input": d["case"]["question"], "response": d["answer"],
                "retrieved_contexts": [doc.page_content for doc in d["used"]],
                "reference": d["case"]["answer"]}
               for d in details if d["case"]["type"] != "no_answer" and d["docs"]]
    result = evaluate(dataset=EvaluationDataset.from_list(samples),
                      metrics=[Faithfulness(), ResponseRelevancy()],
                      llm=LangchainLLMWrapper(app.llm),       # the SAME model grades its answers
                      embeddings=LangchainEmbeddingsWrapper(app.embeddings),
                      run_config=RunConfig(max_workers=2, timeout=300))
    df = result.to_pandas()
    return df["faithfulness"].mean(), df["answer_relevancy"].mean(), df


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

def details_path(label, test_file=None, retrieval_only=None):
    """A retrieval-only run gets its own file, so it never overwrites a run WITH answers."""
    stem = Path(test_file or TEST_FILE).stem
    ro = RETRIEVAL_ONLY if retrieval_only is None else retrieval_only
    suffix = "_retrieval-only" if ro and label != "bm25-pages" else ""
    return RESULT_DIR / f"eval_details_{stem}_{label.replace('+', '_')}{suffix}.csv"


DETAIL_COLUMNS = ["id", "type", "question", "expected_file", "expected_pages", "result", "rank",
                  "retrieved (file p.page)", "bot_answer", "expected_answer", "key_facts",
                  "seconds", "faithfulness", "answer_relevance", "top_relevance_score", "old_version_used"]


def keep_old(path):
    """Before a result file is written again, move the earlier version to test_result/old/<date_time>/:
    test_result/ always holds the LATEST results, and nothing is lost."""
    path = Path(path)
    if path.exists():
        OLD_DIR.mkdir(parents=True, exist_ok=True)
        path.replace(OLD_DIR / path.name)


def save_details(label, details, ragas_df=None):
    path = details_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_old(path)
    ragas_rows = iter(ragas_df.to_dict("records")) if ragas_df is not None else None
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DETAIL_COLUMNS)
        for d in details:
            c = d["case"]
            got = " | ".join(f"{doc.metadata.get('source')} p.{doc.metadata.get('page')}" for doc in d["docs"])
            faith = rel = facts = rank = ""
            if c["type"] == "no_answer":
                result = ("not generated" if d["refused"] is None
                          else "refused OK" if d["refused"] else "ANSWERED (wrong)")
            else:
                result = "hit" if d["rank"] else "doc only" if d["doc_rank"] else "miss"
                rank = d["rank"] or ""
                if d["facts"][1]:
                    facts = f"{d['facts'][0]}/{d['facts'][1]}"
                if ragas_rows is not None and d["docs"]:
                    r = next(ragas_rows, {})
                    faith, rel = r.get("faithfulness", ""), r.get("answer_relevancy", "")
            w.writerow([c["id"], c["type"], c["question"], c["file"],
                        ", ".join(map(str, sorted(c["pages"]))), result, rank, got,
                        d["answer"], c["answer"], facts,
                        f"{d['sec_retrieval'] + d['sec_answer']:.1f}", faith, rel,
                        f"{d['score']:.3f}" if d.get("score") is not None else "",
                        "yes" if d.get("old_fallback") else ""])
    return path


SUMMARY_COLUMNS = ["time", "setup", "questions", "hit@1", "hit@3", "hit@k", "mrr@k", "doc_hit@k",
                   "answer_correct", "no_answer_acc", "sec_per_q", "faithfulness", "answer_relevance",
                   "ragas_judge", "answered_on_miss", "old_fallback", "cutoff_advice"]


def save_summary(label, s, settings, faithfulness="", relevance="", advice=""):
    """One row per run, WITH its settings, so runs can be told apart later."""
    header = SUMMARY_COLUMNS + list(settings)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open(encoding="utf-8-sig") as f:
            old_rows = list(csv.DictReader(f))
            f.seek(0)
            old_header = next(csv.reader(f), [])
        if not set(header) <= set(old_header):         # new columns: add them, KEEP every old row
            header = old_header + [c for c in header if c not in old_header]
            with RESULTS_FILE.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=header, restval="")
                w.writeheader()
                w.writerows(old_rows)
            print("(eval_results.csv: new columns added, old rows kept)")
        else:
            header = old_header                        # same columns: keep the file's order
    new_file = not RESULTS_FILE.exists()
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "setup": label,
           "questions": s["answerable"] + s["no_answer"],
           "hit@1": fmt(s.get("hit@1", "")), "hit@3": fmt(s.get("hit@3", "")),
           "hit@k": fmt(s.get("hit_rate", "")), "mrr@k": fmt(s.get("mrr", "")),
           "doc_hit@k": fmt(s.get("doc_hit_rate", "")), "answer_correct": fmt(s.get("answer_correct", "")),
           "no_answer_acc": fmt(s.get("no_answer_acc", "")), "sec_per_q": f"{s['sec_per_q']:.1f}",
           "faithfulness": fmt(faithfulness), "answer_relevance": fmt(relevance),
           "ragas_judge": settings["chat_model"] if faithfulness != "" else "",
           "answered_on_miss": s.get("answered_on_miss", ""), "old_fallback": s.get("old_fallback", ""),
           "cutoff_advice": " ".join((advice or "").split()), **settings}
    with RESULTS_FILE.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="")
        if new_file:
            w.writeheader()
        w.writerow(row)


def print_scores(label, s, settings, faithfulness=""):
    print("\n" + "=" * 60)
    print(f"  Setup                  : {label}")
    print(f"  Scored at k            : {EVAL_K} (the chatbot answers from {app.TOP_K})")
    if "hit_rate" in s:
        print(f"  Hit@1 / Hit@3 (page)   : {s['hit@1']:.3f} / {s['hit@3']:.3f}")
        print(f"  Hit@{EVAL_K} (page)           : {s['hit_rate']:.3f}   ({s['answerable']} questions)")
        print(f"  MRR@{EVAL_K}                  : {s['mrr']:.3f}")
        print(f"  Right document @{EVAL_K}      : {s['doc_hit_rate']:.3f}")
        if "answer_correct" in s:
            print(f"  Answer key facts found : {s['answer_correct']:.3f}")
        for t, (hits, total) in s["by_type"].items():
            print(f"     {t:<20}: {hits}/{total}")
    if "no_answer_acc" in s:
        print(f"  No-answer refused      : {s['no_answer_acc']:.3f}   ({s['no_answer']} questions)")
    if "answered_on_miss" in s:
        print(f"  Answered a miss        : {s['answered_on_miss']}   (wrong passages, answered anyway)")
        print(f"  Old-version fallback   : {s['old_fallback']} questions")
    print(f"  Seconds per question   : {s['sec_per_q']:.1f}")
    if faithfulness != "":
        print(f"  (RAGAS graded by {settings['chat_model']}, the same model: treat as rough)")


# ---------------------------------------------------------------------------
# 7. Runs
# ---------------------------------------------------------------------------

RESUME = "--resume" in sys.argv


def finished_run(label, settings):
    """--resume: the saved row of a run that already finished with exactly these settings, or None.
    A run counts only when its details file exists too (written at the very end of a run)."""
    if not RESULTS_FILE.exists() or not details_path(label).exists():
        return None
    keys = ("test_file", "test_file_md5", "retrieval_only", "chunks_in_db", "db_version", "eval_k",
            "chat_model", "embed_model", "min_relevance")
    with RESULTS_FILE.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("setup") == label
                and all(str(r.get(k, "")) == str(settings[k]) for k in keys)]
    if "--ragas" in sys.argv:
        rows = [r for r in rows if r.get("faithfulness")]
    return rows[-1] if rows else None


def scores_from_row(row):
    num = lambda k: float(row[k]) if row.get(k) else 0.0
    return {"hit@1": num("hit@1"), "hit@3": num("hit@3"), "hit_rate": num("hit@k"), "mrr": num("mrr@k"),
            "doc_hit_rate": num("doc_hit@k"), "sec_per_q": num("sec_per_q")}


def run_one(cases, setup=None):
    setup = setup or current_setup()
    use_setup(setup)
    label = label_of(setup)
    print(f"\nSetup: {label}   ({TEST_FILE.name})\n")
    vector_db = app.get_vector_db(strategy=setup["strategy"])
    settings = run_settings(setup, vector_db)
    done = finished_run(label, settings) if RESUME else None
    if done:
        print(f"  --resume: already finished on {done['time']} with the same settings: skipped")
        return label, scores_from_row(done), details_path(label)
    details = run_questions(vector_db, cases)
    s = score(details)
    faithfulness = relevance = ""
    ragas_df = None
    if "--ragas" in sys.argv and not RETRIEVAL_ONLY:
        print("\nRAGAS: grading answers (slow)...")
        faithfulness, relevance, ragas_df = score_ragas(details)
    print_scores(label, s, settings, faithfulness)
    advice = suggest_cutoff(details)
    if advice:
        print(advice)
    print("=" * 60)
    save_summary(label, s, settings, faithfulness, relevance, advice)
    path = save_details(label, details, ragas_df)
    print(f"Saved: {RESULTS_FILE} and {path}")
    return label, s, path


def setups_for_all():
    """--all: the 3 strategies with the flags given. --all --setups: each strategy x
    (base, +hybrid, +hybrid+rerank, +hybrid+rerank+rewrite), plus 3 ablation rows (H1 / M5)."""
    full = "--setups" in sys.argv or "--setup" in sys.argv
    base = dict(hybrid=False, rerank=False, rewrite=False, synonyms=False, stemming=True, doc_focus=True)
    if full:
        variants = [{}, {"hybrid": True}, {"hybrid": True, "rerank": True},
                    {"hybrid": True, "rerank": True, "rewrite": True}]
    else:
        cur = current_setup()
        variants = [{k: v for k, v in cur.items() if k != "strategy"}]
    out = [dict(base, strategy=st, **v) for st in ("recursive", "fixed", "semantic") for v in variants]
    if full:
        out += [dict(base, strategy="recursive", hybrid=True, doc_focus=False),     # DOC_FOCUS?
                dict(base, strategy="recursive", hybrid=True, stemming=False),      # stemming?
                dict(base, strategy="recursive", hybrid=True, synonyms=True)]       # synonyms?
    return out


def run_all(cases):
    runs, rows = [], []
    for setup in setups_for_all():
        if not app.is_built(setup["strategy"]):
            print(f"\n(skipped {setup['strategy']}: database not built. "
                  f"Run: python app.py --strategy={setup['strategy']})")
            continue
        label, s, path = run_one(cases, setup)
        runs.append((label, s))
        with path.open(encoding="utf-8-sig") as f:
            part = list(csv.reader(f))
        rows += [[label] + r for r in part[1:]]
    if not runs:
        sys.exit("No database is built. Run: python app.py")
    all_file = RESULT_DIR / f"eval_details_{TEST_FILE.stem}_all{'_retrieval-only' if RETRIEVAL_ONLY else ''}.csv"
    keep_old(all_file)
    with all_file.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["setup"] + DETAIL_COLUMNS)
        w.writerows(rows)
    print("\n" + "=" * 84)
    print(f"  {'setup':<36}{'Hit@1':>8}{'Hit@3':>8}{f'Hit@{EVAL_K}':>8}{f'MRR@{EVAL_K}':>8}"
          f"{'Doc@k':>8}{'s/q':>7}")
    for label, s in runs:
        if "hit_rate" in s:
            print(f"  {label:<36}{s['hit@1']:>8.3f}{s['hit@3']:>8.3f}{s['hit_rate']:>8.3f}"
                  f"{s['mrr']:>8.3f}{s['doc_hit_rate']:>8.3f}{s['sec_per_q']:>7.1f}")
    print("=" * 84)
    print(f"Saved: {all_file} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 8. The floor: plain BM25 over whole pages (review "Evaluation integrity")
# ---------------------------------------------------------------------------

PDF_FOLDER = Path("raw_pdfs")


def page_texts():
    """{(file, doc_id, page): text} of every PAGE of every PDF in raw_pdfs/, read straight from
    the PDF (the reviewer's method). Annual reports are left out, as in the index."""
    import pymupdf
    skip, doc_ids = set(), {}
    for f in app.JSON_FOLDER.glob("*.json"):           # doc_type + doc_id come from clean_pdfs.py
        data = json.loads(f.read_text(encoding="utf-8"))
        doc_ids[data["file_name"]] = (data.get("file_hash") or "")[:12]
        if data.get("metadata", {}).get("doc_type") == "annual_report" and not app.INCLUDE_ANNUAL_REPORTS:
            skip.add(data["file_name"])
    expected = set(doc_ids) - skip
    missing = sorted(n for n in expected if not (PDF_FOLDER / n).exists())
    if missing:                                         # review N1: never publish a floor on a few PDFs
        sys.exit(f"--bm25-pages: {len(missing)} of {len(expected)} PDFs that clean_pdfs.py indexed are not "
                 f"in {PDF_FOLDER}/, e.g. {missing[:3]}.\nPut ALL PDFs back in {PDF_FOLDER}/ and run again.")
    pages = {}
    for pdf in sorted(PDF_FOLDER.glob("*.pdf")):
        if pdf.name in skip or pdf.name not in doc_ids:
            continue
        with pymupdf.open(str(pdf)) as doc:
            for n, page in enumerate(doc, start=1):
                pages[(pdf.name, doc_ids[pdf.name], n)] = page.get_text()
    if not pages:
        sys.exit("No PDF pages found: --bm25-pages needs raw_pdfs/ and output/json/ (run clean_pdfs.py).")
    print(f"BM25 floor: {len(pages)} pages from {len(expected)} PDFs")
    return pages


def run_bm25_pages(cases):
    """Plain BM25: lower-case words, no stemming, no synonyms, no chunking, no version filter."""
    from rank_bm25 import BM25Okapi
    pages = page_texts()
    keys = list(pages)
    plain = lambda t: re.findall(r"\w+", t.lower())
    bm25 = BM25Okapi([plain(pages[k]) for k in keys])
    details = []
    for case in cases:
        t0 = time.perf_counter()
        scores = bm25.get_scores(plain(case["question"]))
        top = sorted(range(len(keys)), key=scores.__getitem__, reverse=True)[:EVAL_K]
        metas = [{"source": keys[i][0], "doc_id": keys[i][1], "page": keys[i][2], "page_end": keys[i][2]}
                 for i in top]
        row = {"case": case, "docs": [type("D", (), {"metadata": m, "page_content": ""}) for m in metas],
               "used": [], "answer": "", "sec_retrieval": time.perf_counter() - t0, "sec_answer": 0.0,
               "score": None, "facts": (0, 0)}
        if case["type"] == "no_answer":
            row["refused"] = None
        else:
            row["rank"] = next((i for i, m in enumerate(metas, start=1) if is_hit(m, case)), None)
            row["doc_rank"] = next((i for i, m in enumerate(metas, start=1) if same_doc(m, case)), None)
        details.append(row)
    s = score(details)
    s.pop("no_answer_acc", None)
    s.pop("answer_correct", None)
    s.pop("answered_on_miss", None)
    s.pop("old_fallback", None)
    label = "bm25-pages"
    settings = run_settings({"strategy": "pages", "hybrid": False, "rerank": False, "rewrite": False,
                             "synonyms": False, "stemming": False, "doc_focus": False},
                            chunks=len(keys))
    settings["retrieval_only"] = True
    print_scores(label + "  (plain BM25 over whole pages: the floor to beat)", s, settings)
    save_summary(label, s, settings)
    print(f"Saved: {RESULTS_FILE} and {save_details(label, details)}")


# ---------------------------------------------------------------------------
# 9. Report: eval_report.md + README table (review Day 4 / Day 5 evidence, M8)
# ---------------------------------------------------------------------------

def _results_by_test():
    if not RESULTS_FILE.exists():
        sys.exit("No eval_results.csv yet: run eval.py first.")
    with RESULTS_FILE.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or "answer_correct" not in rows[0]:
        sys.exit("eval_results.csv is in an old format: run the evals again with this eval.py.")
    latest = {}
    for r in rows:                                     # the last run of each setup wins
        # a retrieval-only run and a run with answers are DIFFERENT rows (one never hides the other)
        latest[(r["test_file"], r["setup"], r["eval_k"], r.get("retrieval_only", ""))] = r
    by_test = {}
    for (test, _, _, _), r in latest.items():
        by_test.setdefault(test, []).append(r)
    for test, group in by_test.items():                # review N1: never mix two versions of a test
        now = test_md5(test) or group[-1]["test_file_md5"]     # set in one table
        old = [r for r in group if r["test_file_md5"] != now]
        if old:
            print(f"(report: {len(old)} runs on an OLDER version of {test} are left out: "
                  f"{', '.join(sorted({r['setup'] for r in old}))}. Run them again.)")
            by_test[test] = [r for r in group if r["test_file_md5"] == now]
    return {t: g for t, g in by_test.items() if g}


def _baseline(r, group):
    if r["setup"] == "bm25-pages":
        return None
    for b in group:
        if (b["strategy"] == r["strategy"] and b["eval_k"] == r["eval_k"]
                and b["hybrid"] == b["rerank"] == b["rewrite"] == "False"
                and b["synonyms"] == "False" and b["stemming"] == b["doc_focus"] == "True"
                and b["min_relevance"] in ("0", "0.0") and b["retrieval_only"] == r["retrieval_only"]):
            return b
    return None


def _wins_losses(r, base):
    """Questions that this setup finds (hit@k) and the baseline does not, and the other way round."""
    def hits(label):
        path = details_path(label, r["test_file"], r.get("retrieval_only") == "True")
        if not path.exists():
            return None
        with path.open(encoding="utf-8-sig") as f:
            return {row["id"]: row["result"] == "hit" for row in csv.DictReader(f)
                    if row["type"] != "no_answer"}
    a, b = hits(r["setup"]), hits(base["setup"])
    if a is None or b is None:
        return ""
    won = [q for q in a if a[q] and not b.get(q)]
    lost = [q for q in a if not a[q] and b.get(q)]
    ids = lambda qs: " ".join(qs[:8]) + (f" (+{len(qs) - 8} more)" if len(qs) > 8 else "")
    return f"+{len(won)} {ids(won)} / -{len(lost)} {ids(lost)}".strip()


def write_report():
    by_test = _results_by_test()
    cols = ("hit@1", "hit@3", "hit@k", "mrr@k", "doc_hit@k", "answer_correct", "no_answer_acc",
            "answered_on_miss", "faithfulness", "answer_relevance", "sec_per_q")
    lines = ["# Evaluation results", "",
             "Every setup is scored at the same k on the same questions. In brackets: change against the",
             "baseline of the same chunking strategy (vectors only, default settings). `bm25-pages` is plain",
             "BM25 over whole pages (no chunking, no tuning): the floor every setup must beat.",
             "`-docfocus`, `-stemming` and `+synonyms` rows are the ablation for review items H1 / M5.",
             "**qa_heldout.csv was never used to tune anything: judge the extras on it.**", ""]
    readme_rows = []
    for test, group in sorted(by_test.items(), key=lambda kv: kv[0] != "qa_test_set.csv"):
        k = group[0]["eval_k"]
        lines += [f"## {test} (k = {k}, md5 {group[0]['test_file_md5']}, {group[0]['questions']} questions)", "",
                  "| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused "
                  "| Answered a miss | Faithfulness | Answer relevance | s/question "
                  "| Won / lost vs baseline (hit@k) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        order = sorted(group, key=lambda r: (r["setup"] != "bm25-pages", r.get("retrieval_only") != "True",
                                             r["strategy"], r["setup"]))
        for r in order:
            b = _baseline(r, group)
            def cell(c):
                v = r.get(c, "")
                if not v or b is None or b is r or not b.get(c) or c in ("sec_per_q", "answered_on_miss"):
                    return v
                return f"{v} ({float(v) - float(b[c]):+.3f})"
            wl = _wins_losses(r, b) if b is not None and b is not r else ""
            name = f"`{r['setup']}`" + (" (retrieval only)" if r.get("retrieval_only") == "True"
                                         and r["setup"] != "bm25-pages" else "")
            lines.append("| " + " | ".join([name] + [cell(c) for c in cols] + [wl]) + " |")
            readme_rows.append(f"| {test} | {name} | {r.get('hit@1', '')} | {r.get('hit@3', '')} | "
                               f"{r.get('hit@k', '')} | {r.get('mrr@k', '')} | {r.get('answer_correct', '')} | "
                               f"{r.get('no_answer_acc', '')} | {r.get('answered_on_miss', '')} | "
                               f"{r.get('faithfulness', '')} | {r.get('answer_relevance', '')} |")
        lines.append("")
        notes = []
        for r in order:                                 # review N8: the cut-off evidence, recorded
            if r.get("cutoff_advice"):
                notes.append(f"- Cut-off (`{r['setup']}`): {r['cutoff_advice']}")
            if r.get("old_fallback") not in (None, "", "0"):
                notes.append(f"- Old-version fallback (`{r['setup']}`): used for {r['old_fallback']} questions")
            g = _hand_grades(test, r)                   # review N6
            if g:
                notes.append(f"- Hand-graded answers without numbers (`{r['setup']}`): {g}")
        if notes:
            lines += notes + [""]
    lines += ["Key facts = share of the numbers (also written as words), amounts, dates and e-mail addresses",
              "of the expected answer that appear in the bot's answer (review M7, N6). Answered a miss =",
              "answerable questions where retrieval missed the right document and the bot answered anyway",
              "(review N3: should be 0). With 40 answerable questions one question is 2.5",
              "points: read the won / lost columns, not only the averages (review M8).",
              "Faithfulness / answer relevance = RAGAS (python eval.py ... --ragas), graded by the same",
              "local model (qwen3:8b), so treat them as rough.", "",
              "All settings of every run (models, chunk size, cut-off, test file md5) are in eval_results.csv."]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved: {REPORT_FILE}")
    _update_readme(readme_rows)


def _update_readme(rows):
    """Replace the table between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md."""
    if not README.exists():
        return
    text = README.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start not in text or end not in text:
        print("(README.md has no RESULTS markers: table not updated)")
        return
    table = "\n".join(["| Test set | Setup | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Key facts | No-answer refused "
                        "| Answered a miss | Faithfulness | Answer relevance |",
                        "|---|---|---|---|---|---|---|---|---|---|---|"] + rows)
    stamp = f"_Filled by `python eval.py --report` on {datetime.now():%Y-%m-%d %H:%M}._"
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    README.write_text(f"{before}{start}\n{stamp}\n\n{table}\n{end}{after}", encoding="utf-8")
    print("README.md results table updated.")


# ---------------------------------------------------------------------------
# 10. Hand grading of answers that have no numbers (review N6)
# ---------------------------------------------------------------------------

def grades_path(test_file, label):
    return RESULT_DIR / f"manual_grades_{Path(test_file).stem}_{label.replace('+', '_')}.csv"


def grade():
    """Shows every answerable question whose expected answer has no key facts, with the bot's
    answer, and asks: correct (y) / partly (p) / wrong (n). Saved to manual_grades_*.csv."""
    label = label_of(current_setup())
    path = details_path(label, retrieval_only=False)
    if not path.exists():
        sys.exit(f"No {path}: run this setup WITH answers first (without --retrieval-only).")
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["type"] != "no_answer" and not r["key_facts"]]
    out, done = grades_path(TEST_FILE, label), {}
    if out.exists():
        with out.open(encoding="utf-8-sig") as f:
            done = {r["id"]: r["grade"] for r in csv.DictReader(f)}
    print(f"{len(rows)} answers without numbers in {path} ({len(done)} graded before)\n")
    for r in rows:
        if r["id"] in done:
            continue
        print("=" * 80)
        print(f"{r['id']}  {r['question']}\n\nEXPECTED: {r['expected_answer']}\n\nBOT:      {r['bot_answer']}\n")
        g = ""
        while g not in ("y", "p", "n"):
            g = input("Correct? y = yes, p = partly, n = no: ").strip().lower()
        done[r["id"]] = g
        with out.open("w", newline="", encoding="utf-8-sig") as f:   # saved after every answer
            w = csv.writer(f)
            w.writerow(["id", "grade"])
            w.writerows(sorted(done.items()))
    print(f"\nSaved: {out}. Run python eval.py --report to show it in the report.")


def _hand_grades(test, r):
    path = grades_path(test, r["setup"])
    if r.get("retrieval_only") == "True" or not path.exists():
        return ""
    with path.open(encoding="utf-8-sig") as f:
        g = [row["grade"] for row in csv.DictReader(f)]
    return f"{g.count('y')}/{len(g)} correct, {g.count('p')} partly, {g.count('n')} wrong"


def print_doc_ids():
    data = app.get_vector_db().get(include=["metadatas"])
    ids = {m.get("source", "?"): m.get("doc_id", "") or "(rebuild: python app.py --rebuild)"
           for m in data["metadatas"]}
    print(f"{'doc_id':<14} file")
    for source, doc_id in sorted(ids.items()):
        print(f"{doc_id:<14} {source}")


def move_old_results():
    """No result CSV stays next to eval.py: each one moves into test_result/, or, when
    test_result/ already has a file of that name, into test_result/old/<date_time>/."""
    RESULT_DIR.mkdir(exist_ok=True)
    moved = 0
    for pattern in ("eval_results.csv", "eval_results_old_*.csv", "eval_details_*.csv", "manual_grades_*.csv"):
        for f in Path(".").glob(pattern):
            target = RESULT_DIR / f.name
            if target.exists():
                OLD_DIR.mkdir(parents=True, exist_ok=True)
                target = OLD_DIR / f.name
            f.replace(target)
            moved += 1
    if moved:
        print(f"(moved {moved} result files from the project folder into {RESULT_DIR}/)")


if __name__ == "__main__":
    move_old_results()
    if "--report" in sys.argv:
        write_report()
    elif "--ids" in sys.argv:
        print_doc_ids()
    elif "--grade" in sys.argv:
        grade()
    elif "--bm25-pages" in sys.argv:
        run_bm25_pages(load_cases())
    elif "--all" in sys.argv:
        run_all(load_cases())
    else:
        run_one(load_cases())