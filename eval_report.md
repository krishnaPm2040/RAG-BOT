# Evaluation results

Every setup is scored at the same k on the same questions. In brackets: change against the
baseline of the same chunking strategy (vectors only, default settings). `bm25-pages` is plain
BM25 over whole pages (no chunking, no tuning): the floor every setup must beat.
`-docfocus`, `-stemming` and `+synonyms` rows are the ablation for review items H1 / M5.
**qa_heldout.csv was never used to tune anything: judge the extras on it.**

## qa_test_set.csv (k = 5, md5 83f40f99, 50 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | Answered a miss | Faithfulness | Answer relevance | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.650 | 0.850 | 0.875 | 0.751 | 0.975 |  |  |  |  |  | 0.0 |  |
| `fixed` (retrieval only) | 0.725 | 0.875 | 0.950 | 0.804 | 0.975 |  |  |  |  |  | 0.3 |  |
| `fixed+hybrid` (retrieval only) | 0.775 (+0.050) | 0.925 (+0.050) | 1.000 (+0.050) | 0.852 (+0.048) | 1.000 (+0.025) |  |  |  |  |  | 0.3 | +2 q027 q030 / -0 |
| `fixed+hybrid+rerank` (retrieval only) | 0.875 (+0.150) | 0.975 (+0.100) | 0.975 (+0.025) | 0.925 (+0.121) | 1.000 (+0.025) |  |  |  |  |  | 2.0 | +2 q027 q030 / -1 q019 |
| `fixed+hybrid+rerank+rewrite` (retrieval only) | 0.900 (+0.175) | 0.950 (+0.075) | 0.950 (+0.000) | 0.925 (+0.121) | 1.000 (+0.025) |  |  |  |  |  | 9.8 | +2 q027 q030 / -2 q019 q033 |
| `recursive` (retrieval only) | 0.775 | 0.925 | 0.950 | 0.847 | 1.000 |  |  |  |  |  | 0.3 |  |
| `recursive+hybrid` (retrieval only) | 0.800 (+0.025) | 0.950 (+0.025) | 0.975 (+0.025) | 0.873 (+0.026) | 1.000 (+0.000) |  |  |  |  |  | 0.2 | +1 q027 / -0 |
| `recursive+hybrid+rerank` (retrieval only) | 0.850 (+0.075) | 1.000 (+0.075) | 1.000 (+0.050) | 0.917 (+0.070) | 1.000 (+0.000) |  |  |  |  |  | 1.9 | +2 q012 q027 / -0 |
| `recursive+hybrid+rerank+rewrite` (retrieval only) | 0.875 (+0.100) | 1.000 (+0.075) | 1.000 (+0.050) | 0.925 (+0.078) | 1.000 (+0.000) |  |  |  |  |  | 20.9 | +2 q012 q027 / -0 |
| `recursive+hybrid+synonyms` (retrieval only) | 0.850 (+0.075) | 0.975 (+0.050) | 1.000 (+0.050) | 0.906 (+0.059) | 1.000 (+0.000) |  |  |  |  |  | 0.3 | +2 q012 q027 / -0 |
| `recursive+hybrid-docfocus` (retrieval only) | 0.825 (+0.050) | 1.000 (+0.075) | 1.000 (+0.050) | 0.904 (+0.057) | 1.000 (+0.000) |  |  |  |  |  | 0.3 | +2 q012 q027 / -0 |
| `recursive+hybrid-stemming` (retrieval only) | 0.825 (+0.050) | 0.950 (+0.025) | 0.950 (+0.000) | 0.887 (+0.040) | 1.000 (+0.000) |  |  |  |  |  | 0.3 | +1 q027 / -1 q015 |
| `semantic` (retrieval only) | 0.725 | 0.825 | 0.875 | 0.783 | 0.950 |  |  |  |  |  | 0.2 |  |
| `semantic+hybrid` (retrieval only) | 0.725 (+0.000) | 0.900 (+0.075) | 0.975 (+0.100) | 0.827 (+0.044) | 1.000 (+0.050) |  |  |  |  |  | 0.2 | +4 q018 q019 q030 q032 / -0 |
| `semantic+hybrid+rerank` (retrieval only) | 0.850 (+0.125) | 0.950 (+0.125) | 0.950 (+0.075) | 0.896 (+0.113) | 1.000 (+0.050) |  |  |  |  |  | 2.6 | +4 q018 q027 q030 q032 / -1 q035 |
| `semantic+hybrid+rerank+rewrite` (retrieval only) | 0.850 (+0.125) | 0.950 (+0.125) | 0.950 (+0.075) | 0.892 (+0.109) | 1.000 (+0.050) |  |  |  |  |  | 10.8 | +4 q018 q027 q030 q032 / -1 q033 |

- Cut-off (`fixed`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`fixed+hybrid`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`fixed+hybrid+rerank`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`fixed+hybrid+rerank+rewrite`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid+rerank`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid+rerank+rewrite`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid+synonyms`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid-docfocus`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid-stemming`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.81, real min 0.79). Keep 0 and let the model refuse.
- Cut-off (`semantic`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.82, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`semantic+hybrid`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.82, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`semantic+hybrid+rerank`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.82, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`semantic+hybrid+rerank+rewrite`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.82, real min 0.76). Keep 0 and let the model refuse.

## qa_heldout.csv (k = 5, md5 aaf5860c, 20 questions)

| Setup | Hit@1 | Hit@3 | Hit@k | MRR@k | Doc hit@k | Key facts | No-answer refused | Answered a miss | Faithfulness | Answer relevance | s/question | Won / lost vs baseline (hit@k) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bm25-pages` | 0.438 | 0.812 | 0.875 | 0.630 | 1.000 |  |  |  |  |  | 0.0 |  |
| `fixed` (retrieval only) | 0.688 | 0.812 | 0.812 | 0.750 | 1.000 |  |  |  |  |  | 0.4 |  |
| `fixed+hybrid` (retrieval only) | 0.750 (+0.062) | 0.875 (+0.063) | 0.875 (+0.063) | 0.802 (+0.052) | 1.000 (+0.000) |  |  |  |  |  | 0.4 | +1 h12 / -0 |
| `fixed+hybrid+rerank` (retrieval only) | 0.688 (+0.000) | 0.875 (+0.063) | 0.875 (+0.063) | 0.771 (+0.021) | 1.000 (+0.000) |  |  |  |  |  | 2.1 | +1 h15 / -0 |
| `recursive` (retrieval only) | 0.625 | 0.875 | 0.875 | 0.740 | 1.000 |  |  |  |  |  | 0.4 |  |
| `recursive+hybrid` (retrieval only) | 0.625 (+0.000) | 0.812 (-0.063) | 0.812 (-0.063) | 0.719 (-0.021) | 1.000 (+0.000) |  |  |  |  |  | 0.4 | +0  / -1 h03 |
| `recursive+hybrid+rerank` (retrieval only) | 0.625 (+0.000) | 0.938 (+0.063) | 0.938 (+0.063) | 0.760 (+0.020) | 1.000 (+0.000) |  |  |  |  |  | 2.9 | +1 h15 / -0 |
| `recursive+hybrid+rerank+rewrite` (retrieval only) | 0.625 (+0.000) | 0.750 (-0.125) | 0.812 (-0.063) | 0.703 (-0.037) | 1.000 (+0.000) |  |  |  |  |  | 10.6 | +1 h12 / -2 h03 h05 |

- Cut-off (`fixed`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`fixed+hybrid`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`fixed+hybrid+rerank`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`recursive`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid+rerank`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.
- Cut-off (`recursive+hybrid+rerank+rewrite`): MIN_RELEVANCE advice : no clean cut on the dev half (no_answer max 0.79, real min 0.76). Keep 0 and let the model refuse.

Key facts = share of the numbers (also written as words), amounts, dates and e-mail addresses
of the expected answer that appear in the bot's answer (review M7, N6). Answered a miss =
answerable questions where retrieval missed the right document and the bot answered anyway
(review N3: should be 0). With 40 answerable questions one question is 2.5
points: read the won / lost columns, not only the averages (review M8).
Faithfulness / answer relevance = RAGAS (python eval.py ... --ragas), graded by the same
local model (qwen3:8b), so treat them as rough.

All settings of every run (models, chunk size, cut-off, test file md5) are in eval_results.csv.
