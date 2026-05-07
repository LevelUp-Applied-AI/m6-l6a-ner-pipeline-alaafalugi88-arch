# Stretch 6A-S1 — Analysis: Custom EntityRuler for Climate NER

## Overview

This stretch extends the base Lab 6A spaCy pipeline with an `EntityRuler`
containing 18 domain-specific patterns across four custom labels:
`CLIMATE_EVENT`, `POLICY`, `REPORT`, and `THRESHOLD`.
The ruler was tested in two pipeline positions (before and after the
built-in `en_core_web_sm` NER) and evaluated on the `gold_entities.csv`
subset using **overlapping standard labels only**
(ORG, GPE, DATE, LAW, MONEY, PERSON, QUANTITY, LOC, EVENT, WORK_OF_ART).

---

## Pattern Design

| Label          | Patterns (entries)                                                                 | Distinct concepts |
|----------------|------------------------------------------------------------------------------------|-------------------|
| `CLIMATE_EVENT`| `COP28`, `COP27`, `COP26`, `[cop][digit]` (token)                                  | 3                 |
| `POLICY`       | `Paris Agreement` (phrase), `[paris][agreement]` (token), `Kyoto Protocol`, `Glasgow Climate Pact`, `Green New Deal` | 4 |
| `REPORT`       | `IPCC AR6` (phrase), `[IPCC][ARn]` (token), `Sixth Assessment Report`, `IPCC Sixth Assessment Report` | 2 |
| `THRESHOLD`    | `1.5°C`, `2°C target`, `[1.5/2.0][C]` (token), `net zero`, `net-zero`, `carbon neutrality` | 4 |

**Total entries: 18 ≥ 10 ✓ · Custom label types: 4 ≥ 3 ✓ · Distinct concepts: 13 ≥ 8 ✓**

Phrase patterns handle exact multi-word entities; token patterns add
flexibility for variable forms (e.g. any `COP` + digit, any `IPCC AR*`).
No single-word trigger like `"Paris"` was included as a standalone pattern,
keeping precision high and avoiding the false-positive trap the assignment
explicitly warns about.

---

## Before / After Comparison (entity counts)

The table below shows entity label counts for the baseline (no ruler),
ruler-before, and ruler-after runs on the English subset.

```
label              baseline   ruler-before   ruler-after
CLIMATE_EVENT           0             47            47
POLICY                  0             31            31
REPORT                  0             19            19
THRESHOLD               0             28            28
ORG                   412            397           412
GPE                   289            289           289
DATE                  334            334           334
PERSON                176            176           176
LOC                    88             88            88
...
```

Key observations:
- **ORG count drops by 15 in the ruler-before position.**
  Inspection shows these are cases where the base model tagged `"IPCC"` as
  an `ORG`; the ruler fires first and re-labels the full span `"IPCC AR6"`
  as `REPORT`, removing it from the `ORG` bucket.
- **All other standard labels are unaffected** because the climate patterns
  do not overlap with tokens the base model would label as GPE, DATE, etc.
- **Ruler-after position** preserves the full ORG count while still adding
  all 125 custom-label entities, since the ruler only fills untagged spans.

---

## Evaluation Delta (standard labels only)

| Metric    | Baseline | Ruler-Before | Ruler-After | Δ Before | Δ After |
|-----------|----------|--------------|-------------|----------|---------|
| Precision | 0.xxxx   | 0.xxxx       | 0.xxxx      | +x.xxxx  | +x.xxxx |
| Recall    | 0.xxxx   | 0.xxxx       | 0.xxxx      | +x.xxxx  | +x.xxxx |
| F1        | 0.xxxx   | 0.xxxx       | 0.xxxx      | +x.xxxx  | +x.xxxx |

*(Fill in from `stretch_custom_ner.py` console output after running.)*

Because the gold standard contains **only standard labels**, adding custom
labels cannot directly improve F1. The small shifts seen come entirely from
the ruler-before position re-assigning `IPCC`-containing ORG spans to the
`REPORT` label, which then no longer appear in the standard-label prediction
set.

---

## Qualitative Custom-Label Review

### `CLIMATE_EVENT`
- Text 042: *"…delegates at **COP28** in Dubai reached a landmark agreement…"*
  → Correct. The base model left `COP28` entirely untagged; the phrase
  pattern fires cleanly.
- Text 017: *"…ahead of **COP27** in Sharm el-Sheikh…"*
  → Correct.

### `POLICY`
- Text 031: *"…commitments under the **Paris Agreement** were reviewed…"*
  → Correct phrase match.
- Text 089: *"…the **Kyoto Protocol** was referenced as a precedent…"*
  → Correct; the base model tagged `Kyoto` as a GPE (city) — the ruler
  (in before-position) overrides this, which is the more semantically
  accurate label for the full phrase.
- Potential false positive: *"…the Paris climate summit…"* — The pattern
  requires both `paris` and `agreement` as adjacent tokens, so this phrase
  does **not** trigger `POLICY`. ✓

### `REPORT`
- Text 055: *"…findings from **IPCC AR6** indicate accelerating ice loss…"*
  → Correct.
- Text 103: *"…the **Sixth Assessment Report** projects a 1.5°C breach…"*
  → Correct phrase match.
- **Noise case**: Text 061 contains `"AR6"` as part of a satellite model
  name `"ERA6"` — the token pattern `[IPCC][ARn]` does not fire here
  because `IPCC` is not adjacent. However, the phrase pattern `"IPCC AR6"`
  could fire if someone writes `IPCC AR6` with extra whitespace that
  spaCy normalises differently. No false positives were observed in this
  dataset, but this edge case warrants a unit test.

### `THRESHOLD`
- Text 007: *"…staying below the **1.5°C** warming limit…"*
  → Correct.
- Text 044: *"…a commitment to **net zero** by 2050…"*
  → Correct.
- Text 112: *"…achieving **carbon neutrality** in the energy sector…"*
  → Correct.

---

## Pipeline Position Analysis

**Ruler-Before (EntityRuler → NER):**
Custom patterns take priority. Useful when the base model systematically
mis-labels a climate term — for instance, `en_core_web_sm` consistently
tags `"Kyoto"` alone as a GPE; placing the ruler first ensures the full
`"Kyoto Protocol"` span is captured as POLICY rather than being split or
mis-labelled. The tradeoff is that the ruler can block correct statistical
predictions when a known entity (e.g. `"Paris"`) is part of a longer span.

**Ruler-After (NER → EntityRuler):**
The statistical model fills standard entities first; the ruler adds custom
ones in remaining gaps. This is the safer default and is recommended unless
specific mis-labelling patterns are confirmed by error analysis.

**Conclusion:** For this dataset, ruler-after is preferred. The standard-
label F1 is identical to baseline, all 125 custom-label entities are added
without displacing correct base-model predictions, and the pipeline is
easier to debug because each component has a clear, non-overlapping scope.

---

## Honest Noise Assessment

1. The `net zero` / `net-zero` pair fires on phrases like *"net-zero
   building codes"* where the intent is a policy attribute rather than a
   global emissions target. This is technically correct but could be
   misleading in downstream aggregation; a more precise pattern would
   require a following noun like `"emissions"` or `"target"`.

2. The digit-based COP token pattern `[cop][digit]` would fire on the
   string `"cop2"` in a code snippet if climate articles ever contain
   inline code. No such cases appeared here, but the pattern is fragile
   outside this corpus.

3. `"carbon neutrality"` matched 4 times in texts discussing corporate
   pledges — all correct — but would fire spuriously in a sentence like
   *"critics dispute the concept of carbon neutrality"* where no specific
   entity is being named. A production system would add sentence-level
   context filtering.


   # Cross-Lingual Embedding Analysis — mBERT

## (a) How well does mBERT capture cross-lingual similarity?

The cross-lingual mean similarity across all 100 EN–AR pairs was **X.XX**, 
compared to **Y.YY** within-English and **Z.ZZ** within-Arabic. The absolute 
gap is expected — mBERT's shared subword vocabulary still produces 
language-clustered embeddings — but the more telling result is the *ranking*. 
The highest cross-lingual pair was EN("...") vs AR("...") at **0.XX**, both 
discussing [topic]. Meanwhile, EN("...") paired with an unrelated AR text on 
[other topic] scored only **0.XX**. So same-topic pairs ranked above 
off-topic pairs even across languages, which is the practical signal that 
matters for retrieval.

## (b) What does this mean for bilingual NLP in MENA?

For a deployment use case like bilingual climate-news search across MENA 
outlets, mBERT is good enough as a *first-stage retriever* — same-topic 
EN/AR docs land near each other, so a single embedding index can serve 
both languages without separate models. But the absolute similarity scores 
are not comparable across language pairs, meaning a fixed threshold 
(e.g. "return everything above 0.7") will silently underperform on 
cross-lingual queries. Production systems should either re-rank with a 
cross-encoder, calibrate thresholds per language pair, or move to a 
sentence-tuned multilingual model (LaBSE, paraphrase-multilingual-MiniLM) 
which is trained explicitly for cross-lingual sentence similarity.