"""
# Lab 6A — NER Pipeline (spaCy vs Hugging Face)
Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""
import unicodedata
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import seaborn as sns


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    # TODO: Load the CSV and return the DataFrame
    cv = pd.read_csv(filepath)
    return cv


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # TODO: Compute shape, language/category value_counts, and word-count
    #       statistics on df['text']
    text_lengths = df['text'].apply(lambda x: len(str(x).split()))
    return {
        'shape': df.shape,
        'lang_counts': df['language'].value_counts().to_dict(),
        'category_counts': df['category'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': text_lengths.mean(),
            'min': text_lengths.min(),
            'max': text_lengths.max()
        },
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # TODO: NFC-normalize the text, run it through nlp(), drop
    #       punctuation/whitespace tokens, return lowercased lemmas
    text = unicodedata.normalize("NFC", text)
    doc = nlp(text)
    tokens =[]
    for token in doc:
        if not token.is_punct and not token.is_space:
            tokens.append(token.lemma_.lower())
    return tokens


def extract_spacy_entities(df, nlp):
    rows = []
    english_df = df[df["language"] == "en"]

    VALID_LABELS = {"PERSON", "ORG", "GPE", "DATE", "EVENT"}

    for _, r in english_df.iterrows():
        text_id = r["id"]
        text = str(r["text"]).strip()

        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ not in VALID_LABELS:
                continue

            rows.append({
                "text_id": text_id,
                "entity_text": ent.text.strip(),
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    return pd.DataFrame(rows)








def merge_hr_entities(raw_entities):
    merged = []
    current_entity = None

    for ent in raw_entities:
       
        label = ent.get('entity_group') or ent.get('entity')

        if label is None:
            continue

        if current_entity is None:
            current_entity = {
                'entity_text': ent['word'],
                'entity_label': label,
                'start_char': ent['start'],
                'end_char': ent['end']
            }
        else:
          
            if label == current_entity['entity_label']:
                current_entity['entity_text'] += ' ' + ent['word']
                current_entity['end_char'] = ent['end']
            else:
                merged.append(current_entity)
                current_entity = {
                    'entity_text': ent['word'],
                    'entity_label': label,
                    'start_char': ent['start'],
                    'end_char': ent['end']
                }

    
    if current_entity is not None:
        merged.append(current_entity)

    return merged

def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # TODO: Filter df to English rows, run each text through
    #       ner_pipeline, merge ## subword tokens, strip B-/I- prefix
    #       from labels (IOB format), return as a DataFrame
    
    rows = []
    english_df = df[df["language"] == "en"]

    for _, row in english_df.iterrows():
        text_id = row["id"]
        text = row["text"]

        raw_entities = ner_pipeline(text)
        merged_entities = merge_hr_entities(raw_entities)

        for ent in merged_entities:
            label = ent['entity_label']

            if label.startswith("B-") or label.startswith("I-"):
                label = label[2:]

            rows.append({
                "text_id": text_id,
                "entity_text": ent['entity_text'],
                "entity_label": label,
                "start_char": ent['start_char'],
                "end_char": ent['end_char']
            })

    return pd.DataFrame(rows)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # TODO: Count entities per label for each system, compute totals,
    #       and derive the three overlap sets by matching on
    #       (text_id, entity_text)
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    hf_counts = hf_df['entity_label'].value_counts().to_dict()
    total_spacy = len(spacy_df)
    total_hf = len(hf_df)  
    spacy_set = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_set = set(zip(hf_df['text_id'], hf_df['entity_text']))
    both = spacy_set.intersection(hf_set)
    spacy_only = spacy_set.difference(hf_set)
    hf_only = hf_set.difference(spacy_set)
    return {
        'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': total_spacy,
        'total_hf': total_hf,
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only,
    }   


def normalize_text(text):
    return str(text).lower().strip()


def evaluate_ner(predicted_df, gold_df):
    predicted_set = set(
        (tid, normalize_text(text), label)
        for tid, text, label in zip(
            predicted_df['text_id'],
            predicted_df['entity_text'],
            predicted_df['entity_label']
        )
    )

    gold_set = set(
        (tid, normalize_text(text), label)
        for tid, text, label in zip(
            gold_df['text_id'],
            gold_df['entity_text'],
            gold_df['entity_label']
        )
    )

    true_positives = predicted_set.intersection(gold_set)

    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    recall = len(true_positives) / len(gold_set) if gold_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def entity_counts_by_category(df, entities_df):
    merged = df.merge(entities_df, left_on="id", right_on="text_id")

    counts = (
        merged.groupby(["category", "entity_label"])
        .size()
        .unstack(fill_value=0)
    )

    return counts


def evaluate_per_category(df, entities_df, gold_df):
    merged_pred = df.merge(entities_df, left_on="id", right_on="text_id")
    merged_gold = df.merge(gold_df, left_on="id", right_on="text_id")

    results = {}

    for category in merged_pred["category"].unique():
        pred = merged_pred[merged_pred["category"] == category]
        gold = merged_gold[merged_gold["category"] == category]

        metrics = evaluate_ner(pred, gold)
        results[category] = metrics

    return results


def entity_cooccurrence(entities_df):
    from itertools import combinations
    from collections import Counter

    co = Counter()

    for tid, group in entities_df.groupby("text_id"):
        entities = list(set(group["entity_text"]))

        for pair in combinations(entities, 2):
            co[tuple(sorted(pair))] += 1

    return co

def compute_entity_tfidf(entities_df):
    from collections import Counter
    import math

    docs = entities_df.groupby("text_id")["entity_text"].apply(list)

    tf = {}
    df_count = Counter()

    for tid, ents in docs.items():
        tf[tid] = Counter(ents)
        for e in set(ents):
            df_count[e] += 1

    N = len(docs)

    tfidf = {}

    for tid in tf:
        tfidf[tid] = {}
        for e in tf[tid]:
            tf_val = tf[tid][e]
            idf = math.log(N / (1 + df_count[e]))
            tfidf[tid][e] = tf_val * idf

    return tfidf


def exact_match(pred, gold):
    """Check if predicted and gold entities match exactly (text and label)."""
    return (normalize_text(pred["entity_text"]) == normalize_text(gold["entity_text"]) and
            pred["entity_label"] == gold["entity_label"])


def partial_match(pred, gold):
    """Check if predicted and gold entities overlap in character span."""
    pred_start = pred["start_char"]
    pred_end = pred["end_char"]
    gold_start = gold["start_char"]
    gold_end = gold["end_char"]
    return not (pred_end < gold_start or pred_start > gold_end)


def type_agnostic_match(pred, gold):
    """Check if predicted and gold entities match on text only (ignoring label)."""
    return normalize_text(pred["entity_text"]) == normalize_text(gold["entity_text"])


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, and F1 from confusion matrix values."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }




def evaluate_advanced(pred_df, gold_df, match_type="exact"):
    tp = 0
    fp = 0
    fn = 0

    matched = set()

    for i, p in pred_df.iterrows():
        found = False

        for j, g in gold_df.iterrows():
            if j in matched:
                continue

            if match_type == "exact" and exact_match(p, g):
                found = True
            elif match_type == "partial" and partial_match(p, g):
                found = True
            elif match_type == "type" and type_agnostic_match(p, g):
                found = True

            if found:
                tp += 1
                matched.add(j)
                break

        if not found:
            fp += 1

    fn = len(gold_df) - len(matched)
    return compute_metrics(tp, fp, fn)







def error_analysis(pred_df, gold_df):
    errors = {
        "boundary": 0,
        "type": 0,
        "missing": 0,
        "spurious": 0
    }

    matched = set()

    for i, p in pred_df.iterrows():
        found = False

        for j, g in gold_df.iterrows():
            if j in matched:
                continue

            if partial_match(p, g):
                if p["entity_label"] != g["entity_label"]:
                    errors["type"] += 1
                else:
                    errors["boundary"] += 1

                matched.add(j)
                found = True
                break

        if not found:
            errors["spurious"] += 1

    errors["missing"] = len(gold_df) - len(matched)

    return errors






    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")

        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")

        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")

        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")

        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")

        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            metrics = evaluate_ner(spacy_entities, gold)
            if metrics is not None:
                print(f"\nspaCy evaluation: {metrics}")
        if hf_entities is not None:
            metrics = evaluate_ner(hf_entities, gold)
            if metrics is not None:
                print(f"\nHF evaluation: {metrics}")


    counts = entity_counts_by_category(df, spacy_entities)
    print("\nEntity Counts by Category:")
    print(counts)



    plt.figure(figsize=(10, 6))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues")
    plt.title("Entity Distribution by Category")
    plt.xlabel("Entity Label")
    plt.ylabel("Category")
    plt.tight_layout()


    plt.savefig("entity_heatmap.png")

    plt.show()


    cat_eval = evaluate_per_category(df, spacy_entities, gold)

    print("\nEvaluation per Category:")
    for cat, metrics in cat_eval.items():
        print(cat, metrics)


    co = entity_cooccurrence(spacy_entities)

    print("\nTop Co-occurrences:")
    for pair, count in list(co.items())[:10]:
        print(pair, count)


    tfidf = compute_entity_tfidf(spacy_entities)

    print("\nSample TF-IDF:")
    for tid in list(tfidf.keys())[:2]:
        print(tid, list(tfidf[tid].items())[:5])


    advanced_exact = evaluate_advanced(spacy_entities, gold, "exact")
    advanced_partial = evaluate_advanced(spacy_entities, gold, "partial")
    advanced_type = evaluate_advanced(spacy_entities, gold, "type")

    print("\nAdvanced Evaluation:")
    print("Exact:", advanced_exact)
    print("Partial:", advanced_partial)
    print("Type-Agnostic:", advanced_type)


    errors = error_analysis(spacy_entities, gold)

    print("\nError Analysis:")
    print(errors)






    counts_spacy = entity_counts_by_category(df, spacy_entities)

plt.figure(figsize=(10, 6))
sns.heatmap(counts_spacy, annot=True, fmt="d", cmap="Blues")
plt.title("spaCy Entity Distribution by Category")
plt.xlabel("Entity Label")
plt.ylabel("Category")
plt.tight_layout()

plt.savefig("spacy_heatmap.png")
plt.show()






counts_hf = entity_counts_by_category(df, hf_entities)

plt.figure(figsize=(10, 6))
sns.heatmap(counts_hf, annot=True, fmt="d", cmap="Greens")
plt.title("HF Entity Distribution by Category")
plt.xlabel("Entity Label")
plt.ylabel("Category")
plt.tight_layout()

plt.savefig("hf_heatmap.png")
plt.show()


                # =========================

