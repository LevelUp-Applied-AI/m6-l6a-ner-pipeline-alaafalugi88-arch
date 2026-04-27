# Lab 6A — NER Pipeline (spaCy vs Hugging Face)

Module 6 Week A — AI.SPIRE Applied AI & ML Systems

Build and compare Named Entity Recognition (NER) pipelines using spaCy and Hugging Face on climate-related text data.

---

## Objective

The goal of this lab is to:

- Understand how different NLP systems behave  
- Compare performance between traditional and deep learning approaches  
- Analyze trade-offs between precision and recall  
- Evaluate models under strict matching conditions  

---

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Notes

- PyTorch is required for the Hugging Face pipeline  
- The spaCy model is lightweight (~12MB)  
- The Hugging Face model (~250MB) downloads once and is cached  
- Annotation styles may differ between predictions and the gold dataset  

---

## Tasks

Complete the following functions in `ner_pipeline.py`:

1. `load_data(filepath)` — Load the dataset  
2. `explore_data(df)` — Return dataset summary (shape, language, categories, length stats)  
3. `preprocess_text(text, nlp)` — Normalize and tokenize text  
4. `extract_spacy_entities(df, nlp)` — Extract entities using spaCy  
5. `extract_hf_entities(df, ner_pipeline)` — Extract entities using Hugging Face  
6. `compare_ner_outputs(spacy_df, hf_df)` — Compare entity outputs  
7. `evaluate_ner(predicted_df, gold_df)` — Compute precision, recall, F1  

---

## Pipeline Overview

### Data Exploration

- Dataset shape: (200, 5)  
- Languages: {'en': 132, 'ar': 68}  
- Categories: {'adaptation': 61, 'science': 50, 'impact': 46, 'policy': 43}  

Text length statistics:
- Mean: 56.99  
- Min: 36  
- Max: 73  

---

### Text Preprocessing

Each text is:
- Unicode-normalized (NFC)  
- Tokenized using spaCy  
- Cleaned (remove punctuation and whitespace)  
- Lemmatized and lowercased  

Example:
```
['the', 'ipcc', 'release', 'its', 'sixth', 'assessment', 'report', 'in', 'march', '2023']
```

---

### Named Entity Recognition

#### spaCy
- Extracted: 649 entities  
- Fast and broad extraction  
- Higher recall  

#### Hugging Face (BERT)
- Model: dslim/bert-base-NER  
- Uses aggregation_strategy="simple"  
- Extracted: 410 entities  
- More selective  

---

## Model Comparison

- Both systems agreed on: 78 entities  
- spaCy-only: 559 entities  
- HF-only: 326 entities  

---

## Evaluation

Entities are matched using:

(text_id, normalized_entity_text, entity_label)

### spaCy Results
- Precision: 0.056  
- Recall: 0.529  
- F1 Score: 0.102  

### Hugging Face Results
- Precision: 0.015  
- Recall: 0.088  
- F1 Score: 0.025  

---

## Results and Insights

- spaCy achieves higher recall due to extracting more entities  
- Precision is lower due to noise  
- Hugging Face is more conservative and context-aware  
- Both models are affected by strict evaluation conditions  

Why scores are low:
- Exact matching is required  
- Small formatting differences count as errors  
- Annotation styles differ between predicted and gold data  

---

## How to Run

```bash
python ner_pipeline.py
```

---

## Output

The script prints:
- Dataset statistics  
- Sample tokens  
- Entity counts  
- Comparison results  
- Evaluation metrics  

---

## Key Takeaways

- NER performance depends on model and evaluation method  
- Higher recall does not mean better performance  
- Exact matching can underestimate real quality  
- Transformer models are powerful but sensitive to data differences  

---

## Submission

1. Create a branch:
```
git checkout -b lab-6a-ner-pipeline
```

2. Push your branch:
```
git push --set-upstream origin lab-6a-ner-pipeline
```

3. Open a Pull Request to `main`

4. Submit your PR link in:
TalentLMS → Module 6 Week A → Lab 6A

---

## Conclusion

This project demonstrates the differences between traditional NLP models and transformer-based approaches. It highlights the importance of evaluation design and the challenges of strict entity matching.

---

## License

This repository is for educational use within the AI.SPIRE program. See LICENSE for details.