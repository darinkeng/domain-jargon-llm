# Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models
## Authors

All authorship information has been anonymized for double-blind peer review.

## Overview

This repository accompanies the paper:  
**"Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models."**

We contribute two novel medical jargon evaluation benchmarks and evaluate a general-purpose Llama-3.1-8B-Instruct model against a medically fine-tuned variant (Llama-3.1-8B-UltraMedical). Using mechanistic interpretability tools, we trace how domain-specific fine-tuning affects the internal representation of jargon knowledge in LLMs.

This repository contains:

- Code for constructing the JU and JI jargon evaluation benchmarks
- Code for the cross-domain generalization analysis (Mat-JI)
- Analysis notebooks for calibration and cross-domain component comparison

---

## Repository Structure

```
├── code/
│   ├── benchmark_construction/
│   │   ├── jargon_filter.ipynb          – Jargon/non-jargon filtering pipeline
│   │   ├── ju_benchmark_construct.py    – JU (Jargon Understanding) MCQ benchmark construction
│   │   └── ji_benchmark_construct.py    – JI (Jargon Identification) binary benchmark construction
│   │   
│   └── analysis/
│       ├── calibration_analysis.ipynb   – ECE and reliability diagram analysis (Figure 2)
│       └── cross_domain_analysis.ipynb  – Cross-domain component generalization (Table 5)
│
└── data/
    ├── README.md                     – Data description, license, and access conditions
    └── benchmarks/
       ├── ju_test.jsonl             – JU test split (1,495 examples)
       ├── ju_dev.jsonl              – JU dev split
       ├── ji_test.jsonl             – JI test split (2,990 examples)
       ├── ji_dev.jsonl              – JI dev split
       └── mat_ji_test.jsonl         – Mat-JI cross-domain test split (2,990 examples)

```

---

## Code Descriptions (`Code/`)

### `benchmark_construction/jargon_filter.ipynb`

Constructs the medical jargon dataset (J) and non-jargon dataset (N) from the README dataset (Yao et al., 2024). The pipeline:

1. Loads README-exp_good from HuggingFace (`bio-nlp-umass/NoteAid-README`)
2. Cleans and normalizes jargon terms and definitions
3. Deduplicates near-identical definitions per term using SBERT cosine similarity
4. Loads a pre-built Wiktionary index from a `.npy` checkpoint (`wikt-en.npy`)
5. For each term, compare its medical definition against non-medical Wiktionary senses
6. Terms whose medical definition is sufficiently distant from all non-medical senses (similarity < 0.5) are labeled jargon; the rest are labeled non-jargon

---

### `benchmark_construction/ju_benchmark_construct.py`

Constructs the Jargon Understanding (JU) multiple-choice benchmark (Section 3.2). Each example presents a jargon term alongside five candidate definitions; the model must select the correct one. Distractors are drawn from the train split using SBERT cosine similarity to select semantically challenging but distinct candidates.

```bash
# Test split (questions from val CSV, distractors from train)
python build_ju_dataset.py --split test \
    --val-csv val_df_ehr_0.5.csv \
    --train-csv train_df_ehr_0.5.csv \
    --output data/benchmarks/ju_test.jsonl

# Dev split (questions and distractors both from train)
python build_ju_dataset.py --split dev \
    --train-csv train_df_ehr_0.5.csv \
    --output data/benchmarks/ju_dev.jsonl

# Preview a few examples without saving
python build_ju_dataset.py --split test \
    --val-csv val_df_ehr_0.5.csv \
    --train-csv train_df_ehr_0.5.csv \
    --preview-only --preview-n 5
```

---

### `benchmark_construction/ji_benchmark_construct.py`

Constructs the Jargon Identification (JI) binary benchmark (Section 3.3). Given a term, the model must answer whether it is medical jargon (A) or not (B). Positive examples are drawn from the jargon set; negative examples are sampled from the non-jargon set to maintain class balance.


```bash
python build_ji_dataset.py \
    --input_jargon exp_good_filter_jargon_0.5.csv \
    --input_non_jargon exp_good_filter_non_jargon_0.5.csv \
    --dataset jargon_detect \
    --dev_frac 0.0 \
    --seed 42 \
    --max_examples 2000
```

---

### `analysis/calibration_analysis.ipynb`

Computes Expected Calibration Error (ECE) and plots reliability diagrams for both benchmark tasks (Section 5.1, Figure 2). Requires prediction CSV files produced by `decompose.py` from [terarachang/LLMDecomp](https://github.com/terarachang/LLMDecomp).

Covers:
- JI calibration (binary, confidence range [0.5, 1.0])
- JU calibration (5-way MCQ, confidence range [0.0, 1.0])
- Confidence-when-wrong histograms
- Summary statistics table (accuracy, ECE, mean confidence, entropy, margin)

---

### `analysis/cross_domain_analysis.ipynb`

Identifies jargon-sensitive model components that generalize across domains (Section 5.4, Table 5). Compares per-component accuracy across three tasks — Med-JI, Mat-JI, and BoolQ — and isolates components that rank highly on both jargon tasks but not on the BoolQ control, indicating domain-agnostic jargon knowledge.

Requires output files from `decompose.py` for all three datasets. See the notebook's Setup section for the exact shell commands.

---
