# Domain-Specific Jargon in Large Language Models: A Comparative Analysis between General-Purpose and Specialist Models
## Authors

- **Darin Keng**<br>
  Department of Statistics, University of Chicago, Chicago, Illinois<br>
  [dkeng@uchicago.edu](mailto:dkeng@uchicago.edu)
- **Zhewei Sun**<br>
  Toyota Technological Institute at Chicago, Chicago, Illinois<br>
  [zsun@ttic.edu](mailto:zsun@ttic.edu)
 
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
## Dataset Description (`data/`)

### Benchmark splits (`data/benchmarks/`)

| File | Task | Split | Examples |
|---|---|---|---|
| `ju_test.jsonl` | JU | Test | 1,495 |
| `ju_train.jsonl` | JU | Train | 4183 |
| `ji_test.jsonl` | JI | Test | 2,990 |
| `ji_train.jsonl` | JI | Dev | 1200 |
| `mat_ji_test.jsonl` | Mat-JI (cross-domain) | Test | 2,990 |

All benchmark files are in JSONL format. Each line is a JSON object with the following fields:
- `text`: Full prompt in Llama-3.1 chat format (system + user + gold answer)
- `options`: List of valid option letters (e.g. `["A", "B"]` for JI, `["A", "B", "C", "D", "E"]` for JU)
- `gold_letter`: The correct answer letter

### Data source and license

The medical jargon data is derived from **README-exp_good** (Yao et al., 2024), distributed under CC-BY-NC 4.0 for research purposes. The Mat-JI benchmark is derived from **MatScholar** (Song et al., 2023), released under the MIT License. **BoolQ** (Clark et al., 2019) is used under CC-BY-SA 3.0. Our derived benchmarks inherit the same access conditions and are intended only for research on model behavior, not for clinical deployment.

---

## Evaluation Pipeline

Our component decomposition and reweighting experiments use the framework from:

> Ting-Yun Chang, Jesse Thomason, and Robin Jia. "When Parts Are Greater Than Sums: Individual LLM Components Can Outperform Full Models." EMNLP 2024.

Code: [terarachang/LLMDecomp](https://github.com/terarachang/LLMDecomp)

We made one modification to the original codebase before running our experiments: adding our jargon datasets to `Demo_Dataset_Map` in `config.py` so that `decompose.py` runs in test-only mode. No other changes to the codebase were required.

### Running `decompose.py` (Section 5.2 — per-component accuracy)

To evaluate per-component accuracy on our benchmarks, pass the dataset name via `--dataset`. The dataset name must match the filename stem of the corresponding JSONL file in LLMDecomp's `data/` directory (e.g. `medical_mcq` for `medical_mcq_test-f1.jsonl`).

```bash
# JU (Jargon Understanding)
python decompose.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset medical_mcq \
    --format 1 --n_shots 0 --batch_size 8 --seed_list 0

```

Run the same commands with `--model_name TsinghuaC3I/Llama-3.1-8B-UltraMedical` to obtain results for the medically fine-tuned model.

### Running `train_components.py` (Section 5.3 — component reweighting)

To learn component weights for reweighting, update the `model` and `task` variables at the top of `train_components.py` to match the model and dataset you want to reweight, then run the script directly.

---

## Data source and license

The medical jargon data is derived from **README-exp_good** (Yao et al., 2024), distributed under CC-BY-NC 4.0 for research purposes. The Mat-JI benchmark is derived from **MatScholar** (Song et al., 2023), released under the MIT License. **BoolQ** (Clark et al., 2019) is used under CC-BY-SA 3.0. Our derived benchmarks inherit the same access conditions and are intended only for research on model behavior, not for clinical deployment.
