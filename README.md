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
│   │   ├── filter_jargon.py          – Jargon/non-jargon filtering pipeline
│   │   ├── build_ju_dataset.py       – JU (Jargon Understanding) MCQ benchmark construction
│   │   ├── build_ji_dataset.py       – JI (Jargon Identification) binary benchmark construction
│   │   └── MCQ_SBERT.py              – Shared SBERT utilities for distractor selection
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
