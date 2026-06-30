"""
build_ji_dataset.py

Generates the Jargon Identification (JI) binary-classification benchmark
described in Section 3.3 of the paper. Given a TERM (with no surrounding
context in the current prompt format), the model must answer whether the
term is medical jargon (A) or not (B).

Two input CSVs are required:
  - jargon CSV:     terms classified as jargon (positive examples -> gold A)
  - non-jargon CSV: terms classified as non-jargon (negative examples -> gold B)

Class counts are balanced before splitting, then divided into a small dev
split and the main test split. All reported results in the paper use the
test split only.

--- On the dev split ---
The dev split is NOT used for any evaluation reported in the paper. It is
generated solely because the component decomposition pipeline from Chang et
al. (2024) -- https://github.com/terarachang/LLMDecomp -- requires both
a dev and test file to exist on disk for any dataset not in its built-in
Demo_Dataset_Map. Specifically, decompose.py (LLMDecomp) checks:

    modes = ['test'] if dataset in Demo_Dataset_Map else ['dev', 'test']

Since our jargon datasets are custom (not in Demo_Dataset_Map), decompose.py
iterates over both modes and will crash if dev files are missing. The dev
files are read purely to cache per-component projections during the
decomposition run; the component reweighting (Section 5.3) and all accuracy
numbers reported in the paper are evaluated on the test split.

The --dev_frac default of 0.1 matches what was used to generate the
benchmark for the paper. Setting --dev_frac 0.0 would disable dev-split
generation if decompose.py is not needed.

--- On EHR context ---
--max_ehr_len and the EHR column are read from the input CSVs but the
current prompt template (format_example) does NOT include EHR context in
the generated text -- only the bare TERM is shown to the model, matching
Appendix G (Table 8) of the paper. The EHR plumbing is kept here only
because it is still read from the CSV upstream; --max_ehr_len currently
has no effect. If you want to reintroduce EHR context into the prompt,
uncomment the relevant line in format_example().

Output: JSONL files compatible with decompose.py in terarachang/LLMDecomp.

Usage:
    python build_ji_dataset.py \
        --input_jargon exp_good_filter_jargon_0.5.csv \
        --input_non_jargon exp_good_filter_non_jargon_0.5.csv \
        --dataset jargon_detect \
        --dev_frac 0.1 \
        --seed 42 \
        --max_examples 2000
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical assistant. Given a TERM, determine whether the term is medical jargon — "
    "specialized medical terminology that a layperson would likely not understand. "
    "Answer only with a single letter (A or B)."
)


def format_example(term: str, ehr_text: str, is_jargon: bool) -> dict:
    """Format a single JI binary-QA example.

    NOTE: ehr_text is accepted but not currently inserted into the prompt;
    only `term` is shown to the model. See module docstring.
    """
    gold_letter = "A" if is_jargon else "B"

    user_content = (
        f"TERM: {term}\n\n"
        # f"EHR CONTEXT: {ehr_text}\n\n"  # currently unused -- see module docstring
        f"Is this term a medical jargon (specialized terminology "
        f"a layperson would not understand)?\n\n"
        f"A) Yes\n"
        f"B) No\n\n"
        f"Answer with A or B only."
    )

    full_text = (
        f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{gold_letter}"
    )

    return {
        "text": full_text,
        "options": ["A", "B"],
        "gold_letter": gold_letter,
    }


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def generate_examples(
    jargon_df: pd.DataFrame,
    non_jargon_df: pd.DataFrame,
    rng: np.random.Generator,
    col_term: str = "term",
    col_ehr: str = "EHR",
) -> list[dict]:
    """Generate balanced positive + negative JI examples.

    Positives: all rows from jargon_df.
    Negatives: sampled (without replacement) from non_jargon_df to match
    the positive count, falling back to all available negatives if there
    aren't enough.
    """
    examples = []

    for _, row in tqdm(jargon_df.iterrows(), total=len(jargon_df), desc="Positives"):
        term = str(row[col_term]).strip()
        ehr = str(row[col_ehr]).strip()
        examples.append(format_example(term, ehr, is_jargon=True))

    n_pos = len(examples)
    n_neg_available = len(non_jargon_df)

    if n_neg_available >= n_pos:
        neg_idx = rng.choice(n_neg_available, size=n_pos, replace=False)
    else:
        neg_idx = np.arange(n_neg_available)
        print(f"  Warning: only {n_neg_available} negatives for {n_pos} positives")

    for i in tqdm(neg_idx, desc="Negatives"):
        row = non_jargon_df.iloc[i]
        term = str(row[col_term]).strip()
        ehr = str(row[col_ehr]).strip()
        examples.append(format_example(term, ehr, is_jargon=False))

    a_count = sum(1 for e in examples if e["gold_letter"] == "A")
    b_count = sum(1 for e in examples if e["gold_letter"] == "B")
    print(f"Generated {len(examples)} examples: {a_count} positive (A), {b_count} negative (B)")
    return examples


def reorder_balanced_prefix(examples: list[dict], rng: np.random.Generator) -> list[dict]:
    """Reorder so the first two examples cover both classes (A then B).

    LLMDecomp's decompose.py uses the dev file as a few-shot prefix when
    n_shots > 0. Putting one example of each class first ensures the prefix
    is class-balanced regardless of how many shots are used. The remainder
    is shuffled randomly.
    """
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["gold_letter"]].append(ex)

    ordered = []
    for letter in ["A", "B"]:
        if by_class[letter]:
            ordered.append(by_class[letter].pop(0))

    remaining = by_class["A"] + by_class["B"]
    perm = rng.permutation(len(remaining)).tolist()
    ordered.extend(remaining[i] for i in perm)

    return ordered


def cap_balanced(examples: list[dict], max_examples: int | None, rng: np.random.Generator) -> list[dict]:
    """Cap total examples while keeping the A/B split as close to even as
    possible. If one class has fewer available examples than half the cap,
    the remainder is given to the other class."""
    if max_examples is None or len(examples) <= max_examples:
        return examples

    print(f"  Capping examples: {len(examples)} -> {max_examples}")
    a_examples = [e for e in examples if e["gold_letter"] == "A"]
    b_examples = [e for e in examples if e["gold_letter"] == "B"]

    half = max_examples // 2
    n_a = min(half, len(a_examples))
    n_b = min(half, len(b_examples))
    # If one class came up short of `half`, let the other absorb the remainder.
    n_a = min(len(a_examples), max_examples - n_b)
    n_b = min(len(b_examples), max_examples - n_a)

    rng.shuffle(a_examples)
    rng.shuffle(b_examples)

    capped = a_examples[:n_a] + b_examples[:n_b]
    perm = rng.permutation(len(capped)).tolist()
    return [capped[i] for i in perm]


def save_jsonl(examples: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved: {path} ({len(examples)} examples)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    task = args.dataset.split("_")[0]
    out_dir = os.path.join("data", task)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # --- Load data ---
    print("Loading data...")
    jargon_df = pd.read_csv(args.input_jargon).dropna(subset=[args.col_ehr, args.col_term])
    non_jargon_df = pd.read_csv(args.input_non_jargon).dropna(subset=[args.col_ehr, args.col_term])
    print(f"  Jargon rows:     {len(jargon_df)}")
    print(f"  Non-jargon rows: {len(non_jargon_df)}")

    # Balance classes before splitting into dev/test.
    min_size = min(len(jargon_df), len(non_jargon_df))
    jargon_df = jargon_df.sample(n=min_size, random_state=args.seed).reset_index(drop=True)
    non_jargon_df = non_jargon_df.sample(n=min_size, random_state=args.seed).reset_index(drop=True)
    print(f"  Balanced to {min_size} per class")

    # Shuffle each class, then split into dev/test.
    jargon_df = jargon_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    non_jargon_df = non_jargon_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    dev_size_j = max(1, int(len(jargon_df) * args.dev_frac)) if args.dev_frac > 0 else 0
    dev_size_n = max(1, int(len(non_jargon_df) * args.dev_frac)) if args.dev_frac > 0 else 0

    dev_jargon = jargon_df.iloc[:dev_size_j].reset_index(drop=True)
    test_jargon = jargon_df.iloc[dev_size_j:].reset_index(drop=True)
    dev_non_jargon = non_jargon_df.iloc[:dev_size_n].reset_index(drop=True)
    test_non_jargon = non_jargon_df.iloc[dev_size_n:].reset_index(drop=True)

    print(f"\n  Dev:  {len(dev_jargon)} jargon + {len(dev_non_jargon)} non-jargon")
    print(f"  Test: {len(test_jargon)} jargon + {len(test_non_jargon)} non-jargon")

    # --- Generate DEV files (multiple seeds, for sensitivity checks) ---
    print(f"\n{'=' * 60}\nGenerating DEV files ({args.n_dev_seeds} seeds)...\n{'=' * 60}")
    for s in range(args.n_dev_seeds):
        print(f"\n--- Seed {s} ---")
        dev_rng = np.random.default_rng(s)
        dev_examples = generate_examples(
            dev_jargon, dev_non_jargon, dev_rng,
            col_term=args.col_term, col_ehr=args.col_ehr,
        )
        dev_examples = reorder_balanced_prefix(dev_examples, dev_rng)
        out_path = os.path.join(out_dir, f"{args.dataset}_dev-f1-s{s}.jsonl")
        save_jsonl(dev_examples, out_path)

    # --- Generate TEST file ---
    print(f"\n{'=' * 60}\nGenerating TEST file...\n{'=' * 60}")
    test_rng = np.random.default_rng(args.seed)
    test_examples = generate_examples(
        test_jargon, test_non_jargon, test_rng,
        col_term=args.col_term, col_ehr=args.col_ehr,
    )

    perm = test_rng.permutation(len(test_examples)).tolist()
    test_examples = [test_examples[i] for i in perm]
    test_examples = cap_balanced(test_examples, args.max_examples, test_rng)

    out_path = os.path.join(out_dir, f"{args.dataset}_test-f1.jsonl")
    save_jsonl(test_examples, out_path)

    # --- Print samples ---
    print(f"\n{'=' * 60}\nSample POSITIVE example (gold = A):\n{'=' * 60}")
    pos = next((e for e in test_examples if e["gold_letter"] == "A"), None)
    if pos:
        print(pos["text"][:500])
        print(f"\nOptions: {pos['options']}")

    print(f"\n{'=' * 60}\nSample NEGATIVE example (gold = B):\n{'=' * 60}")
    neg = next((e for e in test_examples if e["gold_letter"] == "B"), None)
    if neg:
        print(neg["text"])
        print(f"\nOptions: {neg['options']}")

    # --- Format verification summary ---
    a_count = sum(1 for e in test_examples if e["gold_letter"] == "A")
    b_count = sum(1 for e in test_examples if e["gold_letter"] == "B")
    print(f"\n{'=' * 60}\nFormat verification:\n{'=' * 60}")
    print(f"  Dataset name:    {args.dataset}")
    print(f"  Task prefix:     {task}")
    print(f"  Output dir:      {out_dir}/")
    print(f"  Test file:       {args.dataset}_test-f1.jsonl")
    print(f"  Dev files:       {args.dataset}_dev-f1-s{{0..{args.n_dev_seeds - 1}}}.jsonl")
    print(f"  Options:         ['A', 'B']  (A=Yes jargon, B=No)")
    print(f"  n_classes:       2")
    print(f"  Test examples:   {len(test_examples)}")
    print(f"  Test balance:    {a_count} gold-A / {b_count} gold-B")

    print(f"\nReady! Run decompose.py with:")
    print(f"  python decompose.py \\")
    print(f"      --model_name m42-health/Llama3-Med42-8B \\")
    print(f"      --dataset {args.dataset} \\")
    print(f"      --format 1 \\")
    print(f"      --n_shots 0 \\")
    print(f"      --batch_size 4 \\")
    print(f"      --seed_list 0 1 2")

    summary = {
        "input_jargon": args.input_jargon,
        "input_non_jargon": args.input_non_jargon,
        "dataset": args.dataset,
        "dev_jargon": len(dev_jargon),
        "dev_non_jargon": len(dev_non_jargon),
        "test_jargon": len(test_jargon),
        "test_non_jargon": len(test_non_jargon),
        "total_test_examples": len(test_examples),
        "test_positives": a_count,
        "test_negatives": b_count,
        "system_prompt": SYSTEM_PROMPT,
    }
    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_jargon", type=str, required=True,
                         help="Path to jargon CSV (columns: term, EHR).")
    parser.add_argument("--input_non_jargon", type=str, required=True,
                         help="Path to non-jargon CSV (columns: term, EHR).")
    parser.add_argument("--dataset", type=str, default="jargon_detect")
    parser.add_argument("--dev_frac", type=float, default=0.1,
                         help="Fraction of each class held out for the dev split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dev_seeds", type=int, default=5,
                         help="Number of differently-seeded dev files to generate. "
                              "LLMDecomp's decompose.py iterates over seed_list for "
                              "both dev and test modes; this should match --seed_list "
                              "passed to decompose.py.")
    parser.add_argument("--col_ehr", type=str, default="EHR")
    parser.add_argument("--col_term", type=str, default="term")
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Maximum total test examples (pos+neg), class-balanced.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
