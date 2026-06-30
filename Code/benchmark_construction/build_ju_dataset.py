"""
build_ju_dataset.py

Constructs the Jargon Understanding (JU) multiple-choice benchmark described
in Section 3.2 of the paper. For each jargon term, builds a 5-way MCQ where
the correct definition is paired with semantically similar distractor
definitions drawn from a separate distractor pool (the train split), so that
test-time distractors never leak from the test set itself.

Distractor selection (see MCQ_SBERT.select_distractors_from_train) ranks
candidates by SBERT cosine similarity to the correct definition, keeping
those within [min_similarity, max_similarity], and enforces unique terms /
unique definitions across the chosen distractors. If too few candidates fall
within the similarity range, it falls back to the most similar remaining
candidates outside that range.

Usage:
    # Generate the test-split MCQ benchmark (questions from val_df,
    # distractors drawn from train_df):
    python build_ju_dataset.py --split test \
        --val-csv val_df_ehr_0.5.csv --train-csv train_df_ehr_0.5.csv \
        --output medical_mcq_0.5_test.jsonl

    # Generate the dev-split MCQ benchmark (questions AND distractors both
    # drawn from train_df; safe because select_distractors_from_train
    # excludes a term from being its own distractor):
    python build_ju_dataset.py --split dev \
        --train-csv train_df_ehr_0.5.csv \
        --output medical_mcq_0.5_dev.jsonl

    # Preview a handful of generated MCQs without writing a file:
    python build_ju_dataset.py --split test \
        --val-csv val_df_ehr_0.5.csv --train-csv train_df_ehr_0.5.csv \
        --preview-only --preview-n 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from MCQ_SBERT import (
    clean_choices,
    clean_definition,
    compute_definition_embeddings,
    select_distractors_from_train,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SBERT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

COL_TERM = "jargon_term"          # display form of the term
COL_TERM_LOWER = "jargon_clean"   # normalized form, used for matching
COL_DEF = "gen_def_clean"
COL_EHR = "EHR"

K_CHOICES = 5                     # 1 correct + 4 distractors (A-E)
MIN_SIMILARITY = 0.5
MAX_SIMILARITY = 0.9
SEED = 42

SYSTEM_PROMPT = (
    "You are a clinical assistant. Given a medical TERM (jargon), "
    "and several candidate general definitions, select the single best general definition. "
    "Reply with ONE letter only (A, B, C, D, or E)"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_clean_split(csv_path: Path) -> pd.DataFrame:
    """Load a jargon split CSV, drop duplicate (term, def) rows, and apply
    clean_definition() to strip quotation artifacts from gen_def_clean."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=[COL_TERM_LOWER, COL_DEF])
    df = df.reset_index(drop=True)
    df[COL_DEF] = df[COL_DEF].apply(clean_definition)
    return df


def load_or_compute_embeddings(df: pd.DataFrame, cache_path: Path) -> np.ndarray:
    """Load cached SBERT embeddings for a split if present, else compute
    and cache them."""
    if cache_path.exists():
        print(f"Loaded cached embeddings: {cache_path}")
        return np.load(cache_path)

    print(f"Computing embeddings -> {cache_path} (this may take a few minutes)...")
    embeddings = compute_definition_embeddings(
        df, col_definition=COL_DEF, model_name=SBERT_MODEL_NAME
    )
    np.save(cache_path, embeddings)
    return embeddings


# ---------------------------------------------------------------------------
# MCQ formatting
# ---------------------------------------------------------------------------

def format_medical_example_instruct(term: str, choices: list[str], gold_pos: int, abc: str = "ABCDE") -> dict:
    """Format a single MCQ in Llama-3-Instruct style.

    Returns a dict with "text" (full prompt + gold answer), "options"
    (the letter labels used), and "gold_letter" (the correct letter).
    """
    opts = "\n".join(f"{abc[i]}) {choices[i]}" for i in range(len(choices)))

    user_content = (
        f"TERM: {term}\n"
        f"Given a term, choose the best general definition for the term:\n\n"
        f"{opts}\n\n"
        f"Answer with a single letter only."
    )

    gold_letter = abc[gold_pos]
    full_text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n\n"
        f"<|user|>\n{user_content}\n\n"
        f"<|assistant|>\n{gold_letter}"
    )

    return {
        "text": full_text,
        "options": list(abc[:len(choices)]),
        "gold_letter": gold_letter,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_mcq_dataset(
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    K: int = K_CHOICES,
    min_similarity: float = MIN_SIMILARITY,
    max_similarity: float = MAX_SIMILARITY,
    seed: int = SEED,
    record_similarity: bool = False,
) -> tuple[list[dict], int]:
    """Generate the JU MCQ dataset: one example per row of val_df, with
    K-1 distractors drawn from train_df via select_distractors_from_train.

    If record_similarity=True, each example additionally carries SBERT
    similarity diagnostics (sbert_mean/min/max/std, distractor_terms,
    correct_term) for later calibration/analysis. This is the only
    difference versus the lightweight mode; the MCQ construction logic
    itself is identical either way.

    Returns:
        (examples, skipped_count)
    """
    rng = np.random.default_rng(seed)
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    examples = []
    skipped = 0

    for i in tqdm(range(len(val_df)), desc="Generating JU MCQs"):
        distractor_indices = select_distractors_from_train(
            val_idx=i,
            val_df=val_df,
            val_embeddings=val_embeddings,
            train_df=train_df,
            train_embeddings=train_embeddings,
            col_term=COL_TERM_LOWER,
            col_def=COL_DEF,
            K=K,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
        )

        if len(distractor_indices) < K - 1:
            skipped += 1
            continue

        distractor_defs = train_df.iloc[distractor_indices][COL_DEF].astype(str).tolist()
        distractor_terms = train_df.iloc[distractor_indices][COL_TERM].astype(str).tolist()
        correct_def = str(val_df.iloc[i][COL_DEF])
        correct_term = str(val_df.iloc[i][COL_TERM])

        all_defs = distractor_defs + [correct_def]
        all_terms = distractor_terms + [correct_term]
        all_defs = clean_choices(all_terms, all_defs)  # strip the term from its own definition

        # Shuffle so the correct answer's position is randomized.
        indices = list(range(K))
        rng.shuffle(indices)
        shuffled_defs = [all_defs[idx] for idx in indices]
        gold_pos = indices.index(K - 1)  # correct answer was appended last, pre-shuffle

        term = val_df.iloc[i][COL_TERM]
        example = format_medical_example_instruct(term, shuffled_defs, gold_pos)

        if record_similarity:
            correct_embedding = val_embeddings[i].reshape(1, -1)
            similarities = [
                float(cosine_similarity(correct_embedding, train_embeddings[idx].reshape(1, -1))[0][0])
                for idx in distractor_indices
            ]
            example["sbert_similarities"] = similarities
            example["sbert_mean"] = float(np.mean(similarities))
            example["sbert_max"] = float(np.max(similarities))
            example["sbert_min"] = float(np.min(similarities))
            example["sbert_std"] = float(np.std(similarities))
            example["distractor_terms"] = distractor_terms
            example["correct_term"] = str(val_df.iloc[i][COL_TERM_LOWER])

        examples.append(example)

    return examples, skipped


def save_jsonl(examples: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples -> {output_path}")


# ---------------------------------------------------------------------------
# Preview mode
# ---------------------------------------------------------------------------

def preview_examples(
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    train_embeddings: np.ndarray,
    n: int = 5,
    seed: int = SEED,
) -> None:
    """Print a handful of generated MCQs with their distractor similarity
    scores, for manually sanity-checking benchmark quality (e.g. after
    changing min_similarity/max_similarity).

    This is the interactive-inspection equivalent of running
    generate_mcq_dataset() with record_similarity=True, but limited to a
    small random sample and printed instead of saved.
    """
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(val_df), size=min(n, len(val_df)), replace=False)

    for example_num, i in enumerate(sample_indices, 1):
        print(f"\n{'=' * 80}\nEXAMPLE {example_num}/{len(sample_indices)}\n{'=' * 80}")

        distractor_indices = select_distractors_from_train(
            val_idx=i,
            val_df=val_df,
            val_embeddings=val_embeddings,
            train_df=train_df,
            train_embeddings=train_embeddings,
            col_term=COL_TERM_LOWER,
            col_def=COL_DEF,
            K=K_CHOICES,
            min_similarity=MIN_SIMILARITY,
            max_similarity=MAX_SIMILARITY,
        )

        if len(distractor_indices) == 0:
            print("Skipping - not enough candidates")
            continue

        correct_term = val_df.iloc[i][COL_TERM_LOWER]
        correct_def = val_df.iloc[i][COL_DEF]
        correct_embedding = val_embeddings[i].reshape(1, -1)

        print(f"\nQUESTION TERM: {correct_term}")
        print(f"CORRECT DEFINITION: {correct_def}\n")

        distractor_defs = train_df.iloc[distractor_indices][COL_DEF].astype(str).tolist()
        distractor_terms = train_df.iloc[distractor_indices][COL_TERM_LOWER].astype(str).tolist()

        print("DISTRACTORS FROM TRAIN SET (with similarity scores):")
        print("-" * 80)

        similarities = []
        for j, idx in enumerate(distractor_indices, 1):
            distractor_embedding = train_embeddings[idx].reshape(1, -1)
            sim = cosine_similarity(correct_embedding, distractor_embedding)[0][0]
            similarities.append(sim)
            print(f"\n{j}. [{distractor_terms[j - 1]}] (Similarity: {sim:.3f})")
            print(f"   {distractor_defs[j - 1]}")

        print("\n" + "-" * 80)
        print("SIMILARITY STATISTICS:")
        print(f"  Mean: {np.mean(similarities):.3f}  Min: {np.min(similarities):.3f}  Max: {np.max(similarities):.3f}")

        # Build and show the shuffled MCQ as it would appear in the dataset.
        all_defs = distractor_defs + [correct_def]
        all_terms = distractor_terms + [correct_term]
        shuffle_idx = list(range(K_CHOICES))
        rng.shuffle(shuffle_idx)
        choices_shuf = [all_defs[idx] for idx in shuffle_idx]
        terms_shuf = [all_terms[idx] for idx in shuffle_idx]
        gold_pos_new = shuffle_idx.index(K_CHOICES - 1)

        print("\n" + "-" * 80)
        print("MCQ FORMAT (shuffled):")
        print("-" * 80)
        for j, (choice, term) in enumerate(zip(choices_shuf, terms_shuf)):
            marker = " (correct)" if j == gold_pos_new else ""
            print(f"\n{chr(65 + j)}) {choice}{marker}")
            print(f"   [{term}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    train_df = load_and_clean_split(Path(args.train_csv))
    train_embeddings = load_or_compute_embeddings(train_df, Path(args.train_embeddings_cache))
    print(f"Train set: {len(train_df)} terms")

    if args.split == "test":
        if args.val_csv is None:
            raise ValueError("--val-csv is required for --split test")
        val_df = load_and_clean_split(Path(args.val_csv))
        val_embeddings = load_or_compute_embeddings(val_df, Path(args.val_embeddings_cache))
        print(f"Val (test questions) set: {len(val_df)} terms")
    elif args.split == "dev":
        # Dev set draws BOTH questions and the distractor pool from train_df.
        # This is safe: select_distractors_from_train excludes a term from
        # being offered as its own distractor (mask on col_term != correct_term).
        val_df = train_df
        val_embeddings = train_embeddings
    else:
        raise ValueError(f"Unknown split: {args.split}")

    if args.preview_only:
        preview_examples(val_df, val_embeddings, train_df, train_embeddings, n=args.preview_n)
        return

    examples, skipped = generate_mcq_dataset(
        val_df=val_df,
        val_embeddings=val_embeddings,
        train_df=train_df,
        train_embeddings=train_embeddings,
        K=K_CHOICES,
        min_similarity=MIN_SIMILARITY,
        max_similarity=MAX_SIMILARITY,
        seed=SEED,
        record_similarity=args.record_similarity,
    )

    print(f"\nGenerated {len(examples)} examples, skipped {skipped} (insufficient distractors)")

    output_path = Path(args.output)
    save_jsonl(examples, output_path)

    if examples:
        print("\nSample MCQ:")
        print("=" * 80)
        print(examples[0]["text"])
        print(f"\nOptions: {examples[0]['options']}")
        print(f"Gold letter: {examples[0]['gold_letter']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["test", "dev"], required=True,
                         help="'test' uses --val-csv for questions; 'dev' uses train_csv for both questions and distractors.")
    parser.add_argument("--val-csv", type=str, default=None,
                         help="Path to the val/test split CSV (required for --split test).")
    parser.add_argument("--train-csv", type=str, required=True,
                         help="Path to the train split CSV (always used as the distractor pool).")
    parser.add_argument("--val-embeddings-cache", type=str, default="val_embeddings.npy",
                         help="Path to cache/load val split SBERT embeddings.")
    parser.add_argument("--train-embeddings-cache", type=str, default="train_embeddings.npy",
                         help="Path to cache/load train split SBERT embeddings.")
    parser.add_argument("--output", type=str, default=None,
                         help="Output JSONL path (required unless --preview-only).")
    parser.add_argument("--record-similarity", action="store_true",
                         help="Attach SBERT similarity diagnostics to each saved example.")
    parser.add_argument("--preview-only", action="store_true",
                         help="Print a few example MCQs instead of generating the full dataset.")
    parser.add_argument("--preview-n", type=int, default=5,
                         help="Number of examples to print in --preview-only mode.")

    args = parser.parse_args()
    if not args.preview_only and args.output is None:
        parser.error("--output is required unless --preview-only is set")
    return args


if __name__ == "__main__":
    main(parse_args())
