
"""Part D1: TF-IDF policy retrieval and grounded policy answers.

Uses local policy documents and the supplied offline mock_llm.py.
Does not call external APIs or perform financial transactions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm


ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "clean" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"
RAG_DIR = OUTPUT_DIR / "rag"

ABSTENTION = "I could not find that information in the policy documents."
TOP_K = 3
MIN_SIMILARITY = 0.08


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving readable policy text."""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_policy_into_passages(text: str) -> list[str]:
    """Split a policy into sentence-like passages."""
    text = normalize_text(text)
    if not text:
        return []

    passages = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in passages if p.strip()]


def load_policy_documents() -> list[dict[str, str]]:
    """Load every .txt policy and represent its sentences as passages."""
    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    files = sorted(POLICY_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt policy documents found in {POLICY_DIR}"
        )

    passages: list[dict[str, str]] = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8-sig")
        for passage_number, passage in enumerate(
            split_policy_into_passages(text), start=1
        ):
            passages.append(
                {
                    "doc_name": file_path.name,
                    "passage_id": f"{file_path.stem}_{passage_number:03d}",
                    "text": passage,
                }
            )

    if not passages:
        raise ValueError("Policy files were found, but no passages were extracted.")

    return passages


def build_retriever(
    passages: list[dict[str, str]],
) -> tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF index over policy passages."""
    texts = [item["text"] for item in passages]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


def retrieve_passages(
    question: str,
    passages: list[dict[str, str]],
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Return the highest-scoring policy passages for a question."""
    question = normalize_text(question)
    if not question:
        return []

    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )[:top_k]

    results: list[dict[str, Any]] = []

    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        results.append(
            {
                **passages[index],
                "score": round(score, 6),
            }
        )

    return results


def extract_message_content(response: Any) -> str:
    """Extract text from the supplied mock's chat response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        content = None

    if isinstance(content, str):
        return content.strip()

    if content is None:
        return ""

    return str(content).strip()


def answer_from_policy(
    question: str,
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the offline policy assistant using retrieved evidence only."""
    if not retrieved:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "model_called": False,
        }

    best_score = max(item["score"] for item in retrieved)
    if best_score < MIN_SIMILARITY:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "model_called": False,
        }

    context_parts = [
        f"[{item['doc_name']}] {item['text']}"
        for item in retrieved
    ]
    context = "\n".join(context_parts)

    system_prompt = (
        "POLICY ASSISTANT\n"
        "Answer customer questions using only the policy context provided "
        "in the latest user message. Do not invent policy details. "
        "If the context does not contain the answer, respond exactly: "
        f"{ABSTENTION}"
    )

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {normalize_text(question)}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = mock_llm.chat(messages)
        answer = extract_message_content(response)
    except Exception as exc:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "model_called": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not answer:
        answer = ABSTENTION

    abstained = answer.strip().casefold() == ABSTENTION.casefold()

    return {
        "answer": answer,
        "abstained": abstained,
        "model_called": True,
    }


def evaluate_answer(
    question: str,
    expected_doc: str,
    expected_fact: str,
    retrieved: list[dict[str, Any]],
    answer: str,
    abstained: bool,
) -> dict[str, Any]:
    """Evaluate retrieval and answer behavior against the supplied labels."""
    expected_doc = str(expected_doc or "").strip()
    expected_fact = str(expected_fact or "").strip()
    answer_lower = answer.casefold()

    retrieved_docs = [item["doc_name"] for item in retrieved]
    if expected_doc and expected_doc.casefold() != "none":
        doc_hit = any(
            doc.casefold() == expected_doc.casefold()
            for doc in retrieved_docs
        )
    else:
        doc_hit = None

    if expected_fact:
        fact_terms = [
            token.casefold()
            for token in re.findall(r"\b[a-zA-Z0-9]+\b", expected_fact)
            if len(token) > 2
        ]
        unique_terms = list(dict.fromkeys(fact_terms))
        matched_terms = [
            token for token in unique_terms if token in answer_lower
        ]
        fact_coverage = (
            len(matched_terms) / len(unique_terms)
            if unique_terms
            else None
        )
    else:
        matched_terms = []
        fact_coverage = None

    expected_unanswerable = (
        expected_doc.casefold() == "none"
        or not expected_fact
    )

    return {
        "expected_doc_hit": doc_hit,
        "expected_fact_term_coverage": (
            round(fact_coverage, 4)
            if fact_coverage is not None
            else None
        ),
        "matched_expected_fact_terms": ";".join(matched_terms),
        "expected_unanswerable": expected_unanswerable,
        "abstention_correct": (
            abstained if expected_unanswerable else None
        ),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(parents=True, exist_ok=True)

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {EVAL_FILE}"
        )

    print("=" * 65)
    print("PART D1 — TF-IDF POLICY RAG")
    print("=" * 65)

    passages = load_policy_documents()
    vectorizer, matrix = build_retriever(passages)

    print(f"Policy documents loaded: {len(list(POLICY_DIR.glob('*.txt')))}")
    print(f"Policy passages indexed: {len(passages)}")

    # Save the indexed passage catalogue for reproducibility.
    passage_catalogue = pd.DataFrame(passages)
    passage_catalogue.to_csv(
        RAG_DIR / "policy_passages.csv",
        index=False,
        encoding="utf-8",
    )

    evaluation = pd.read_csv(EVAL_FILE).fillna("")

    required_columns = {"question", "expected_doc", "expected_fact"}
    missing_columns = required_columns - set(evaluation.columns)
    if missing_columns:
        raise ValueError(
            f"policy_eval.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result_rows: list[dict[str, Any]] = []

    for row_number, row in evaluation.iterrows():
        question = normalize_text(row["question"])
        expected_doc = normalize_text(row["expected_doc"])
        expected_fact = normalize_text(row["expected_fact"])

        retrieved = retrieve_passages(
            question,
            passages,
            vectorizer,
            matrix,
        )

        response = answer_from_policy(question, retrieved)
        answer = response["answer"]
        abstained = response["abstained"]

        evaluation_metrics = evaluate_answer(
            question=question,
            expected_doc=expected_doc,
            expected_fact=expected_fact,
            retrieved=retrieved,
            answer=answer,
            abstained=abstained,
        )

        result_rows.append(
            {
                "evaluation_id": row_number + 1,
                "question": question,
                "expected_doc": expected_doc,
                "expected_fact": expected_fact,
                "retrieved_documents": ";".join(
                    dict.fromkeys(
                        item["doc_name"] for item in retrieved
                    )
                ),
                "retrieved_passages": json.dumps(
                    [
                        {
                            "doc_name": item["doc_name"],
                            "passage_id": item["passage_id"],
                            "score": item["score"],
                            "text": item["text"],
                        }
                        for item in retrieved
                    ],
                    ensure_ascii=False,
                ),
                "answer": answer,
                "abstained": abstained,
                "model_called": response["model_called"],
                "error": response.get("error", ""),
                **evaluation_metrics,
            }
        )

        print(f"\n[{row_number + 1}/{len(evaluation)}] {question}")
        print(f"Retrieved: {', '.join(item['doc_name'] for item in retrieved) or 'None'}")
        print(f"Answer: {answer}")

    results = pd.DataFrame(result_rows)
    result_path = RAG_DIR / "policy_rag_eval.csv"
    results.to_csv(result_path, index=False, encoding="utf-8")

    total = len(results)
    doc_hit_rows = results["expected_doc_hit"].dropna()
    doc_hit_rate = (
        float(doc_hit_rows.mean())
        if not doc_hit_rows.empty
        else None
    )

    unanswerable_rows = results[
        results["expected_unanswerable"].astype(bool)
    ]
    abstention_values = unanswerable_rows["abstention_correct"].dropna()
    abstention_accuracy = (
        float(abstention_values.astype(bool).mean())
        if not abstention_values.empty
        else None
    )

    summary = {
        "component": "Part D1 - TF-IDF Policy RAG",
        "policy_document_count": len(list(POLICY_DIR.glob("*.txt"))),
        "passage_count": len(passages),
        "evaluation_question_count": total,
        "retrieval_top_k": TOP_K,
        "minimum_similarity_threshold": MIN_SIMILARITY,
        "expected_document_hit_rate": doc_hit_rate,
        "unanswerable_question_count": len(unanswerable_rows),
        "unanswerable_abstention_accuracy": abstention_accuracy,
        "model_calls": int(results["model_called"].astype(bool).sum()),
        "error_count": int(results["error"].astype(str).str.len().gt(0).sum()),
        "outputs": [
            "outputs/rag/policy_passages.csv",
            "outputs/rag/policy_rag_eval.csv",
            "outputs/rag/policy_rag_summary.json",
        ],
        "notes": [
            "TF-IDF is used for local retrieval; no external embedding API is used.",
            "Answer generation uses the supplied offline mock_llm.py.",
            "Fact-term coverage is a diagnostic, not a semantic correctness score.",
            "Evaluation results are measured from this run, not assumed in advance.",
        ],
    }

    summary_path = RAG_DIR / "policy_rag_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("PART D1 RUN SUMMARY")
    print("=" * 65)
    print(f"Evaluation questions: {total}")
    print(f"Expected-document hit rate: {doc_hit_rate}")
    print(f"Unanswerable questions: {len(unanswerable_rows)}")
    print(f"Abstention accuracy: {abstention_accuracy}")
    print(f"Errors: {summary['error_count']}")
    print(f"Results: {result_path.relative_to(ROOT)}")
    print(f"Summary: {summary_path.relative_to(ROOT)}")

    checks = {
        "Policy passages were indexed": len(passages) > 0,
        "Evaluation dataset was processed": total == len(evaluation) and total > 0,
        "Every evaluation has an answer": results["answer"].astype(str).str.len().gt(0).all(),
        "Evaluation results saved": result_path.exists(),
        "Summary saved": summary_path.exists(),
    }

    print("\nVALIDATION")
    for label, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")

    if not all(checks.values()):
        print("\nOne or more checks failed. Please share the full output.")
        return 1

    print("\nPART D1 COMPLETED — proceed only after reviewing these results.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
