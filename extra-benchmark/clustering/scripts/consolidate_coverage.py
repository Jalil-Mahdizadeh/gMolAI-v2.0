#!/usr/bin/env python3
"""Consolidate chemistry, adapter, common-support, and vector-validity accounting."""

from __future__ import annotations

from collections import Counter
import json

from benchmark_io import BENCHMARK_DIR, atomic_write_json, load_json, read_panel_tsv, sha256_file, write_csv


MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")
SCREENED = MODELS[1:]


def reason_category(reason: str) -> str:
    if "tokenized SMILES" in reason and "supports at most 109" in reason:
        return "molai_token_length_exceeds_109"
    if "unsupported MolAI character" in reason:
        return "molai_unsupported_character"
    if "exceeds_202" in reason:
        return "token_length_exceeds_202"
    if "tokenization_not_lossless" in reason:
        return "tokenization_not_lossless"
    if "unknown_token" in reason:
        return "unknown_token"
    return reason.split(":", 1)[0][:120]


def main() -> None:
    classy_candidates = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_candidates.tsv")
    classy_final = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "classyfire_common.tsv")
    qmugs_attempt = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_attempt_060000.tsv")
    qmugs_final = read_panel_tsv(BENCHMARK_DIR / "inputs" / "prepared" / "qmugs_common.tsv")
    class_state = load_json(BENCHMARK_DIR / "state" / "classyfire_common.json")
    qmugs_state = load_json(BENCHMARK_DIR / "state" / "qmugs_common.json")
    prep_class = load_json(BENCHMARK_DIR / "audit" / "classyfire_preparation.json")
    prep_qmugs = load_json(BENCHMARK_DIR / "audit" / "qmugs_preparation.json")
    rows = []
    reason_rows = []
    labels = sorted({row["subclass"] for row in classy_candidates})
    by_label_indices = {
        label: {index for index, row in enumerate(classy_candidates) if row["subclass"] == label}
        for label in labels
    }
    screens = {}
    accepted = {"gmolai": set(range(len(classy_candidates)))}
    for model in SCREENED:
        report = load_json(BENCHMARK_DIR / "artifacts" / "screens" / f"{model}-classyfire.json")
        screens[model] = report
        accepted[model] = set(map(int, report["accepted_indices"]))
        categories = Counter(reason_category(record["reason"]) for record in report["rejections"])
        for category, count in sorted(categories.items()):
            reason_rows.append({"benchmark": "classyfire", "model": model, "reason_category": category, "count": count})
    for label in labels:
        candidate_indices = by_label_indices[label]
        candidate_count = len(candidate_indices)
        for model in MODELS:
            model_accepted = len(candidate_indices & accepted[model])
            validity = load_json(BENCHMARK_DIR / "audit" / f"embedding-{model}-classyfire.json")
            rows.append({
                "benchmark": "classyfire", "subclass": label, "model": model,
                "source_rows": 3000, "chemistry_policy_accepted_unique": candidate_count,
                "chemistry_rejected_or_removed": 3000 - candidate_count,
                "duplicate_occurrences_removed": 0, "conflicting_identities_removed": 0,
                "adapter_attempted": candidate_count, "adapter_accepted": model_accepted,
                "adapter_rejected": candidate_count - model_accepted,
                "all_model_common_before_balance": int(class_state["common_counts_before_balance"][label]),
                "frozen_final_common": int(class_state["balance_per_subclass"]),
                "adapter_accepted_not_in_final_common": model_accepted - int(class_state["balance_per_subclass"]),
                "nonfinite_vectors_on_final_common": int(validity["nonfinite_vectors"]),
                "zero_norm_vectors_on_final_common": int(validity["zero_norm_vectors"]),
            })
    qmugs_accepted = {"gmolai": set(range(len(qmugs_attempt)))}
    for model in SCREENED:
        report = load_json(BENCHMARK_DIR / "artifacts" / "screens" / f"{model}-qmugs_060000.json")
        qmugs_accepted[model] = set(map(int, report["accepted_indices"]))
        categories = Counter(reason_category(record["reason"]) for record in report["rejections"])
        for category, count in sorted(categories.items()):
            reason_rows.append({"benchmark": "qmugs", "model": model, "reason_category": category, "count": count})
    for model in MODELS:
        validity = load_json(BENCHMARK_DIR / "audit" / f"embedding-{model}-qmugs.json")
        model_accepted = len(qmugs_accepted[model])
        rows.append({
            "benchmark": "qmugs", "subclass": "ALL", "model": model,
            "source_rows": int(prep_qmugs["source_rows"]),
            "chemistry_policy_accepted_unique": int(prep_qmugs["eligible_unique"]),
            "chemistry_rejected_or_removed": int(prep_qmugs["source_rows"]) - int(prep_qmugs["eligible_unique"]),
            "duplicate_occurrences_removed": int(prep_qmugs["duplicate_same_chembl_identity_occurrences_removed"]),
            "conflicting_identities_removed": int(prep_qmugs["cross_chembl_conflicting_identities_removed"]),
            "adapter_attempted": len(qmugs_attempt), "adapter_accepted": model_accepted,
            "adapter_rejected": len(qmugs_attempt) - model_accepted,
            "all_model_common_before_balance": int(qmugs_state["all_model_common_before_truncation"]),
            "frozen_final_common": len(qmugs_final),
            "adapter_accepted_not_in_final_common": model_accepted - len(qmugs_final),
            "nonfinite_vectors_on_final_common": int(validity["nonfinite_vectors"]),
            "zero_norm_vectors_on_final_common": int(validity["zero_norm_vectors"]),
        })
    output = BENCHMARK_DIR / "outputs" / "tables" / "full_coverage_accounting.csv"
    write_csv(output, rows, tuple(rows[0]))
    reasons = BENCHMARK_DIR / "outputs" / "tables" / "adapter_rejection_reasons.csv"
    if reason_rows:
        write_csv(reasons, reason_rows, tuple(reason_rows[0]))
    report = {
        "schema_version": 1, "status": "ok", "rows": len(rows),
        "coverage_table": str(output), "coverage_table_sha256": sha256_file(output),
        "rejection_reason_table": str(reasons), "rejection_reason_table_sha256": sha256_file(reasons),
        "classyfire_preparation_sha256": sha256_file(BENCHMARK_DIR / "audit" / "classyfire_preparation.json"),
        "qmugs_preparation_sha256": sha256_file(BENCHMARK_DIR / "audit" / "qmugs_preparation.json"),
    }
    atomic_write_json(BENCHMARK_DIR / "audit" / "coverage_accounting.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

