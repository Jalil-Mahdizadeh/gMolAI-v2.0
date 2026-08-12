#!/usr/bin/env python3
"""Run the existing common representation probes and summarize results."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys

from benchmark_io import (
    BENCHMARK_DIR,
    REPOSITORY_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_protocol,
    sha256_file,
)


SOURCE_DIR = REPOSITORY_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from gmolai_retrain.probes import run_representation_probes


MODELS = ("gmolai", "morgan", "molai", "molformer", "smi_ted", "molclr_gin", "kermt_v2")


def summary_row(model: str, result: dict, dimension: int) -> dict:
    diagnostics = result["embedding_diagnostics"]
    topology = result["held_out_linear_probe"]
    scaffold_topology = result["scaffold_disjoint_linear_probe"]
    similarity = result["similarity"]
    clustering = result["clustering"]
    latent_cluster = clustering["latent_spherical_kmeans"]
    return {
        "model": model,
        "dimension": dimension,
        "train_rows": topology["train_graphs"],
        "test_rows": diagnostics["graphs"],
        "effective_rank": diagnostics["effective_rank"],
        "effective_rank_fraction": diagnostics["effective_rank_ratio"],
        "participation_ratio": diagnostics["participation_ratio"],
        "participation_ratio_fraction": diagnostics[
            "participation_ratio_fraction"
        ],
        "median_coordinate_std": diagnostics["median_coordinate_std"],
        "minimum_coordinate_std": diagnostics["minimum_coordinate_std"],
        "top_eigenvalue_fraction": diagnostics["top_eigenvalue_fraction"],
        "topology_mean_r2": topology["mean_r2"],
        "topology_median_r2": topology["median_r2"],
        "topology_mean_standardized_mae": topology["mean_standardized_mae"],
        "scaffold_disjoint_mean_r2": scaffold_topology["mean_r2"],
        "scaffold_disjoint_fraction": result[
            "scaffold_disjoint_validation_fraction"
        ],
        "morgan_recall_at_10": similarity["latent_to_morgan_recall_at_10"],
        "cosine_tanimoto_spearman": similarity[
            "latent_cosine_vs_morgan_spearman"
        ],
        "neighbor_tanimoto_enrichment": similarity[
            "neighbor_tanimoto_enrichment"
        ],
        "scaffold_neighbor_purity_at_10": similarity[
            "scaffold_neighbor_purity_at_10"
        ],
        "scaffold_purity_enrichment": similarity["scaffold_purity_enrichment"],
        "clustering_rows": clustering["graphs"],
        "clustering_scaffolds": clustering["scaffold_clusters"],
        "clustering_ari": latent_cluster["adjusted_rand_index"],
        "clustering_ari_std": latent_cluster["adjusted_rand_index_std"],
        "clustering_nmi": latent_cluster["normalized_mutual_information"],
        "clustering_nmi_std": latent_cluster[
            "normalized_mutual_information_std"
        ],
        "clustering_homogeneity": latent_cluster["homogeneity"],
        "clustering_completeness": latent_cluster["completeness"],
    }


def main() -> None:
    protocol = load_protocol()
    work_dir = Path(protocol["repository"]["work_dir"])
    results: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for model in MODELS:
        train = BENCHMARK_DIR / "outputs" / "payloads" / f"{model}-train.pt"
        test = BENCHMARK_DIR / "outputs" / "payloads" / f"{model}-test.pt"
        output = BENCHMARK_DIR / "outputs" / "probes" / f"{model}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        results[model] = run_representation_probes(
            train_embeddings=train,
            validation_embeddings=test,
            work_dir=work_dir,
            output=output,
            similarity_graphs=int(protocol["diagnostics"]["similarity_rows"]),
            seed=int(protocol["diagnostics"]["seed"]),
        )
        hashes[model] = sha256_file(output)

    baseline = results[MODELS[0]]["training_descriptor_baseline"]
    morgan_clustering = results[MODELS[0]]["clustering"]["morgan_spherical_kmeans"]
    for model in MODELS[1:]:
        if results[model]["training_descriptor_baseline"] != baseline:
            raise RuntimeError(f"Shared descriptor baseline changed for {model}")
        if results[model]["clustering"]["morgan_spherical_kmeans"] != morgan_clustering:
            raise RuntimeError(f"Shared Morgan clustering baseline changed for {model}")

    rows = [
        summary_row(
            model,
            results[model],
            int(protocol["comparators"][model]["dimension"]),
        )
        for model in MODELS
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary_csv = BENCHMARK_DIR / "outputs" / "test_partition_summary.csv"
    atomic_write_text(summary_csv, stream.getvalue())

    coverage_train = load_json(BENCHMARK_DIR / "state" / "common_train.json")
    coverage_test = load_json(BENCHMARK_DIR / "state" / "common_test.json")
    summary_json = BENCHMARK_DIR / "outputs" / "test_partition_summary.json"
    payload = {
        "schema_version": 1,
        "status": "ok",
        "execution": "inference_only",
        "evidence_source": "locked_internal_pretraining_test_partition",
        "test_population_rows": 1088766,
        "attempted_test_panel_rows": coverage_test["attempted_rows"],
        "common_test_rows": coverage_test["common_rows"],
        "attempted_train_probe_rows": coverage_train["attempted_rows"],
        "common_train_probe_rows": coverage_train["common_rows"],
        "coverage": {
            "train": coverage_train["models"],
            "test": coverage_test["models"],
        },
        "shared_descriptor_baseline": baseline,
        "shared_morgan_clustering_baseline": morgan_clustering,
        "models": {row["model"]: row for row in rows},
        "probe_artifact_sha256": hashes,
        "interpretation_limits": [
            "Morgan-neighbour agreement measures similarity to Morgan geometry, not absolute quality.",
            "Raw effective rank is dimension-dependent; use the reported rank fraction alongside it.",
            "This retrospective additive comparison cannot change the frozen gMolAI checkpoint or calibrator.",
            "No endpoint labels are available or used in the locked internal partition.",
        ],
    }
    atomic_write_json(summary_json, payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_csv": str(summary_csv.relative_to(REPOSITORY_ROOT)),
                "summary_csv_sha256": sha256_file(summary_csv),
                "summary_json": str(summary_json.relative_to(REPOSITORY_ROOT)),
                "summary_json_sha256": sha256_file(summary_json),
                "common_train_rows": coverage_train["common_rows"],
                "common_test_rows": coverage_test["common_rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
