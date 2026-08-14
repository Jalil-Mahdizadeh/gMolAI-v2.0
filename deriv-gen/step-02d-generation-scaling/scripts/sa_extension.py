#!/usr/bin/env python3
"""Additive synthetic-accessibility analysis for the completed Step-2d study."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rdkit  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Contrib.SA_Score import sascorer  # noqa: E402

from common import (  # noqa: E402
    REPO_ROOT,
    STEP_ROOT,
    atomic_numpy_savez,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    load_json,
    numeric_summary,
    sha256_file,
    utc_now,
    write_hash_ledger,
)


RDLogger.DisableLog("rdApp.*")
EXTENSION_CONFIG = Path("config/sa_extension.json")
EXTENSION_PROTOCOL = Path("PROTOCOL_SA_EXTENSION.md")
EXTENSION_SOURCES = (
    Path("scripts/sa_extension.py"),
    Path("scripts/run_sa_extension.sh"),
    EXTENSION_CONFIG,
    EXTENSION_PROTOCOL,
)
ALLOWED_BASE_MUTATIONS = {
    "README.md",
    "RESULTS.md",
    "DECISION.md",
    "outputs/figures/quality_locality_diversity_scaling.png",
    "outputs/SHA256SUMS",
    "state/REPORT_COMPLETE.json",
    "state/COMPLETE.json",
}
BASE_DOC_SNAPSHOTS = {
    "README.md": "inputs/sa_extension_base_README.md",
    "RESULTS.md": "inputs/sa_extension_base_RESULTS.md",
    "DECISION.md": "inputs/sa_extension_base_DECISION.md",
}
SA_OUTPUTS = (
    "outputs/tables/final_sa_scores_by_identity.parquet",
    "outputs/tables/final_candidate_sa_comparison.parquet",
    "outputs/tables/final_seed_sa_metrics.parquet",
    "outputs/tables/final_sa_summary_by_budget.csv",
    "outputs/tables/final_sa_summary_by_category_budget.csv",
    "outputs/tables/final_sa_bootstrap.csv",
    "outputs/raw/final_sa_joint_histogram_budget1000.npz",
    "outputs/sa_extension_analysis_summary.json",
    "outputs/sa_extension_decision.json",
    "outputs/sa_extension_verification.json",
    "outputs/figures/synthetic_accessibility_scaling.png",
    "outputs/figures/synthetic_accessibility_by_category.png",
    "outputs/figures/synthetic_accessibility_seed_candidate_density.png",
)


def extension_config(root: Path) -> dict[str, Any]:
    return load_json(root / EXTENSION_CONFIG)


def parse_hash_ledger(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        if relative in result:
            raise RuntimeError(f"Duplicate hash-ledger entry: {relative}")
        result[relative] = digest
    return result


def assert_source_hashes(root: Path, state: dict[str, Any]) -> None:
    for relative, expected in state["registered_extension_source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Registered SA-extension source changed: {relative}")


def assert_base_protected(root: Path, state: dict[str, Any]) -> None:
    for relative, expected in state["protected_base_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Protected base Step-2d artifact changed: {relative}")


def scorer_paths() -> tuple[Path, Path]:
    source = Path(inspect.getfile(sascorer)).resolve()
    fragments = source.with_name("fpscores.pkl.gz")
    if not source.is_file() or not fragments.is_file():
        raise FileNotFoundError("Pinned RDKit SA scorer or fragment model is missing")
    return source, fragments


def register_extension(root: Path, repo_root: Path) -> None:
    state_path = root / "state/SA_EXTENSION_REGISTERED.json"
    if state_path.exists():
        state = load_json(state_path)
        assert_source_hashes(root, state)
        assert_base_protected(root, state)
        print(json.dumps(state, sort_keys=True))
        return
    if not (root / "state/COMPLETE.json").is_file():
        raise RuntimeError("Base Step 2d must be complete before the SA extension")
    base_verification = load_json(root / "outputs/verification.json")
    if base_verification.get("status") != "passed":
        raise RuntimeError("Base Step-2d verification is not passed")
    if base_verification.get("test_rows") != 0 or base_verification.get("endpoint_labels_used") is not False:
        raise RuntimeError("Base scientific boundary changed")
    if any((root / relative).exists() for relative in SA_OUTPUTS):
        raise RuntimeError("Refusing to register after SA-extension outputs exist")

    base_ledger_path = root / "outputs/SHA256SUMS"
    base_ledger = parse_hash_ledger(base_ledger_path)
    failed = []
    for relative, expected in base_ledger.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            failed.append(relative)
    if failed:
        raise RuntimeError(f"Base Step-2d ledger failed before extension: {failed[:10]}")
    protected = {
        relative: digest
        for relative, digest in base_ledger.items()
        if relative not in ALLOWED_BASE_MUTATIONS
    }
    for source in EXTENSION_SOURCES:
        if not (root / source).is_file():
            raise FileNotFoundError(root / source)
    source_hashes = {
        relative.as_posix(): sha256_file(root / relative)
        for relative in EXTENSION_SOURCES
    }
    snapshot_hashes: dict[str, str] = {}
    for original, snapshot in BASE_DOC_SNAPSHOTS.items():
        original_path = root / original
        snapshot_path = root / snapshot
        atomic_write_text(snapshot_path, original_path.read_text(encoding="utf-8"), root)
        snapshot_hashes[snapshot] = sha256_file(snapshot_path)

    score_source, fragment_model = scorer_paths()
    structures = root / "intermediate/final_unique_molecules.parquet"
    candidates = root / "outputs/tables/final_candidate_characterization.parquet"
    cfg = extension_config(root)
    state = {
        "schema_version": 1,
        "status": "registered_before_sa_scoring",
        "registered_at": utc_now(),
        "study_id": cfg["study_id"],
        "amendment_status": cfg["status"],
        "registered_extension_source_sha256": source_hashes,
        "extension_config_sha256": sha256_file(root / EXTENSION_CONFIG),
        "extension_protocol_sha256": sha256_file(root / EXTENSION_PROTOCOL),
        "base_ledger_sha256": sha256_file(base_ledger_path),
        "protected_base_sha256": protected,
        "allowed_base_mutations": sorted(ALLOWED_BASE_MUTATIONS),
        "base_document_snapshot_sha256": snapshot_hashes,
        "identity_input_sha256": sha256_file(structures),
        "pair_input_sha256": sha256_file(candidates),
        "rdkit_version": rdkit.__version__,
        "sa_scorer_source": str(score_source),
        "sa_scorer_source_sha256": sha256_file(score_source),
        "sa_fragment_model": str(fragment_model),
        "sa_fragment_model_sha256": sha256_file(fragment_model),
        "repo_root": str(repo_root),
        "candidate_generation_changed": False,
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "property_optimization": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


def _score_molecule(smiles: str) -> float:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Registered canonical identity did not parse: {smiles}")
    value = sascorer.calculateScore(molecule)
    if value is None or not math.isfinite(float(value)) or not (1.0 <= float(value) <= 10.0):
        raise ValueError(f"Invalid SA score for {smiles}: {value}")
    return float(value)


def _score_chunk(smiles: list[str]) -> list[float]:
    RDLogger.DisableLog("rdApp.*")
    return [_score_molecule(value) for value in smiles]


def component_test(root: Path) -> None:
    state_path = root / "state/SA_EXTENSION_COMPONENT_TESTS.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    registered = load_json(root / "state/SA_EXTENSION_REGISTERED.json")
    assert_source_hashes(root, registered)
    assert_base_protected(root, registered)
    references = {
        "ethanol": "CCO",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    }
    first = {name: _score_molecule(smiles) for name, smiles in references.items()}
    second = {name: _score_molecule(smiles) for name, smiles in references.items()}
    if first != second:
        raise RuntimeError("SA scorer is not deterministic on reference molecules")
    if not all(1.0 <= value <= 10.0 for value in first.values()):
        raise RuntimeError("SA reference score left the documented range")
    state = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "reference_scores": first,
        "deterministic_repeat": True,
        "rdkit_version": registered["rdkit_version"],
        "sa_scorer_source_sha256": registered["sa_scorer_source_sha256"],
        "sa_fragment_model_sha256": registered["sa_fragment_model_sha256"],
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _prefixed_summary(record: dict[str, Any], prefix: str, values: np.ndarray) -> None:
    for key, value in numeric_summary(values).items():
        record[f"{prefix}_{key}"] = value


def _bootstrap_seed_macro(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    arrays = {column: frame[column].to_numpy(dtype=np.float64) for column in columns}
    n = len(frame)
    if n == 0:
        raise RuntimeError("Cannot bootstrap an empty seed table")
    rng = np.random.default_rng(seed)
    estimates = {column: np.empty(resamples, dtype=np.float64) for column in columns}
    for draw in range(resamples):
        indices = rng.integers(0, n, size=n)
        for column, values in arrays.items():
            estimates[column][draw] = float(values[indices].mean())
    rows = []
    for column, values in arrays.items():
        low, high = np.quantile(estimates[column], [0.025, 0.975])
        rows.append(
            {
                "metric": column,
                "seed_rows": n,
                "point_estimate": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "resamples": resamples,
                "bootstrap_unit": "seed",
                "bootstrap_seed": seed,
            }
        )
    return rows


def _pair_summary(frame: pd.DataFrame, *, seed_total: int) -> dict[str, Any]:
    if frame.empty:
        raise RuntimeError("Synthetic-accessibility summary population is empty")
    record: dict[str, Any] = {
        "pair_rows": int(len(frame)),
        "distinct_candidate_identities": int(frame["candidate_hash"].nunique()),
        "seed_rows": int(frame["query_position"].nunique()),
        "seed_coverage": float(frame["query_position"].nunique() / seed_total),
        "fraction_no_harder_than_seed": float(frame["is_no_harder_than_seed"].mean()),
        "fraction_not_more_than_half_point_harder": float(
            frame["is_not_more_than_half_point_harder"].mean()
        ),
        "fraction_absolute_delta_within_half_point": float(
            frame["is_absolute_delta_within_half_point"].mean()
        ),
        "fraction_at_least_one_point_easier": float(
            frame["is_at_least_one_point_easier"].mean()
        ),
        "fraction_at_least_one_point_harder": float(
            frame["is_at_least_one_point_harder"].mean()
        ),
    }
    _prefixed_summary(
        record, "candidate_sa", frame["candidate_sa_score"].to_numpy(dtype=np.float64)
    )
    _prefixed_summary(
        record, "matched_seed_sa", frame["seed_sa_score"].to_numpy(dtype=np.float64)
    )
    _prefixed_summary(
        record, "delta_sa", frame["delta_sa_candidate_minus_seed"].to_numpy(dtype=np.float64)
    )
    return record


def analyze_extension(root: Path, workers: int) -> None:
    state_path = root / "state/SA_EXTENSION_ANALYSIS_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    registered = load_json(root / "state/SA_EXTENSION_REGISTERED.json")
    assert_source_hashes(root, registered)
    assert_base_protected(root, registered)
    if not (root / "state/SA_EXTENSION_COMPONENT_TESTS.json").is_file():
        raise RuntimeError("SA-extension component tests must pass before analysis")
    cfg = extension_config(root)
    started = time.monotonic()
    structures_path = root / "intermediate/final_unique_molecules.parquet"
    candidates_path = root / "outputs/tables/final_candidate_characterization.parquet"
    if sha256_file(structures_path) != registered["identity_input_sha256"]:
        raise RuntimeError("Registered SA identity input changed")
    if sha256_file(candidates_path) != registered["pair_input_sha256"]:
        raise RuntimeError("Registered SA pair input changed")

    structures = pd.read_parquet(structures_path)
    expected_indices = np.arange(len(structures), dtype=np.int64)
    if not np.array_equal(structures["structure_index"].to_numpy(dtype=np.int64), expected_indices):
        raise RuntimeError("Global structure indices are not contiguous and ordered")
    if structures["molecule_hash"].duplicated().any() or structures["canonical_smiles"].isna().any():
        raise RuntimeError("Global structure table is not identity-unique and complete")
    smiles = structures["canonical_smiles"].astype(str).tolist()
    workers = max(1, min(int(workers), mp.cpu_count()))
    block_size = max(1024, int(cfg["execution"]["chunksize"]) * 16)
    block_count = math.ceil(len(smiles) / block_size)
    print(
        f"SA extension: scoring {len(smiles):,} global identities with {workers} workers",
        flush=True,
    )
    sascorer.readFragmentScores()
    scores: list[float] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for block_index, block_scores in enumerate(
            executor.map(_score_chunk, _chunks(smiles, block_size), chunksize=1), start=1
        ):
            scores.extend(block_scores)
            if block_index % 25 == 0 or block_index == block_count:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"  scored {len(scores):,}/{len(smiles):,} identities "
                    f"({len(scores) / elapsed:,.0f} mol/s)",
                    flush=True,
                )
    score_array = np.asarray(scores, dtype=np.float64)
    if len(score_array) != len(structures):
        raise RuntimeError("SA scoring lost molecular identities")
    if not np.isfinite(score_array).all() or (score_array < 1.0).any() or (score_array > 10.0).any():
        raise RuntimeError("SA scores violate the registered finite [1, 10] range")
    identity_scores = structures[
        ["structure_index", "molecule_hash", "canonical_smiles", "scaffold", "heavy_atom_count"]
    ].copy()
    identity_scores["sa_score"] = score_array
    identity_path = root / "outputs/tables/final_sa_scores_by_identity.parquet"
    atomic_write_parquet(identity_path, identity_scores, root)

    candidate_columns = [
        "strategy",
        "query_position",
        "target_index",
        "candidate_hash",
        "seed_hash",
        "candidate_structure_index",
        "first_proposal_rank",
        "chemical_category",
        "is_seed_identity",
        "is_genuine_nonseed",
        "is_one_cut_mmp",
        "retains_nonempty_seed_scaffold",
        "is_novel_to_decoder_training",
        "is_useful_local",
    ]
    candidates = pd.read_parquet(candidates_path, columns=candidate_columns)
    if candidates.duplicated(["strategy", "query_position", "candidate_hash"]).any():
        raise RuntimeError("Seed-candidate identity rows are not unique")
    candidate_indices = candidates["candidate_structure_index"].to_numpy(dtype=np.int64)
    if (candidate_indices < 0).any() or (candidate_indices >= len(score_array)).any():
        raise RuntimeError("Candidate structure index lies outside the score table")
    hash_to_index = structures.set_index("molecule_hash")["structure_index"]
    seed_indices = candidates["seed_hash"].map(hash_to_index)
    if seed_indices.isna().any():
        raise RuntimeError("A final seed is absent from the global score table")
    candidates["seed_structure_index"] = seed_indices.astype(np.int64)
    candidates["candidate_sa_score"] = score_array[candidate_indices]
    candidates["seed_sa_score"] = score_array[
        candidates["seed_structure_index"].to_numpy(dtype=np.int64)
    ]
    candidates["delta_sa_candidate_minus_seed"] = (
        candidates["candidate_sa_score"] - candidates["seed_sa_score"]
    )
    exact_delta = candidates.loc[
        candidates["is_seed_identity"].astype(bool), "delta_sa_candidate_minus_seed"
    ].to_numpy(dtype=np.float64)
    if len(exact_delta) == 0 or not np.array_equal(exact_delta, np.zeros_like(exact_delta)):
        raise RuntimeError("Exact seed identities are not exact zero-delta SA controls")
    thresholds = cfg["comparisons"]
    delta = candidates["delta_sa_candidate_minus_seed"]
    candidates["is_no_harder_than_seed"] = delta <= float(thresholds["no_harder_delta_max"])
    candidates["is_not_more_than_half_point_harder"] = delta <= float(
        thresholds["not_more_than_half_point_harder_delta_max"]
    )
    candidates["is_absolute_delta_within_half_point"] = delta.abs() <= float(
        thresholds["absolute_within_half_point_delta_max"]
    )
    candidates["is_at_least_one_point_easier"] = delta <= float(
        thresholds["substantially_easier_delta_max"]
    )
    candidates["is_at_least_one_point_harder"] = delta >= float(
        thresholds["substantially_harder_delta_min"]
    )
    comparison_path = root / "outputs/tables/final_candidate_sa_comparison.parquet"
    atomic_write_parquet(comparison_path, candidates, root)

    genuine = candidates.loc[candidates["is_genuine_nonseed"].astype(bool)].copy()
    seed_total = int(candidates["query_position"].nunique())
    budgets = [int(value) for value in cfg["population"]["budgets"]]
    budget_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for budget in budgets:
        current = genuine.loc[genuine["first_proposal_rank"] <= budget].copy()
        row = {"strategy": str(current["strategy"].iloc[0]), "budget": budget}
        row.update(_pair_summary(current, seed_total=seed_total))
        current["no_harder_float"] = current["is_no_harder_than_seed"].astype(np.float64)
        current["not_more_than_half_harder_float"] = current[
            "is_not_more_than_half_point_harder"
        ].astype(np.float64)
        current["absolute_within_half_float"] = current[
            "is_absolute_delta_within_half_point"
        ].astype(np.float64)
        current["one_point_easier_float"] = current[
            "is_at_least_one_point_easier"
        ].astype(np.float64)
        current["one_point_harder_float"] = current[
            "is_at_least_one_point_harder"
        ].astype(np.float64)
        seed_metrics = (
            current.groupby(["strategy", "query_position"], sort=True, as_index=False)
            .agg(
                seed_sa_score=("seed_sa_score", "first"),
                candidate_count=("candidate_hash", "size"),
                candidate_sa_mean=("candidate_sa_score", "mean"),
                candidate_sa_median=("candidate_sa_score", "median"),
                candidate_sa_minimum=("candidate_sa_score", "min"),
                delta_sa_mean=("delta_sa_candidate_minus_seed", "mean"),
                delta_sa_median=("delta_sa_candidate_minus_seed", "median"),
                delta_sa_minimum=("delta_sa_candidate_minus_seed", "min"),
                delta_sa_maximum=("delta_sa_candidate_minus_seed", "max"),
                fraction_no_harder=("no_harder_float", "mean"),
                fraction_not_more_than_half_point_harder=(
                    "not_more_than_half_harder_float",
                    "mean",
                ),
                fraction_absolute_delta_within_half_point=(
                    "absolute_within_half_float",
                    "mean",
                ),
                fraction_at_least_one_point_easier=("one_point_easier_float", "mean"),
                fraction_at_least_one_point_harder=("one_point_harder_float", "mean"),
            )
        )
        seed_metrics.insert(2, "budget", budget)
        seed_frames.append(seed_metrics)
        row.update(
            {
                "seed_macro_mean_delta_sa": float(seed_metrics["delta_sa_mean"].mean()),
                "seed_macro_median_of_seed_median_delta_sa": float(
                    seed_metrics["delta_sa_median"].median()
                ),
                "seed_macro_mean_fraction_no_harder": float(
                    seed_metrics["fraction_no_harder"].mean()
                ),
                "seed_fraction_with_any_no_harder_candidate": float(
                    (
                        seed_metrics["fraction_no_harder"]
                        * seed_metrics["candidate_count"]
                        >= 1.0 - 1e-12
                    ).mean()
                ),
            }
        )
        bootstrap = _bootstrap_seed_macro(
            seed_metrics,
            ["delta_sa_mean", "fraction_no_harder"],
            resamples=int(thresholds["bootstrap_resamples"]),
            seed=int(thresholds["bootstrap_seed"]) + budget,
        )
        for item in bootstrap:
            item.update({"strategy": row["strategy"], "budget": budget})
            bootstrap_rows.append(item)
            if item["metric"] == "delta_sa_mean":
                row["seed_macro_mean_delta_sa_ci_low"] = item["ci_low"]
                row["seed_macro_mean_delta_sa_ci_high"] = item["ci_high"]
            elif item["metric"] == "fraction_no_harder":
                row["seed_macro_fraction_no_harder_ci_low"] = item["ci_low"]
                row["seed_macro_fraction_no_harder_ci_high"] = item["ci_high"]
        budget_rows.append(row)
        for category in thresholds["categories"]:
            subset = current.loc[current["chemical_category"] == category]
            category_row = {
                "strategy": row["strategy"],
                "budget": budget,
                "chemical_category": category,
            }
            category_row.update(_pair_summary(subset, seed_total=seed_total))
            category_rows.append(category_row)

    seed_table = pd.concat(seed_frames, ignore_index=True).sort_values(
        ["strategy", "query_position", "budget"], ignore_index=True
    )
    budget_table = pd.DataFrame(budget_rows).sort_values("budget", ignore_index=True)
    category_table = pd.DataFrame(category_rows).sort_values(
        ["budget", "chemical_category"], ignore_index=True
    )
    bootstrap_table = pd.DataFrame(bootstrap_rows).sort_values(
        ["budget", "metric"], ignore_index=True
    )
    seed_path = root / "outputs/tables/final_seed_sa_metrics.parquet"
    budget_path = root / "outputs/tables/final_sa_summary_by_budget.csv"
    category_path = root / "outputs/tables/final_sa_summary_by_category_budget.csv"
    bootstrap_path = root / "outputs/tables/final_sa_bootstrap.csv"
    atomic_write_parquet(seed_path, seed_table, root)
    atomic_write_csv(budget_path, budget_table, root)
    atomic_write_csv(category_path, category_table, root)
    atomic_write_csv(bootstrap_path, bootstrap_table, root)

    maximum = genuine.loc[genuine["first_proposal_rank"] <= max(budgets)]
    edges = np.linspace(
        float(cfg["plots"]["joint_sa_range"][0]),
        float(cfg["plots"]["joint_sa_range"][1]),
        int(cfg["plots"]["joint_sa_histogram_bins"]) + 1,
    )
    joint_counts, seed_edges, candidate_edges = np.histogram2d(
        maximum["seed_sa_score"].to_numpy(dtype=np.float64),
        maximum["candidate_sa_score"].to_numpy(dtype=np.float64),
        bins=(edges, edges),
    )
    joint_path = root / "outputs/raw/final_sa_joint_histogram_budget1000.npz"
    atomic_numpy_savez(
        joint_path,
        root,
        counts=joint_counts.astype(np.int64),
        seed_edges=seed_edges,
        candidate_edges=candidate_edges,
    )

    outputs = {
        "identity_scores": identity_path,
        "candidate_comparison": comparison_path,
        "seed_metrics": seed_path,
        "budget_summary": budget_path,
        "category_summary": category_path,
        "bootstrap": bootstrap_path,
        "joint_histogram": joint_path,
    }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "study_id": cfg["study_id"],
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "workers": workers,
        "rdkit_version": registered["rdkit_version"],
        "sa_scorer_source_sha256": registered["sa_scorer_source_sha256"],
        "sa_fragment_model_sha256": registered["sa_fragment_model_sha256"],
        "global_identity_rows": len(identity_scores),
        "seed_candidate_rows": len(candidates),
        "genuine_nonseed_pair_rows_at_1000": len(maximum),
        "seed_rows": seed_total,
        "budgets": budgets,
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "candidate_generation_changed": False,
        "encoder_training": False,
        "decoder_training": False,
        "latent_perturbation": False,
        "mmp_direction_editing": False,
        "property_optimization": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    summary_path = root / "outputs/sa_extension_analysis_summary.json"
    atomic_write_json(summary_path, summary, root)
    state = {
        "schema_version": 1,
        "status": "complete",
        "completed_at": utc_now(),
        "summary_sha256": sha256_file(summary_path),
        "output_sha256": summary["output_sha256"],
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))

def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _append_section(base: str, marker: str, section: str) -> str:
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if start in base or end in base:
        raise RuntimeError(f"Base document already contains extension marker {marker}")
    return base.rstrip() + "\n\n" + start + "\n" + section.strip() + "\n" + end + "\n"


def report_extension(root: Path) -> None:
    state_path = root / "state/SA_EXTENSION_REPORT_COMPLETE.json"
    if state_path.exists():
        print(state_path.read_text(encoding="utf-8"))
        return
    registered = load_json(root / "state/SA_EXTENSION_REGISTERED.json")
    assert_source_hashes(root, registered)
    assert_base_protected(root, registered)
    if not (root / "state/SA_EXTENSION_ANALYSIS_COMPLETE.json").is_file():
        raise RuntimeError("SA analysis must complete before reporting")
    cfg = extension_config(root)
    table_root, figure_root = root / "outputs/tables", root / "outputs/figures"
    summary = pd.read_csv(table_root / "final_sa_summary_by_budget.csv").sort_values("budget")
    category = pd.read_csv(table_root / "final_sa_summary_by_category_budget.csv")
    base_summary = pd.read_csv(table_root / "final_budget_summary.csv").sort_values("budget")
    budgets = [int(value) for value in cfg["population"]["budgets"]]
    if summary["budget"].astype(int).tolist() != budgets:
        raise RuntimeError("SA report budgets differ from the frozen extension protocol")
    maximum = summary.loc[summary["budget"] == max(budgets)].iloc[0]

    fig = plt.figure(figsize=(7.4, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.25], hspace=0.08)
    first = fig.add_subplot(grid[0])
    first.plot(base_summary["budget"], base_summary["raw_policy_acceptance_fraction"],
               marker="o", color="#31a354", linewidth=2.2, label="policy accepted", zorder=2)
    first.plot(base_summary["budget"], base_summary["raw_valid_fraction"],
               marker="o", markersize=9, markerfacecolor="none", markeredgewidth=1.8,
               linestyle="--", color="#2b8cbe", linewidth=1.8, label="RDKit valid", zorder=4)
    first.plot(base_summary["budget"], base_summary["novel_fraction_among_genuine_nonseed"],
               marker="o", color="#756bb1", linewidth=2.0, label="novel among non-seed")
    first.set_ylim(0.0, 1.02)
    first.set_ylabel("Fraction")
    first.tick_params(labelbottom=False)
    second = first.twinx()
    second.plot(base_summary["budget"], base_summary["median_seed_candidate_morgan_nonseed"],
                marker="s", linestyle="--", color="#e6550d", label="median seed similarity")
    second.plot(base_summary["budget"], base_summary["within_pairwise_morgan_weighted_mean"],
                marker="s", linestyle=":", color="#636363", label="within-set similarity")
    second.set_ylim(0.0, 1.02)
    second.set_ylabel("Morgan/Tanimoto")
    lines = first.get_lines() + second.get_lines()
    first.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="lower left")
    first.set_title("Validity, novelty, locality, and diversity")
    gap = fig.add_subplot(grid[1], sharex=first)
    gap_values = 10000.0 * (base_summary["raw_valid_fraction"] -
                            base_summary["raw_policy_acceptance_fraction"])
    gap.plot(base_summary["budget"], gap_values, marker="o", color="#2b8cbe")
    gap.fill_between(base_summary["budget"], 0.0, gap_values, color="#9ecae1", alpha=0.35)
    gap.set_ylabel("Valid - policy\n(basis points)")
    gap.set_xlabel("Raw proposal budget per seed")
    gap.grid(axis="y", alpha=0.25)
    quality_path = figure_root / "quality_locality_diversity_scaling.png"
    _save_figure(fig, quality_path)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)
    x = summary["budget"].to_numpy(dtype=np.float64)
    for prefix, color, label in (("candidate_sa", "#3182bd", "candidate SA"),
                                 ("matched_seed_sa", "#756bb1", "matched seed SA")):
        axes[0].plot(x, summary[f"{prefix}_median"], marker="o", color=color,
                     label=f"{label} median")
        axes[0].fill_between(x, summary[f"{prefix}_q25"], summary[f"{prefix}_q75"],
                             color=color, alpha=0.16, label=f"{label} IQR")
    axes[0].set_ylabel("SA score (lower is easier)")
    axes[0].set_title("Synthetic accessibility relative to matched seeds")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].plot(x, summary["delta_sa_median"], marker="o", color="#e6550d",
                 label="median delta SA")
    axes[1].fill_between(x, summary["delta_sa_q25"], summary["delta_sa_q75"],
                         color="#fdae6b", alpha=0.3, label="delta SA IQR")
    axes[1].set_ylabel("delta SA (candidate - seed)")
    axes[1].set_xlabel("Raw proposal budget per seed")
    fraction_axis = axes[1].twinx()
    fraction_axis.plot(x, summary["fraction_no_harder_than_seed"], marker="s",
                       color="#31a354", label="fraction no harder")
    fraction_axis.set_ylim(0.0, 1.0)
    fraction_axis.set_ylabel("Fraction no harder")
    lines = axes[1].get_lines()[1:] + fraction_axis.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")
    axes[1].grid(alpha=0.2)
    scaling_path = figure_root / "synthetic_accessibility_scaling.png"
    _save_figure(fig, scaling_path)

    labels = {"one_cut_mmp_derivative": "One-cut MMP",
              "scaffold_preserving_non_mmp_analogue": "Same-scaffold non-MMP",
              "scaffold_changing_analogue": "Scaffold-changing",
              "acyclic_non_mmp_analogue": "Acyclic non-MMP"}
    maximum_category = category.loc[category["budget"] == max(budgets)].copy()
    maximum_category["display"] = maximum_category["chemical_category"].map(labels)
    maximum_category = maximum_category.set_index("chemical_category").loc[
        cfg["comparisons"]["categories"]].reset_index()
    y = np.arange(len(maximum_category))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    median = maximum_category["delta_sa_median"].to_numpy(dtype=np.float64)
    low = maximum_category["delta_sa_q25"].to_numpy(dtype=np.float64)
    high = maximum_category["delta_sa_q75"].to_numpy(dtype=np.float64)
    axes[0].errorbar(median, y, xerr=np.vstack((median-low, high-median)),
                     fmt="o", capsize=4, color="#e6550d")
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, maximum_category["display"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("delta SA median and IQR")
    axes[0].set_title("Relative difficulty")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(y, maximum_category["fraction_no_harder_than_seed"],
                 color="#31a354", alpha=0.85)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Fraction no harder than seed")
    axes[1].set_title("Budget 1,000")
    for index, value in enumerate(maximum_category["fraction_no_harder_than_seed"]):
        axes[1].text(min(float(value)+0.02, 0.94), index,
                     f"{100*float(value):.1f}%", va="center")
    category_figure_path = figure_root / "synthetic_accessibility_by_category.png"
    _save_figure(fig, category_figure_path)

    with np.load(root / "outputs/raw/final_sa_joint_histogram_budget1000.npz") as payload:
        counts, seed_edges, candidate_edges = (payload["counts"], payload["seed_edges"],
                                                payload["candidate_edges"])
    fig, axis = plt.subplots(figsize=(6.8, 5.8))
    mesh = axis.pcolormesh(seed_edges, candidate_edges, np.log10(counts.T+1.0),
                           cmap="viridis", shading="auto")
    axis.plot([1, 10], [1, 10], color="white", linestyle="--", linewidth=1.4,
              label="candidate = seed")
    axis.set(xlim=(1, 10), ylim=(1, 10), xlabel="Seed SA score",
             ylabel="Candidate SA score",
             title="Seed-candidate synthetic-accessibility density (budget 1,000)")
    axis.legend(loc="upper left")
    fig.colorbar(mesh, ax=axis).set_label("log10(pair count + 1)")
    density_path = figure_root / "synthetic_accessibility_seed_candidate_density.png"
    _save_figure(fig, density_path)

    analysis_summary = load_json(root / "outputs/sa_extension_analysis_summary.json")
    decision = {
        "schema_version": 1, "study_id": cfg["study_id"],
        "status": "descriptive_post_completion_extension", "score": cfg["score"],
        "population": {"global_identity_rows": int(analysis_summary["global_identity_rows"]),
                       "seed_candidate_rows": int(analysis_summary["seed_candidate_rows"]),
                       "genuine_nonseed_pairs_at_1000": int(maximum["pair_rows"]),
                       "seed_rows": int(maximum["seed_rows"])},
        "budget_1000": {
            "candidate_sa_median": float(maximum["candidate_sa_median"]),
            "matched_seed_sa_median": float(maximum["matched_seed_sa_median"]),
            "delta_sa_median": float(maximum["delta_sa_median"]),
            "delta_sa_q25": float(maximum["delta_sa_q25"]),
            "delta_sa_q75": float(maximum["delta_sa_q75"]),
            "fraction_no_harder_than_seed": float(maximum["fraction_no_harder_than_seed"]),
            "fraction_not_more_than_half_point_harder":
                float(maximum["fraction_not_more_than_half_point_harder"]),
            "fraction_at_least_one_point_easier":
                float(maximum["fraction_at_least_one_point_easier"]),
            "fraction_at_least_one_point_harder":
                float(maximum["fraction_at_least_one_point_harder"]),
            "seed_macro_mean_delta_sa": float(maximum["seed_macro_mean_delta_sa"]),
            "seed_macro_mean_delta_sa_ci": [float(maximum["seed_macro_mean_delta_sa_ci_low"]),
                                             float(maximum["seed_macro_mean_delta_sa_ci_high"])],
            "seed_macro_mean_fraction_no_harder":
                float(maximum["seed_macro_mean_fraction_no_harder"]),
            "seed_macro_fraction_no_harder_ci":
                [float(maximum["seed_macro_fraction_no_harder_ci_low"]),
                 float(maximum["seed_macro_fraction_no_harder_ci_high"])]},
        "original_step2d_decision_changed": False,
        "bounded_conclusion": "SA score is descriptive heuristic complexity evidence, not a feasible synthesis route.",
        "scope_limits": ["RDKit SA score is not retrosynthesis or experimental validation.",
                         "No candidate was regenerated, removed, reranked, or used to alter Step 2d.",
                         "No training, latent perturbation, property optimization, test data, or endpoint labels."]}
    decision_path = root / "outputs/sa_extension_decision.json"
    atomic_write_json(decision_path, decision, root)

    table_lines = ["| Budget | Candidate SA median | Seed SA median | Median delta SA | No harder | No more than 0.5 harder |",
                   "|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        table_lines.append(f"| {int(row.budget):,} | {row.candidate_sa_median:.3f} | "
                           f"{row.matched_seed_sa_median:.3f} | {row.delta_sa_median:+.3f} | "
                           f"{100*row.fraction_no_harder_than_seed:.2f}% | "
                           f"{100*row.fraction_not_more_than_half_point_harder:.2f}% |")
    category_lines = ["| Category at budget 1,000 | Pairs | Median delta SA | No harder |",
                      "|---|---:|---:|---:|"]
    for row in maximum_category.itertuples(index=False):
        category_lines.append(f"| {row.display} | {int(row.pair_rows):,} | "
                              f"{row.delta_sa_median:+.3f} | "
                              f"{100*row.fraction_no_harder_than_seed:.2f}% |")
    results_section = f"""## Post-completion synthetic-accessibility comparison

This additive analysis scored all {decision['population']['global_identity_rows']:,}
globally unique policy-accepted identities once with RDKit
{registered['rdkit_version']} Contrib SA_Score, then evaluated all
{decision['population']['seed_candidate_rows']:,} unique seed-candidate rows.
Lower is heuristically easier; delta SA is candidate minus matched seed. Exact
seed identities are excluded below.

{chr(10).join(table_lines)}

At budget 1,000, seed-macro mean delta SA was
{maximum['seed_macro_mean_delta_sa']:+.3f} (95% seed-bootstrap CI
{maximum['seed_macro_mean_delta_sa_ci_low']:+.3f} to
{maximum['seed_macro_mean_delta_sa_ci_high']:+.3f}). Seed-macro mean fraction no
harder was {100*maximum['seed_macro_mean_fraction_no_harder']:.2f}% (95% CI
{100*maximum['seed_macro_fraction_no_harder_ci_low']:.2f}% to
{100*maximum['seed_macro_fraction_no_harder_ci_high']:.2f}%).

{chr(10).join(category_lines)}

This heuristic is not a route, yield, availability, or experimental
synthesizability claim. Generation, ranking, gates, and budget remain unchanged.
See PROTOCOL_SA_EXTENSION.md and the final_sa tables for full definitions."""
    decision_section = f"""## Post-completion SA-score context

At budget 1,000, median candidate-minus-seed SA was
**{maximum['delta_sa_median']:+.3f}** (IQR {maximum['delta_sa_q25']:+.3f} to
{maximum['delta_sa_q75']:+.3f}); **{100*maximum['fraction_no_harder_than_seed']:.2f}%**
of genuine non-seed pairs were no harder than their seed.

This does not change the Step 2d strategy, 1,000-proposal recommendation, or
large-library decision. SA score is not a synthesis-feasibility claim."""
    readme_section = """## Synthetic-accessibility extension

The post-completion no-generation comparison is defined in
PROTOCOL_SA_EXTENSION.md. Machine-readable outputs are under outputs/tables and
the three SA figures plus corrected validity figure are under outputs/figures."""
    sections = {"RESULTS.md": results_section, "DECISION.md": decision_section,
                "README.md": readme_section}
    updated_docs = {}
    for original, snapshot in BASE_DOC_SNAPSHOTS.items():
        updated = _append_section((root/snapshot).read_text(encoding="utf-8"),
                                  "STEP2D_SA_EXTENSION", sections[original])
        atomic_write_text(root/original, updated, root)
        updated_docs[original] = sha256_file(root/original)

    report_state_path = root / "state/REPORT_COMPLETE.json"
    report_state = load_json(report_state_path)
    report_state.update({"completed_at": utc_now(),
                         "results_sha256": sha256_file(root/"RESULTS.md"),
                         "decision_markdown_sha256": sha256_file(root/"DECISION.md"),
                         "readme_sha256": sha256_file(root/"README.md")})
    report_state.setdefault("figure_sha256", {})["quality_locality_diversity"] = sha256_file(quality_path)
    report_state["sa_extension"] = {
        "status": "complete", "decision_sha256": sha256_file(decision_path),
        "document_sha256": updated_docs,
        "figure_sha256": {"synthetic_accessibility_scaling": sha256_file(scaling_path),
                          "synthetic_accessibility_by_category": sha256_file(category_figure_path),
                          "synthetic_accessibility_seed_candidate_density": sha256_file(density_path)}}
    atomic_write_json(report_state_path, report_state, root)
    state = {"schema_version": 1, "status": "complete", "completed_at": utc_now(),
             "decision_sha256": sha256_file(decision_path), "document_sha256": updated_docs,
             "figure_sha256": {"quality_locality_diversity": sha256_file(quality_path),
                               "synthetic_accessibility_scaling": sha256_file(scaling_path),
                               "synthetic_accessibility_by_category": sha256_file(category_figure_path),
                               "synthetic_accessibility_seed_candidate_density": sha256_file(density_path)},
             "test_rows": 0, "endpoint_labels_used": False}
    atomic_write_json(state_path, state, root)
    print(json.dumps(state, sort_keys=True))




def verify_extension(root: Path) -> None:
    """Verify scientific boundaries, joins, summaries, plots, and artifact hashes."""
    registered = load_json(root / "state/SA_EXTENSION_REGISTERED.json")
    assert_source_hashes(root, registered)
    assert_base_protected(root, registered)
    cfg = extension_config(root)
    analysis = load_json(root / "outputs/sa_extension_analysis_summary.json")

    identities = pd.read_parquet(
        root / "outputs/tables/final_sa_scores_by_identity.parquet"
    )
    comparisons = pd.read_parquet(
        root / "outputs/tables/final_candidate_sa_comparison.parquet"
    )
    seed_metrics = pd.read_parquet(
        root / "outputs/tables/final_seed_sa_metrics.parquet"
    )
    budget = pd.read_csv(root / "outputs/tables/final_sa_summary_by_budget.csv")
    category = pd.read_csv(
        root / "outputs/tables/final_sa_summary_by_category_budget.csv"
    )
    bootstrap = pd.read_csv(root / "outputs/tables/final_sa_bootstrap.csv")
    base_candidates = pd.read_parquet(
        root / "outputs/tables/final_candidate_characterization.parquet",
        columns=["candidate_structure_index", "is_seed_identity", "is_genuine_nonseed"],
    )

    if len(identities) != int(analysis["global_identity_rows"]):
        raise RuntimeError("SA identity-table row count differs from the sealed analysis")
    if len(comparisons) != len(base_candidates) or len(comparisons) != int(
        analysis["seed_candidate_rows"]
    ):
        raise RuntimeError("SA comparison table is not a lossless candidate-table join")
    if identities["molecule_hash"].duplicated().any():
        raise RuntimeError("SA identity table is not globally identity-unique")
    scores = identities["sa_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 1).any() or (scores > 10).any():
        raise RuntimeError("SA identity scores are not finite values in [1, 10]")
    candidate_indices = base_candidates["candidate_structure_index"].to_numpy(
        dtype=np.int64
    )
    observed_scores = comparisons["candidate_sa_score"].to_numpy(dtype=np.float64)
    if not np.array_equal(observed_scores, scores[candidate_indices]):
        raise RuntimeError("Candidate SA scores do not reproduce the registered identity join")
    if not np.array_equal(
        comparisons["is_seed_identity"].to_numpy(dtype=bool),
        base_candidates["is_seed_identity"].to_numpy(dtype=bool),
    ):
        raise RuntimeError("Seed-identity flags changed during SA analysis")
    exact_delta = comparisons.loc[
        comparisons["is_seed_identity"].astype(bool),
        "delta_sa_candidate_minus_seed",
    ].to_numpy(dtype=np.float64)
    if len(exact_delta) == 0 or not np.array_equal(exact_delta, np.zeros_like(exact_delta)):
        raise RuntimeError("Exact seed identities failed the zero-delta control")

    budgets = [int(value) for value in cfg["population"]["budgets"]]
    if budget["budget"].astype(int).tolist() != budgets:
        raise RuntimeError("SA budget summary does not match the frozen budgets")
    expected_seed_rows = int(analysis["seed_rows"]) * len(budgets)
    if len(seed_metrics) != expected_seed_rows:
        raise RuntimeError("Seed-macro table is incomplete")
    expected_categories = len(cfg["comparisons"]["categories"]) * len(budgets)
    if len(category) != expected_categories:
        raise RuntimeError("Category-by-budget SA table is incomplete")
    if len(bootstrap) != 2 * len(budgets):
        raise RuntimeError("Seed-bootstrap table is incomplete")
    for column in (
        "fraction_no_harder_than_seed",
        "fraction_not_more_than_half_point_harder",
        "fraction_absolute_delta_within_half_point",
        "fraction_at_least_one_point_easier",
        "fraction_at_least_one_point_harder",
    ):
        values = budget[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError(f"Invalid fraction in {column}")
    maximum = comparisons.loc[
        comparisons["is_genuine_nonseed"].astype(bool)
        & (comparisons["first_proposal_rank"] <= max(budgets))
    ]
    if len(maximum) != int(analysis["genuine_nonseed_pair_rows_at_1000"]):
        raise RuntimeError("Maximum-budget genuine-pair population changed")
    with np.load(root / "outputs/raw/final_sa_joint_histogram_budget1000.npz") as payload:
        if int(payload["counts"].sum()) != len(maximum):
            raise RuntimeError("Joint SA histogram does not cover every maximum-budget pair")

    for original, snapshot in BASE_DOC_SNAPSHOTS.items():
        if sha256_file(root / snapshot) != registered["base_document_snapshot_sha256"][snapshot]:
            raise RuntimeError(f"Base-document snapshot changed: {snapshot}")
        text_value = (root / original).read_text(encoding="utf-8")
        if text_value.count("<!-- STEP2D_SA_EXTENSION:START -->") != 1:
            raise RuntimeError(f"Missing or duplicated SA section in {original}")
        if text_value.count("<!-- STEP2D_SA_EXTENSION:END -->") != 1:
            raise RuntimeError(f"Missing or duplicated SA section end in {original}")

    from PIL import Image

    figure_names = (
        "quality_locality_diversity_scaling.png",
        "synthetic_accessibility_scaling.png",
        "synthetic_accessibility_by_category.png",
        "synthetic_accessibility_seed_candidate_density.png",
    )
    figure_dimensions: dict[str, list[int]] = {}
    for name in figure_names:
        path = root / "outputs/figures" / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        if width < 800 or height < 500:
            raise RuntimeError(f"SA figure is unexpectedly small: {name}")
        figure_dimensions[name] = [width, height]

    boundary_keys = (
        "candidate_generation_changed",
        "encoder_training",
        "decoder_training",
        "latent_perturbation",
        "mmp_direction_editing",
        "property_optimization",
    )
    if any(analysis.get(key) is not False for key in boundary_keys):
        raise RuntimeError("A frozen Step-2d scientific boundary was violated")
    if analysis.get("test_rows") != 0 or analysis.get("endpoint_labels_used") is not False:
        raise RuntimeError("Forbidden test rows or endpoint labels were used")
    decision = load_json(root / "outputs/sa_extension_decision.json")
    if decision.get("original_step2d_decision_changed") is not False:
        raise RuntimeError("SA extension improperly changed the original Step-2d decision")

    verification = {
        "schema_version": 1,
        "status": "passed",
        "verified_at": utc_now(),
        "checks": {
            "registered_sources_unchanged": True,
            "protected_base_artifacts_unchanged": True,
            "identity_score_population_complete": True,
            "candidate_join_lossless": True,
            "exact_seed_zero_delta_control": True,
            "summaries_complete": True,
            "joint_histogram_complete": True,
            "figures_readable": True,
            "documents_updated_once": True,
            "frozen_scientific_boundary_preserved": True,
        },
        "rows": {
            "global_identities": len(identities),
            "seed_candidate_pairs": len(comparisons),
            "genuine_nonseed_pairs_at_1000": len(maximum),
            "seed_budget_rows": len(seed_metrics),
        },
        "figure_dimensions": figure_dimensions,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    verification_path = root / "outputs/sa_extension_verification.json"
    atomic_write_json(verification_path, verification, root)
    atomic_write_json(
        root / "state/SA_EXTENSION_VERIFIED.json",
        {
            "schema_version": 1,
            "status": "passed",
            "verified_at": verification["verified_at"],
            "verification_sha256": sha256_file(verification_path),
        },
        root,
    )
    complete_path = root / "state/COMPLETE.json"
    complete = load_json(complete_path)
    complete["sa_extension"] = {
        "status": "passed",
        "verification_sha256": sha256_file(verification_path),
        "original_step2d_decision_changed": False,
        "test_rows": 0,
        "endpoint_labels_used": False,
    }
    atomic_write_json(complete_path, complete, root)
    write_hash_ledger(root, root / "outputs/SHA256SUMS")
    print(json.dumps(verification, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--step-root", type=Path, default=STEP_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("register")
    subparsers.add_parser("component-test")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--workers", type=int, default=64)
    subparsers.add_parser("report")
    subparsers.add_parser("verify")
    args = parser.parse_args()
    root = args.step_root.resolve()
    if root != STEP_ROOT.resolve():
        raise RuntimeError(f"Unexpected Step-2d root: {root}")
    if args.command == "register":
        register_extension(root, args.repo_root.resolve())
    elif args.command == "component-test":
        component_test(root)
    elif args.command == "analyze":
        analyze_extension(root, args.workers)
    elif args.command == "report":
        report_extension(root)
    elif args.command == "verify":
        verify_extension(root)


if __name__ == "__main__":
    main()
