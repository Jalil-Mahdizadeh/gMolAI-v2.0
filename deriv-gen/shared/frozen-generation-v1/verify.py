#!/usr/bin/env python3
"""Verify the immutable deriv-gen decoder and sampling-strategy contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(__file__).with_name("contract.json")
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_bound_path(repo_root: Path, relative_path: str) -> Path:
    require(
        not Path(relative_path).is_absolute(),
        f"Contract path must be relative: {relative_path}",
    )
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise RuntimeError(f"Contract path escapes repository: {relative_path}") from error
    require(path.is_file(), f"Missing frozen file: {relative_path}")
    return path


def verify_binding(repo_root: Path, label: str, binding: dict[str, Any]) -> None:
    path = resolve_bound_path(repo_root, str(binding["path"]))
    observed = sha256_file(path)
    require(observed == binding["sha256"], f"SHA-256 mismatch for {label}: {path}")
    if "size_bytes" in binding:
        require(
            path.stat().st_size == int(binding["size_bytes"]),
            f"Size mismatch for {label}: {path}",
        )


def find_named(values: list[dict[str, Any]], name: str, label: str) -> dict[str, Any]:
    matches = [value for value in values if value.get("name") == name]
    require(len(matches) == 1, f"Expected one {label} named {name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    contract = load_json(CONTRACT_PATH)

    require(contract.get("schema_version") == 1, "Unsupported contract schema")
    require(contract.get("status") == "frozen", "Generation contract is not frozen")
    require(contract["change_control"].get("immutable") is True, "Contract is not immutable")

    decoder = contract["decoder"]
    for label, binding in decoder["artifacts"].items():
        verify_binding(repo_root, f"decoder.{label}", binding)
    for label, binding in contract["frozen_bindings"].items():
        verify_binding(repo_root, label, binding)

    runtime = decoder["artifacts"]["runtime_checkpoint"]
    inference = decoder["artifacts"]["inference_export"]
    training = load_json(
        resolve_bound_path(
            repo_root,
            contract["frozen_bindings"]["decoder_training_seal"]["path"],
        )
    )
    export = load_json(
        resolve_bound_path(
            repo_root,
            contract["frozen_bindings"]["decoder_export_seal"]["path"],
        )
    )
    require(training.get("status") == "complete", "Step-2 training seal is not complete")
    require(
        training.get("best_checkpoint_sha256") == runtime["sha256"],
        "Training seal does not bind runtime decoder",
    )
    require(
        training.get("embedding_space") == decoder["conditioning_representation"],
        "Decoder representation changed",
    )
    require(
        training.get("decoder_parameters") == decoder["parameters"],
        "Decoder parameter count changed",
    )
    require(export.get("status") == "complete", "Step-2 decoder export is not complete")
    require(
        export.get("sha256") == inference["sha256"],
        "Export seal does not bind inference artifact",
    )
    require(export.get("size_bytes") == inference["size_bytes"], "Export seal size changed")
    require(
        export.get("source_training_checkpoint_sha256") == runtime["sha256"],
        "Inference export source decoder changed",
    )
    require(
        export.get("contains_optimizer_state") is False,
        "Inference export unexpectedly contains optimizer state",
    )
    require(
        export.get("contains_gmolai_parameters") is False,
        "Inference export unexpectedly contains gMolAI parameters",
    )

    bindings = contract["frozen_bindings"]
    manifest = load_json(resolve_bound_path(repo_root, bindings["step2d_input_manifest"]["path"]))
    protocol = load_json(resolve_bound_path(repo_root, bindings["step2d_protocol"]["path"]))
    strategy_seal = load_json(
        resolve_bound_path(repo_root, bindings["step2d_strategy_seal"]["path"])
    )
    decision = load_json(resolve_bound_path(repo_root, bindings["step2d_decision"]["path"]))
    verification = load_json(resolve_bound_path(repo_root, bindings["step2d_verification"]["path"]))

    manifest_files = manifest["files"]
    require(
        manifest_files["decoder_checkpoint"]["path"] == runtime["path"],
        "Step-2d runtime decoder path changed",
    )
    require(
        manifest_files["decoder_checkpoint"]["sha256"] == runtime["sha256"],
        "Step-2d runtime decoder hash changed",
    )
    require(
        manifest_files["decoder_inference_export"]["path"] == inference["path"],
        "Step-2d inference export path changed",
    )
    require(
        manifest_files["decoder_inference_export"]["sha256"]
        == inference["sha256"],
        "Step-2d inference export hash changed",
    )

    sampling = contract["sampling_strategy"]
    generation = protocol["generation"]
    selected = sampling["selected_definition"]
    pool = sampling["base_sample_pool"]
    require(
        find_named(
            generation["development_strategies"], selected["name"], "strategy"
        )
        == selected,
        "Selected strategy definition changed",
    )
    require(
        find_named(
            generation["development_base_sample_pools"], pool["name"], "sample pool"
        )
        == pool,
        "Selected sample-pool definition changed",
    )
    require(protocol.get("seed") == sampling["global_seed"], "Global generation seed changed")
    require(
        generation.get("maximum_proposals_per_seed")
        == sampling["raw_proposals_per_seed"],
        "Proposal budget changed",
    )
    require(generation.get("beam_width") == sampling["beam_pool_width"], "Beam width changed")
    require(
        generation.get("maximum_smiles_bytes") == sampling["maximum_smiles_bytes"],
        "Maximum SMILES length changed",
    )
    require(
        generation.get("sampling_seed_definition")
        == sampling["sampling_seed_definition"],
        "Sampling seed rule changed",
    )
    require(generation.get("beam_order") == sampling["beam_order"], "Beam ordering changed")
    require(generation.get("sample_order") == sampling["sample_order"], "Sample ordering changed")
    require(generation.get("hybrid_order") == sampling["hybrid_order"], "Hybrid ordering changed")
    require(
        generation.get("target_chemistry_used_for_generation_or_ordering") is False,
        "Target chemistry entered generation ordering",
    )

    require(
        strategy_seal.get("status") == "frozen_before_final_generation",
        "Original strategy seal is not frozen",
    )
    require(strategy_seal.get("selected_strategy") == selected, "Original strategy seal changed")
    require(
        strategy_seal.get("global_seed") == sampling["global_seed"],
        "Strategy-seal seed changed",
    )
    require(
        decision.get("selected_generation_strategy") == selected,
        "Step-2d decision strategy changed",
    )
    require(
        decision.get("recommended_raw_proposal_budget")
        == sampling["raw_proposals_per_seed"],
        "Step-2d recommended budget changed",
    )
    require(verification.get("status") == "passed", "Step-2d verification is not passed")
    require(
        verification.get("selected_strategy") == selected["name"],
        "Step-2d verified strategy changed",
    )
    require(
        verification.get("recommended_budget") == sampling["raw_proposals_per_seed"],
        "Step-2d verified budget changed",
    )
    require(
        verification.get("frozen_decoder_checkpoint_sha256") == runtime["sha256"],
        "Step-2d verified decoder changed",
    )

    print(f"PASS: {contract['contract_id']}")


if __name__ == "__main__":
    main()
