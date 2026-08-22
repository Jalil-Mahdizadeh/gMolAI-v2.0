#!/usr/bin/env python3
"""Freeze all-seven common support and eligible targets after verified encoding."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os

import numpy as np

from benchmark_io import (
    BENCHMARK_DIR,
    atomic_write_json,
    columns_of,
    load_json,
    load_protocol,
    read_panel_tsv,
    read_tsv,
    sha256_file,
    sha256_lines,
    write_csv,
    write_tsv,
)


def atomic_savez(path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output: {temporary}")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    protocol = load_protocol()
    models = tuple(protocol["models"]["primary_order"])
    prepared = BENCHMARK_DIR / "inputs/prepared"
    candidate_panel = prepared / "molecule_candidates.tsv"
    candidate_memberships = prepared / "memberships_candidates.tsv"
    target_path = prepared / "targets.tsv"
    molecules = read_panel_tsv(candidate_panel)
    memberships = read_tsv(candidate_memberships)
    targets = read_tsv(target_path)
    molecule_by_hash = {row["molecule_hash"]: row for row in molecules}
    target_by_id = {row["target_id"]: row for row in targets}
    accepted_by_model: dict[str, set[str]] = {}
    index_by_model: dict[str, dict[str, int]] = {}
    validation_by_model = {}
    for model in models:
        model_panel = prepared / "model_panels" / f"{model}.tsv"
        rows = read_panel_tsv(model_panel)
        audit_path = BENCHMARK_DIR / "audits" / f"embedding-{model}.json"
        audit = load_json(audit_path)
        matrix_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{model}.npy"
        metadata_path = BENCHMARK_DIR / "embeddings/model-panels" / f"{model}.json"
        if audit.get("status") != "ok" or audit.get("model") != model:
            raise RuntimeError(f"Missing valid representation audit for {model}")
        if int(audit.get("rows", -1)) != len(rows):
            raise RuntimeError(f"Representation coverage differs for {model}")
        if audit.get("input_sha256") != sha256_file(model_panel):
            raise RuntimeError(f"Representation panel hash differs for {model}")
        if audit.get("embedding_sha256") != sha256_file(matrix_path):
            raise RuntimeError(f"Representation matrix hash differs for {model}")
        if audit.get("metadata_sha256") != sha256_file(metadata_path):
            raise RuntimeError(f"Representation metadata hash differs for {model}")
        accepted_by_model[model] = {row["molecule_hash"] for row in rows}
        index_by_model[model] = {
            row["molecule_hash"]: int(row["panel_index"]) for row in rows
        }
        validation_by_model[model] = {
            "model_panel": str(model_panel),
            "model_panel_sha256": sha256_file(model_panel),
            "rows_with_validated_representation": len(rows),
            "identity_sha256": sha256_lines(row["molecule_hash"] for row in rows),
            "embedding_sha256": audit["embedding_sha256"],
            "metadata_sha256": audit["metadata_sha256"],
            "validation_sha256": sha256_file(audit_path),
        }
    common_before_target_eligibility = set.intersection(*accepted_by_model.values())
    if not common_before_target_eligibility:
        raise RuntimeError("All-seven representation intersection is empty")
    prepared_by_target_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    common_by_target_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in memberships:
        key = (row["target_id"], row["label"])
        prepared_by_target_label[key].add(row["molecule_hash"])
        if row["molecule_hash"] in common_before_target_eligibility:
            common_by_target_label[key].add(row["molecule_hash"])
    minimum_actives = int(
        protocol["coverage_and_eligibility"]["primary_target_minimum_common_actives"]
    )
    minimum_inactives = int(
        protocol["coverage_and_eligibility"]["primary_target_minimum_common_inactives"]
    )
    eligible_targets = {
        target_id
        for target_id in target_by_id
        if len(common_by_target_label[(target_id, "active")]) >= minimum_actives
        and len(common_by_target_label[(target_id, "inactive_or_lower_affinity")])
        >= minimum_inactives
    }
    final_memberships = [
        dict(row)
        for row in memberships
        if row["target_id"] in eligible_targets
        and row["molecule_hash"] in common_before_target_eligibility
    ]
    final_memberships.sort(
        key=lambda row: (row["target_id"], row["label"], row["molecule_hash"])
    )
    for index, row in enumerate(final_memberships):
        row["membership_index"] = index
    final_identities = {row["molecule_hash"] for row in final_memberships}
    final_molecules = [
        dict(molecule_by_hash[identity]) for identity in sorted(final_identities)
    ]
    for index, row in enumerate(final_molecules):
        row["panel_index"] = index
    if not final_molecules or not eligible_targets:
        raise RuntimeError("Frozen primary population is empty")
    common_panel = prepared / "common_panel.tsv"
    common_membership_path = prepared / "common_memberships.tsv"
    write_tsv(common_panel, final_molecules, columns_of(candidate_panel))
    write_tsv(
        common_membership_path,
        final_memberships,
        columns_of(candidate_memberships),
    )
    maps = {
        model: np.asarray(
            [index_by_model[model][row["molecule_hash"]] for row in final_molecules],
            dtype=np.int64,
        )
        for model in models
    }
    index_map_path = BENCHMARK_DIR / "state" / "model_index_maps.npz"
    atomic_savez(index_map_path, **maps)

    manifest_rows = []
    for row in memberships:
        target_id = row["target_id"]
        identity = row["molecule_hash"]
        common = identity in common_before_target_eligibility
        target_eligible = target_id in eligible_targets
        if not common:
            missing = [model for model in models if identity not in accepted_by_model[model]]
            exclusion = "missing_representation:" + ";".join(missing)
        elif not target_eligible:
            exclusion = "target_below_prespecified_common_support_eligibility"
        else:
            exclusion = ""
        output = {
            "target_id": target_id,
            "target_class": target_by_id[target_id]["target_class"],
            "label": row["label"],
            "molecule_hash": identity,
            "canonical_smiles": row["canonical_smiles"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
            **{
                f"represented_{model}": identity in accepted_by_model[model]
                for model in models
            },
            "all_seven_common_support": common,
            "primary_target_eligible": target_eligible,
            "final_population": common and target_eligible,
            "exclusion_reason": exclusion,
        }
        manifest_rows.append(output)
    result_tables = BENCHMARK_DIR / "results/tables"
    dataset_manifest_path = result_tables / "dataset_manifest.csv"
    write_csv(dataset_manifest_path, manifest_rows, tuple(manifest_rows[0]))

    population_target_rows = []
    for target in targets:
        target_id = target["target_id"]
        prepared_active = len(prepared_by_target_label[(target_id, "active")])
        prepared_inactive = len(
            prepared_by_target_label[(target_id, "inactive_or_lower_affinity")]
        )
        common_active = len(common_by_target_label[(target_id, "active")])
        common_inactive = len(
            common_by_target_label[(target_id, "inactive_or_lower_affinity")]
        )
        eligible = target_id in eligible_targets
        reasons = []
        if common_active < minimum_actives:
            reasons.append(f"common_actives_{common_active}_below_{minimum_actives}")
        if common_inactive < minimum_inactives:
            reasons.append(f"common_inactives_{common_inactive}_below_{minimum_inactives}")
        population_target_rows.append(
            {
                "target_id": target_id,
                "target_class": target["target_class"],
                "source_active": target["source_active_count"],
                "source_inactive_or_lower_affinity": target["source_inactive_count"],
                "prepared_active": prepared_active,
                "prepared_inactive_or_lower_affinity": prepared_inactive,
                "common_support_active": common_active,
                "common_support_inactive_or_lower_affinity": common_inactive,
                "common_support_total": common_active + common_inactive,
                "primary_target_eligible": eligible,
                "primary_exclusion_reason": ";".join(reasons),
            }
        )
    population_target_path = result_tables / "population_target_summary.csv"
    write_csv(
        population_target_path,
        population_target_rows,
        tuple(population_target_rows[0]),
    )

    coverage_rows = []
    for model in models:
        accepted = accepted_by_model[model]
        coverage_rows.append(
            {
                "model": model,
                "display_name": protocol["models"][model]["display_name"],
                "attempted_unique_molecules": len(molecules),
                "validated_representation_rows": len(accepted),
                "rejected_before_common_support": len(molecules) - len(accepted),
                "validated_coverage_fraction": len(accepted) / max(1, len(molecules)),
                "all_seven_common_before_target_eligibility": len(
                    common_before_target_eligibility
                ),
                "final_eligible_target_unique_molecules": len(final_molecules),
                "validated_identity_sha256": validation_by_model[model][
                    "identity_sha256"
                ],
            }
        )
    coverage_path = result_tables / "model_coverage.csv"
    write_csv(coverage_path, coverage_rows, tuple(coverage_rows[0]))

    coverage_detail = []
    for target in targets:
        target_id = target["target_id"]
        for label in ("active", "inactive_or_lower_affinity"):
            attempted = prepared_by_target_label[(target_id, label)]
            for model in models:
                covered = attempted & accepted_by_model[model]
                coverage_detail.append(
                    {
                        "target_id": target_id,
                        "target_class": target["target_class"],
                        "label": label,
                        "model": model,
                        "attempted_memberships": len(attempted),
                        "represented_memberships": len(covered),
                        "rejected_memberships": len(attempted) - len(covered),
                        "coverage_fraction": len(covered) / max(1, len(attempted)),
                        "represented_identity_sha256": sha256_lines(sorted(covered)),
                    }
                )
    detail_path = result_tables / "model_coverage_by_target_label.csv"
    write_csv(detail_path, coverage_detail, tuple(coverage_detail[0]))

    result = {
        "schema_version": 1,
        "status": "frozen",
        "freeze_stage": "after_all_model_representation_validation_before_anchor_generation_or_retrieval",
        "protocol_sha256": sha256_file(BENCHMARK_DIR / "protocol.json"),
        "attempted_unique_molecules": len(molecules),
        "all_seven_common_before_target_eligibility": len(
            common_before_target_eligibility
        ),
        "all_seven_common_identity_sha256": sha256_lines(
            sorted(common_before_target_eligibility)
        ),
        "source_targets": len(targets),
        "eligible_targets": len(eligible_targets),
        "eligible_target_ids": sorted(eligible_targets),
        "final_unique_molecules": len(final_molecules),
        "final_memberships": len(final_memberships),
        "final_active_memberships": sum(
            row["label"] == "active" for row in final_memberships
        ),
        "final_inactive_or_lower_affinity_memberships": sum(
            row["label"] == "inactive_or_lower_affinity"
            for row in final_memberships
        ),
        "common_panel": str(common_panel),
        "common_panel_sha256": sha256_file(common_panel),
        "common_identity_sha256": sha256_lines(
            row["molecule_hash"] for row in final_molecules
        ),
        "common_memberships": str(common_membership_path),
        "common_memberships_sha256": sha256_file(common_membership_path),
        "model_index_maps": str(index_map_path),
        "model_index_maps_sha256": sha256_file(index_map_path),
        "dataset_manifest": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "population_target_summary": str(population_target_path),
        "population_target_summary_sha256": sha256_file(population_target_path),
        "model_coverage": str(coverage_path),
        "model_coverage_sha256": sha256_file(coverage_path),
        "model_coverage_by_target_label": str(detail_path),
        "model_coverage_by_target_label_sha256": sha256_file(detail_path),
        "models": validation_by_model,
        "anchors_generated": False,
        "performance_inspected": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path = BENCHMARK_DIR / "state" / "POPULATION_FROZEN.json"
    atomic_write_json(state_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

