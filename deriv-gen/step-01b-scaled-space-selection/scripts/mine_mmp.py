#!/usr/bin/env python3
"""Mine scalable core-independent MMP observations with DuckDB."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from scaled_common import (
    atomic_write_csv,
    atomic_write_json,
    ensure_within,
    sha256_file,
    support_tier,
)


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def copy_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    target: Path,
    step_root: Path,
) -> None:
    target = ensure_within(target, step_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        connection.execute(
            f"COPY ({query}) TO {quoted(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def validate_stage_inputs(step_root: Path) -> dict[str, Path]:
    seal_path = step_root / "state" / "FRAGMENTATION_COMPLETE.json"
    if not seal_path.is_file():
        raise RuntimeError("Fragmentation stage is not sealed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    result: dict[str, Path] = {}
    for name, record in seal["outputs"].items():
        path = step_root / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Fragmentation artifact changed: {name}")
        result[name] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step-root",
        type=Path,
        default=Path("/repo/deriv-gen/step-01b-scaled-space-selection"),
    )
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    step_root = args.step_root.resolve()
    config = json.loads(
        (step_root / "config" / "protocol.json").read_text(encoding="utf-8")
    )
    paths = validate_stage_inputs(step_root)
    intermediate = step_root / "intermediate"
    tables = step_root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    threads = args.threads or min(48, len(os.sched_getaffinity(0)))
    temporary_dir = ensure_within(step_root / "state" / "duckdb_tmp", step_root)
    temporary_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(":memory:")
    try:
        connection.execute(f"SET threads={int(threads)}")
        connection.execute("SET memory_limit='200GB'")
        connection.execute(f"SET temp_directory={quoted(temporary_dir)}")
        group_paths: dict[str, Path] = {}
        observation_paths: dict[str, Path] = {}
        for split in ("train", "validation"):
            fragment_path = paths[f"{split}_fragments"]
            group_path = intermediate / f"{split}_fragment_groups.parquet"
            observation_path = intermediate / f"{split}_mmp_observations.parquet"
            print(f"aggregating {split} fragment groups", flush=True)
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP VIEW selected_fragments AS
                SELECT * EXCLUDE (row_number_in_molecule_core)
                FROM (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY core, molecule_index
                            ORDER BY substituent, molecule_index
                        ) AS row_number_in_molecule_core
                    FROM read_parquet({quoted(fragment_path)})
                )
                WHERE row_number_in_molecule_core = 1
                """
            )
            group_query = """
                SELECT
                    core,
                    substituent,
                    min(core_heavy_atoms)::INTEGER AS core_heavy_atoms,
                    min(substituent_heavy_atoms)::INTEGER
                        AS substituent_heavy_atoms,
                    arg_min(parent_heavy_atoms, molecule_index)::INTEGER
                        AS representative_parent_heavy_atoms,
                    min(molecule_index)::BIGINT AS representative_index,
                    count(*)::BIGINT AS molecule_count,
                    count(DISTINCT parent_heavy_atoms)::INTEGER
                        AS distinct_parent_heavy_atom_counts
                FROM selected_fragments
                GROUP BY core, substituent
            """
            copy_query(connection, group_query, group_path, step_root)
            inconsistent = connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet({quoted(group_path)})
                WHERE distinct_parent_heavy_atom_counts != 1
                """
            ).fetchone()[0]
            if inconsistent:
                raise RuntimeError(
                    f"{split} has {inconsistent} inconsistent core/substituent groups"
                )
            settings = config["mmp"]
            observation_query = f"""
                SELECT
                    a.core,
                    a.substituent || '>>' || b.substituent AS transform,
                    a.substituent AS lhs_substituent,
                    b.substituent AS rhs_substituent,
                    a.representative_index::BIGINT AS lhs_index,
                    b.representative_index::BIGINT AS rhs_index,
                    a.molecule_count::BIGINT AS lhs_group_molecules,
                    b.molecule_count::BIGINT AS rhs_group_molecules,
                    (a.molecule_count * b.molecule_count)::HUGEINT
                        AS molecule_pair_multiplicity,
                    a.core_heavy_atoms::INTEGER AS core_heavy_atoms,
                    a.substituent_heavy_atoms::INTEGER
                        AS lhs_substituent_heavy_atoms,
                    b.substituent_heavy_atoms::INTEGER
                        AS rhs_substituent_heavy_atoms,
                    a.representative_parent_heavy_atoms::INTEGER
                        AS lhs_parent_heavy_atoms,
                    b.representative_parent_heavy_atoms::INTEGER
                        AS rhs_parent_heavy_atoms
                FROM read_parquet({quoted(group_path)}) AS a
                INNER JOIN read_parquet({quoted(group_path)}) AS b
                    ON a.core = b.core
                    AND a.substituent < b.substituent
                WHERE abs(
                    a.substituent_heavy_atoms - b.substituent_heavy_atoms
                ) <= {int(settings["max_variable_heavy_atom_delta"])}
                  AND abs(
                    a.representative_parent_heavy_atoms
                    - b.representative_parent_heavy_atoms
                  ) <= {int(settings["max_parent_heavy_atom_delta"])}
            """
            print(f"mining {split} core-transformation observations", flush=True)
            copy_query(
                connection, observation_query, observation_path, step_root
            )
            duplicate = connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT core, transform, count(*) AS n
                    FROM read_parquet({quoted(observation_path)})
                    GROUP BY core, transform
                    HAVING n != 1
                )
                """
            ).fetchone()[0]
            if duplicate:
                raise RuntimeError(
                    f"{split} observation table has duplicate core/transform rows"
                )
            group_paths[split] = group_path
            observation_paths[split] = observation_path

        support_path = intermediate / "train_transformation_support.parquet"
        support_query = f"""
            SELECT
                transform,
                count(*)::BIGINT AS train_cores,
                count(*)::BIGINT AS train_core_transform_observations,
                sum(molecule_pair_multiplicity)::HUGEINT
                    AS train_molecule_pair_multiplicity
            FROM read_parquet({quoted(observation_paths["train"])})
            GROUP BY transform
        """
        copy_query(connection, support_query, support_path, step_root)

        eligible_path = intermediate / "eligible_validation_mmp_observations.parquet"
        minimum_support = int(config["retrieval"]["minimum_train_cores"])
        eligible_query = f"""
            SELECT
                v.*,
                s.train_cores::BIGINT AS train_cores,
                CASE
                    WHEN s.train_cores >= 20 THEN '20+'
                    WHEN s.train_cores >= 10 THEN '10-19'
                    WHEN s.train_cores >= 5 THEN '5-9'
                    ELSE '2-4'
                END AS support_tier
            FROM read_parquet({quoted(observation_paths["validation"])}) AS v
            INNER JOIN read_parquet({quoted(support_path)}) AS s
                USING (transform)
            LEFT JOIN read_parquet({quoted(observation_paths["train"])}) AS t
                ON v.transform = t.transform AND v.core = t.core
            WHERE t.core IS NULL
              AND s.train_cores >= {minimum_support}
        """
        copy_query(connection, eligible_query, eligible_path, step_root)

        support = connection.execute(
            f"""
            SELECT
                s.*,
                coalesce(v.validation_unseen_core_observations, 0)::BIGINT
                    AS validation_unseen_core_observations,
                coalesce(v.validation_unseen_cores, 0)::BIGINT
                    AS validation_unseen_cores
            FROM read_parquet({quoted(support_path)}) AS s
            LEFT JOIN (
                SELECT
                    transform,
                    count(*) AS validation_unseen_core_observations,
                    count(DISTINCT core) AS validation_unseen_cores
                FROM read_parquet({quoted(eligible_path)})
                GROUP BY transform
            ) AS v USING (transform)
            ORDER BY train_cores DESC, transform
            """
        ).fetchdf()
        support["support_tier"] = [
            support_tier(int(value)) if int(value) >= 2 else "1"
            for value in support["train_cores"]
        ]
        atomic_write_csv(
            tables / "mmp_support_by_transformation.csv", support, step_root
        )

        thresholds: list[dict[str, int]] = []
        for threshold in config["mmp"]["support_thresholds"]:
            selected = support["train_cores"] >= int(threshold)
            with_validation = selected & (
                support["validation_unseen_core_observations"] > 0
            )
            thresholds.append(
                {
                    "minimum_train_cores": int(threshold),
                    "transformations": int(selected.sum()),
                    "core_transform_observations": int(
                        support.loc[
                            selected, "train_core_transform_observations"
                        ].sum()
                    ),
                    "molecule_pair_multiplicity": int(
                        support.loc[
                            selected, "train_molecule_pair_multiplicity"
                        ].sum()
                    ),
                    "transformations_with_unseen_core_validation": int(
                        with_validation.sum()
                    ),
                    "unseen_core_validation_observations": int(
                        support.loc[
                            with_validation,
                            "validation_unseen_core_observations",
                        ].sum()
                    ),
                }
            )
        threshold_frame = pd.DataFrame(thresholds)
        atomic_write_csv(
            tables / "mmp_support_thresholds.csv", threshold_frame, step_root
        )

        summary: dict[str, object] = {}
        for split in ("train", "validation"):
            observation_path = observation_paths[split]
            group_path = group_paths[split]
            row = connection.execute(
                f"""
                SELECT
                    (SELECT count(*) FROM read_parquet({quoted(group_path)}))
                        AS fragment_groups,
                    count(*) AS core_transform_observations,
                    count(DISTINCT transform) AS transformations,
                    sum(molecule_pair_multiplicity)
                        AS molecule_pair_multiplicity
                FROM read_parquet({quoted(observation_path)})
                """
            ).fetchone()
            summary[split] = {
                "fragment_groups": int(row[0]),
                "core_transform_observations": int(row[1]),
                "transformations": int(row[2]),
                "molecule_pair_multiplicity": int(row[3]),
            }
        eligible_count, eligible_transforms = connection.execute(
            f"""
            SELECT count(*), count(DISTINCT transform)
            FROM read_parquet({quoted(eligible_path)})
            """
        ).fetchone()
        summary["eligible_unseen_core_validation"] = {
            "observations": int(eligible_count),
            "transformations": int(eligible_transforms),
        }
    finally:
        connection.close()

    output_paths = {
        "train_fragment_groups": group_paths["train"],
        "validation_fragment_groups": group_paths["validation"],
        "train_mmp_observations": observation_paths["train"],
        "validation_mmp_observations": observation_paths["validation"],
        "train_transformation_support": support_path,
        "eligible_validation_mmp_observations": eligible_path,
        "mmp_support_by_transformation": tables
        / "mmp_support_by_transformation.csv",
        "mmp_support_thresholds": tables / "mmp_support_thresholds.csv",
    }
    seal = {
        "schema_version": 1,
        "status": "complete",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "threads": threads,
        "summary": summary,
        "outputs": {
            name: {
                "path": str(path.relative_to(step_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in output_paths.items()
        },
    }
    atomic_write_json(
        step_root / "state" / "MMP_MINING_COMPLETE.json", seal, step_root
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
