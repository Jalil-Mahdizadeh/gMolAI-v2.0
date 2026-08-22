#!/usr/bin/env python3
"""Create the retained-artifact checksum manifest and immutable completion seal."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark_io import BENCHMARK_DIR, atomic_write_json, atomic_write_text, load_json, sha256_file


def retained_files() -> list[Path]:
    paths: set[Path] = set()
    for name in (".gitignore", "PROTOCOL.md", "README.md", "RESULTS.md", "protocol.json", "run_lbvs.sbatch"):
        paths.add(BENCHMARK_DIR / name)
    for directory in ("scripts", "tests", "audits", "state", "results", "figures"):
        paths.update(path for path in (BENCHMARK_DIR / directory).rglob("*") if path.is_file())
    for directory in ("inputs/prepared", "artifacts/screens", "embeddings/model-panels"):
        paths.update(path for path in (BENCHMARK_DIR / directory).rglob("*") if path.is_file())
    paths.update(
        {
            BENCHMARK_DIR / "inputs/raw/VSDS_vd-v3.rar",
            BENCHMARK_DIR / "inputs/raw/VSDS_TrueDecoy_gap_Supplementary_Data_1.xlsx",
        }
    )
    excluded = {
        BENCHMARK_DIR / "state/COMPLETE.json",
        BENCHMARK_DIR / "results/SHA256SUMS",
    }
    return sorted(
        (
            path
            for path in paths - excluded
            if path.name != ".gitkeep"
            and path.suffix not in {".partial", ".orig", ".rej", ".pyc"}
            and "__pycache__" not in path.parts
        ),
        key=lambda path: str(path.relative_to(BENCHMARK_DIR)),
    )


def verify_existing(manifest_path: Path, complete: dict) -> None:
    if complete.get("status") != "complete":
        raise RuntimeError("Existing completion state is not complete")
    if complete.get("sha256_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Existing checksum manifest changed")
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = BENCHMARK_DIR / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"Completed artifact changed: {path}")


def main() -> None:
    verification_path = BENCHMARK_DIR / "audits/verification.json"
    verification = load_json(verification_path)
    if verification.get("status") != "ok":
        raise RuntimeError("Independent verification must pass before finalization")
    complete_path = BENCHMARK_DIR / "state/COMPLETE.json"
    checksum_path = BENCHMARK_DIR / "results/SHA256SUMS"
    if complete_path.exists():
        verify_existing(checksum_path, load_json(complete_path))
        print(complete_path.read_text(encoding="utf-8").strip())
        return
    files = retained_files()
    missing = [path for path in files if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"Missing or unsafe retained artifacts: {missing}")
    entries = [
        (sha256_file(path), str(path.relative_to(BENCHMARK_DIR))) for path in files
    ]
    atomic_write_text(
        checksum_path,
        "".join(f"{digest}  {relative}\n" for digest, relative in entries),
    )
    population = load_json(BENCHMARK_DIR / "state/POPULATION_FROZEN.json")
    retrieval = load_json(BENCHMARK_DIR / "state/RETRIEVAL_COMPLETE.json")
    result = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "VSDS-vd v3 TrueDecoy_gap ligand retrieval",
        "models": verification["models_verified"],
        "targets": verification["targets_verified"],
        "final_unique_molecules": population["final_unique_molecules"],
        "retrieval_rows": retrieval["rows"],
        "retained_artifacts": len(entries),
        "sha256_manifest": str(checksum_path),
        "sha256_manifest_sha256": sha256_file(checksum_path),
        "verification": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(complete_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
