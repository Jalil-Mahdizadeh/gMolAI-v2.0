from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from inference._decoder import decode_tokens, proportional_merge  # noqa: E402
from inference.gmolai import (  # noqa: E402
    GLOBAL_SEED,
    SAMPLE_POOL_NAME,
    SAMPLING_PHASE,
    Proposal,
    build_parser,
    filter_candidates,
    safe_seed_filename,
    sample_seed,
)


def test_byte_smiles_decode_is_lossless() -> None:
    smiles = "N[C@@H](C)C(=O)O"
    tokens = [ord(value) + 3 for value in smiles]
    decoded, error = decode_tokens([*tokens, 2, 0, 0])
    assert decoded == smiles
    assert error == ""


def test_frozen_balanced_merge_order() -> None:
    result = proportional_merge(
        [("beam", 0), ("beam", 1)],
        [("sample", 0), ("sample", 1)],
    )
    assert result == [
        ("beam", 0),
        ("sample", 0),
        ("beam", 1),
        ("sample", 1),
    ]


def test_sampling_seed_matches_step2d_definition() -> None:
    molecule_hash = hashlib.sha256(b"CCO").hexdigest()
    digest = hashlib.sha256(
        "\x1f".join(
            (
                str(GLOBAL_SEED),
                SAMPLING_PHASE,
                SAMPLE_POOL_NAME,
                molecule_hash,
            )
        ).encode("utf-8")
    ).hexdigest()
    expected = int(digest[:16], 16) % (2**63 - 1)
    assert sample_seed(molecule_hash) == expected


def test_candidate_filter_is_valid_unique_and_excludes_seed() -> None:
    config = json.loads(
        (REPOSITORY_ROOT / "inference" / "models" / "resolved_config.json")
        .read_text(encoding="utf-8")
    )
    seed_smiles = "CCO"
    seed_hash = hashlib.sha256(seed_smiles.encode("utf-8")).hexdigest()
    raw = (
        ("CCN", ""),
        ("CCN", ""),
        (seed_smiles, ""),
        ("C1", ""),
        ("c1ccccc1", ""),
        ("", "missing_eos"),
    )
    proposals = [
        Proposal(
            proposal_rank=index,
            source_kind="sample",
            source_rank=index,
            raw_smiles=smiles,
            token_error=error,
            decoder_log_probability=-float(index),
            generated_length=len(smiles) + 1,
        )
        for index, (smiles, error) in enumerate(raw, start=1)
    ]
    rows, stats = filter_candidates(
        proposals,
        seed_input_row=1,
        seed_id="ethanol",
        seed_canonical_smiles=seed_smiles,
        seed_hash=seed_hash,
        resolved_config=config,
        include_seed=False,
    )
    assert [row["canonical_smiles"] for row in rows] == ["CCN", "c1ccccc1"]
    assert len({row["canonical_smiles"] for row in rows}) == len(rows)
    assert all(0.0 <= row["morgan_tanimoto"] <= 1.0 for row in rows)
    assert stats["raw_proposals"] == 6
    assert stats["token_decodable_proposals"] == 5
    assert stats["policy_accepted_proposals"] == 4
    assert stats["duplicate_or_redundant_proposals"] == 1
    assert stats["excluded_seed_identity"] == 1
    assert stats["retained_unique_candidates"] == 2


def test_seed_filenames_are_safe_and_collision_resistant() -> None:
    name = safe_seed_filename(7, "../../A compound", "a" * 64)
    assert name == "seed-000007-A-compound-aaaaaaaa.csv"
    assert "/" not in name


def test_decode_defaults_to_frozen_thousand_proposal_prefix() -> None:
    args = build_parser().parse_args(["decode"])
    assert args.proposal_budget == 1000
    assert args.include_seed is False
