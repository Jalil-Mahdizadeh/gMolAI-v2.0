#!/usr/bin/env python3
"""Compare exact gMolAI graph preprocessing alternatives."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fast_graph import fast_featurize_molecule
from tune import read_smiles


def main() -> None:
    values = read_smiles(16_384)
    results = []
    for name, parser in (
        ("MolFromSmiles_default", lambda value: Chem.MolFromSmiles(value)),
        ("MolFromSmiles_quick_copy", lambda value: Chem.MolFromSmiles(value, sanitize=True)),
        ("MolFromSmiles_params", None),
    ):
        if name == "MolFromSmiles_params":
            parameters = Chem.SmilesParserParams()
            parameters.removeHs = False
            parameters.sanitize = True
            parser = lambda value, p=parameters: Chem.MolFromSmiles(value, p)
        started = time.perf_counter()
        atoms = 0
        for value in values:
            molecule = parser(value)
            x, edge_index, edge_attr = fast_featurize_molecule(molecule)
            atoms += len(x)
        elapsed = time.perf_counter() - started
        row = {
            "parser": name,
            "rows": len(values),
            "atoms": atoms,
            "seconds": elapsed,
            "rows_per_second": len(values) / elapsed,
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
