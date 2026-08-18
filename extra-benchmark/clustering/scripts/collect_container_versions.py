#!/usr/bin/env python3
"""Record Python and installed-distribution versions for every frozen model SIF."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from benchmark_io import BENCHMARK_DIR, atomic_write_json, atomic_write_text, load_protocol, sha256_file


QUERY = r"""
import json
import platform
from importlib.metadata import distributions

packages = {}
for distribution in distributions():
    name = distribution.metadata.get("Name") or distribution.metadata.get("Summary") or "UNKNOWN"
    packages[str(name)] = str(distribution.version)
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
}, sort_keys=True))
"""


def main() -> None:
    protocol = load_protocol()
    primary_order = protocol["models"]["primary_order"]
    model_to_image = {
        model: protocol["models"][model]["container"]
        for model in primary_order
    }
    base_image = protocol["models"]["gmolai"]["container"]
    model_to_image.update({"morgan_count": base_image, "descriptor13": base_image})
    expected_hashes = {
        record["container"]: record["container_sha256"]
        for record in protocol["models"].values()
        if isinstance(record, dict) and record.get("container")
    }
    image_labels: dict[str, str] = {}
    for model, image in model_to_image.items():
        image_labels.setdefault(image, "base" if image == base_image else model)
    output_directory = BENCHMARK_DIR / "audit" / "container_packages"
    inventories = {}
    for image, label in image_labels.items():
        completed = subprocess.run(
            ["apptainer", "exec", "--cleanenv", image, "python", "-c", QUERY],
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(completed.stdout)
        lines = [f"python=={payload['python']}"]
        lines.extend(f"{name}=={version}" for name, version in payload["packages"].items())
        inventory_path = output_directory / f"{label}.txt"
        atomic_write_text(inventory_path, "\n".join(lines) + "\n")
        inventories[label] = {
            "image": image,
            "image_sha256": expected_hashes[image],
            "python": payload["python"],
            "platform": payload["platform"],
            "installed_distributions": len(payload["packages"]),
            "inventory": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "models": sorted(model for model, candidate in model_to_image.items() if candidate == image),
        }
    result = {
        "schema_version": 1,
        "status": "ok",
        "method": "importlib.metadata distributions queried inside each immutable SIF",
        "inventories": inventories,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = BENCHMARK_DIR / "audit" / "container_packages.json"
    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
