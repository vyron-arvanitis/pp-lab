from __future__ import annotations

import argparse
import json

from pp_lab.config import DEFAULT_DATASET_PATH, PDG_MAPPING_PATH, resolve_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate a PDG-to-index mapping from a parquet dataset.")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-path", default=str(PDG_MAPPING_PATH))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import awkward as ak
    import numpy as np

    data = ak.from_parquet(str(resolve_path(args.dataset_path)))

    if hasattr(data, "particles"):
        pdg_values = ak.flatten(data.particles.pdg).to_numpy()
    elif hasattr(data, "x") and hasattr(data.x, "pdg"):
        pdg_values = ak.flatten(data.x.pdg).to_numpy()
    else:
        raise ValueError("Could not find particle PDG values in the parquet dataset.")

    unique_pdg_ids = np.unique(pdg_values)
    mapping = list(zip(unique_pdg_ids.tolist(), range(1, len(unique_pdg_ids) + 1)))
    output_path = resolve_path(args.output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2)
    print(f"Wrote PDG mapping to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
