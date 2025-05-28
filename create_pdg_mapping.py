#!/usr/bin/env python

import json
import awkward as ak
import numpy as np

filename = "smartbkg_dataset_4k.parquet"
data = ak.from_parquet(filename)
unique_pdg_ids = np.unique(ak.flatten(data.x.pdg).to_numpy())
mapping = list(zip(unique_pdg_ids.tolist(), range(1, len(unique_pdg_ids) + 1)))
with open("pdg_mapping.json", "w") as f:
    json.dump(mapping, f)
