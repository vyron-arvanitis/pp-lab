from __future__ import annotations

import argparse
import json
from pathlib import Path

from pp_lab.config import EMBEDDING_ANALYSIS_DIR, PDG_MAPPING_PATH, resolve_path

PDG_NAMES = {
    211: "pi+",
    -211: "pi-",
    111: "pi0",
    22: "gamma",
    13: "mu-",
    -13: "mu+",
    321: "K+",
    -321: "K-",
    2212: "p",
    -2212: "pbar",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze learned PDG embeddings from a trained model.")
    parser.add_argument("--model-dir", default="models_fulltrain/optimal_model_cylindrical")
    parser.add_argument("--pdg-mapping-path", default=str(PDG_MAPPING_PATH))
    parser.add_argument("--output-dir", default=str(EMBEDDING_ANALYSIS_DIR))
    parser.add_argument("--perplexity", type=float, default=30)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.manifold import TSNE

    from pp_lab.models import from_config

    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)

    model = from_config(config)
    state = torch.load(model_dir / "state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    emb_weight = state["embedding_layer.weight"]
    with resolve_path(args.pdg_mapping_path).open(encoding="utf-8") as handle:
        pdg2idx = {int(pdg): int(idx) for pdg, idx in json.load(handle)}
    idx2pdg = {idx: pdg for pdg, idx in pdg2idx.items() if idx != 0}
    valid_indices = sorted(idx2pdg)

    embeddings = emb_weight[valid_indices]
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = normalized @ normalized.T

    results = {
        "cosine_mean_all": similarity.mean().item(),
        "cosine_std_all": similarity.std().item(),
    }

    mask = ~torch.eye(similarity.size(0), dtype=torch.bool)
    off_diag = similarity[mask]
    results["cosine_mean_off"] = off_diag.mean().item()
    results["cosine_std_off"] = off_diag.std().item()

    def topk(pdg: int, k: int = 5):
        pos = valid_indices.index(pdg2idx[pdg])
        row = similarity[pos].numpy()
        neighbours = np.argsort(row)[-(k + 1) :][::-1]
        output = []
        for neighbour in neighbours:
            if neighbour == pos:
                continue
            neighbour_pdg = idx2pdg[valid_indices[neighbour]]
            output.append((neighbour_pdg, row[neighbour], PDG_NAMES.get(neighbour_pdg, str(neighbour_pdg))))
            if len(output) == k:
                break
        return output

    for particle, anti_particle in ((13, -13), (211, -211), (321, -321), (2212, -2212)):
        i = valid_indices.index(pdg2idx[particle])
        j = valid_indices.index(pdg2idx[anti_particle])
        results[f"sim_{particle}_{anti_particle}"] = similarity[i, j].item()

    for pdg in (211, -211, 111, 22, 13, -13):
        results[f"nn_sim_{pdg}"] = topk(pdg, k=1)[0][1]

    pd.DataFrame([results]).to_csv(output_dir / "pdg_embedding_results.csv", index=False)

    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=0, init="pca")
    coords = tsne.fit_transform(normalized.numpy())
    highlight_codes = [13, 22, 111, 211, 321, 2212]
    cmap = plt.get_cmap("tab10", len(highlight_codes))
    code_to_color = {code: cmap(i) for i, code in enumerate(highlight_codes)}

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=8, alpha=0.4, label="_nolegend_")
    for code in highlight_codes:
        subset = np.array([abs(idx2pdg[index]) == code for index in valid_indices])
        points = coords[subset]
        plt.scatter(points[:, 0], points[:, 1], c=[code_to_color[code]], s=20, alpha=0.9, label=PDG_NAMES.get(code, str(code)))
    plt.title("t-SNE of PDG Embeddings")
    plt.xticks([])
    plt.yticks([])
    plt.legend(title="Particle", loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "pdg_embeddings_tsne.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(off_diag.numpy(), bins=50)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title("PDG embedding similarity distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "pdg_embedding_hist.png", dpi=150)
    plt.close()
    print(f"Saved embedding analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
