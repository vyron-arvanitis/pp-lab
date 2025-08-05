import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from models import from_config
import seaborn as sns
from matplotlib import patheffects

# ----------------------------------------------------------------------------------
# Study of the embedding layer: are similar particles embedded to similar vectors? (yes!)
# ----------------------------------------------------------------------------------

# Setup paths
repo_root   = Path(".")
out_dir     = repo_root / "pdg_embedding_analysis"
out_dir.mkdir(exist_ok=True)

# Load model state_dict and config and instantiate the model class
model_dir   = Path("models_fulltrain/optimal_model_cylindrical")
ckpt_path   = model_dir / "state.pt"
config_path = model_dir / "config.json"
pdg_map_path = Path("pdg_mapping.json")
config = json.load(open(config_path))
model = from_config(config)
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

# Extract the embedding matrix
emb_weight = state["embedding_layer.weight"]

# Load PDG mapping, invert and exlude padding token
raw = json.load(open(pdg_map_path))  
pdg2idx = { str(pdg): idx for pdg, idx in raw }
idx2pdg = { idx: pdg for pdg, idx in raw if idx != 0 }
valid_idx = sorted(idx2pdg.keys())

for pdg_str, idx in pdg2idx.items():
    idx = int(idx)
    pdg  = int(pdg_str)
    if idx != 0:
        idx2pdg[idx] = pdg
valid_idx = sorted(idx2pdg.keys())

# Build and inspect cosine‐similarity matrix
E = emb_weight[valid_idx]             
E_norm = F.normalize(E, p=2, dim=1)   
S = E_norm @ E_norm.T              
print(f"Cosine‐sim matrix: mean={S.mean().item():.4f}, std={S.std().item():.4f}")

results_emb = {}
results_emb["cosine_mean_all"] = S.mean().item()
results_emb["cosine_std_all"]  = S.std().item()


# Off-diagonal statistics
# Masking
N = S.size(0)
eye = torch.eye(N, dtype=torch.bool)
mask = ~eye                
# Extract the off‐diagonal similarities
off_diag = S[mask]
# Compute mean and std
mean_off = off_diag.mean().item()
std_off  = off_diag.std().item()
results_emb["cosine_mean_off"] = off_diag.mean().item()
results_emb["cosine_std_off"]  = off_diag.std().item()
print(f"Off‐diagonal cosine‐sim mean = {mean_off:.4f}, std = {std_off:.4f}")


# example particles (alternatively use the 'particle' module)
pdg_names = {
    211:"π⁺", -211:"π⁻", 111:"π⁰",
     22:"γ",   13:"μ⁻",  -13:"μ⁺",
    321:"K⁺", -321:"K⁻",2212:"p",  -2212:"p̄"
}

# nearest neighbours function
def topk(pdg, k=5):
    idx = pdg2idx[str(pdg)]
    if idx == 0: raise ValueError("PDG=0 is padding")
    pos = valid_idx.index(idx)
    sim_row = S[pos].numpy()
    nn = np.argsort(sim_row)[- (k+1) :][::-1]
    out = []
    for j in nn:
        if j == pos: continue
        pdg_j = idx2pdg[ valid_idx[j] ]
        out.append(( pdg_j, sim_row[j], pdg_names.get(pdg_j, str(pdg_j)) ))
        if len(out) == k: break
    return out

# Antiparticle similarity score
pairs = [(p, -p) for p in (13,211,321,2212)]
scores = []
for p, pm in pairs:
    i = valid_idx.index(pdg2idx[str(p)])
    j = valid_idx.index(pdg2idx[str(pm)])
    key = f"sim_{p}_{pm}"
    results_emb[key] = S[i,j].item()
    
    # TODO: check if this is ok commented out or not!
    scores.append( S[i,j].item() )
    
# Save nearest‐neighbour top1 sims
for pdg in (211, -211, 111, 22, 13, -13):
    nn = topk(pdg, k=1)[0]
    results_emb[f"nn_sim_{pdg}"] = nn[1]

print("Avg antiparticle cosine sim:", np.mean(scores).round(4))
print("Avg π± sim:", topk(211,1)[0][1])
print("Avg p/p̄ sim:", topk(2212,1)[0][1])  
    
# Save results
df_res = pd.DataFrame([results_emb])
df_res.to_csv(out_dir/"pdg_embedding_results.csv", index=False)


# t-SNE visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca")
coords = tsne.fit_transform(E_norm.numpy())

highlight_codes = [13, 22, 111, 211, 321, 2212]
# cmap = cm.get_cmap("tab10", len(highlight_codes))
cmap = plt.get_cmap("tab10", len(highlight_codes))
code2color = {code: cmap(i) for i, code in enumerate(highlight_codes)}

plt.figure(figsize=(6,6))
plt.scatter(coords[:,0], coords[:,1],
            c="lightgray", s=8, alpha=0.4, label="_nolegend_")

for code in highlight_codes:
    mask = np.array([abs(idx2pdg[i]) == code for i in valid_idx])
    pts = coords[mask]
    plt.scatter(pts[:,0], pts[:,1],
                c=[code2color[code]], s=20, alpha=0.9,
                label=pdg_names.get(code, str(code)))
    
plt.title("t-SNE of PDG Embeddings", fontsize=14, pad=10)
plt.xticks([]); plt.yticks([])
plt.legend(title="Particle", loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig(out_dir/"pdg_embeddings_tsne.png", dpi=300)
plt.close()


# Histogram of off‐diagonals
plt.figure(figsize=(6,4))
plt.hist(off_diag, bins=50)
plt.xlabel("Cosine similarity")
plt.ylabel("Count")
plt.title("PDG embedding similarity distribution")
plt.savefig(out_dir/"pdg_embedding_hist.png", dpi=150)
plt.close()






