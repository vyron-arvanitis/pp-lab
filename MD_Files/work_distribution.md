# Work Distribution

This document summarizes the contributions of each team member in the lab project.

---

## Vyron

- Wrote the introduction to the **Belle II experiment** and the problem tackled in this lab.  
- Methodology-> The dataset, transforming feature values, Unecessary features, Model architectures
- Models -> Transformer
- Results ->ROC curves and speedup results
- Evaluated the **Transformer model**.  
- Investigated:  
  - Feature **normalization**  
  - **Redundant features** in the dataset  
  - The effect of **reversing GCN with DeepSet Layers**  
- Implemented and included final **configuration parameters**.
- Speedup model comparison and evaluation of all models
- Code cleanup
- Docstrings

---

## Paul

- Wrote the introduction to **Deep Sets** and **Graph Neural Networks**.  
- Implemented and evaluated:  
  - **DeepSet model** 
  - **DeepSet + GCN model**
- Refactored:  
  - All models to use a unified `OutputLayer` class  
  - The main notebook into a standalone Python script (`workbook.py`)  
- Investigated the impact of increasing the number of **GCN** and **Linear layers**.
- Speedup model comparison and evaluation
- Code cleanup

---

## Emanuele

- Implemented and evaluated the **MLP (Multi-Layer Perceptron)** model with normalization.  
- Investigated the effect of **input masking** on performance.  
- Explored the internal behavior of the **embedding layer for PDG IDs**, analyzing whether similar particles are mapped to similar embeddings.

---

## Moritz

- Collaborated with Aggelos on creatubg the **Avenger (optimal)** and **Transformer** models.  
- Investigated **overfitting symptoms** and proposed effective **regularization strategies**.  
- Explored and tested **alternative model architectures**.  
- Docstrings


---

## Aggelos

- Investigated **cylindrical symmetry** in the input data and its modeling implications.  
- Collaborated with Moritz on evaluating the **Avenger (optimal)** model.  
- Tested various sequences of **GCN** and **Linear layers**, identifying the most effective combinations.  
- Contributed to **architectural improvements** based on approximate symmetries in the dataset.
