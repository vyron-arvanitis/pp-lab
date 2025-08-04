# Work Distribution

This document summarizes the contributions of each team member in the lab project.

The report was written by everybody equally. Individual work can be seen in section 4,
where everybody wrote the part of his own investigation. 

---

## Vyron

- Evaluated the **Transformer model**.  
- Investigated:  
  - Feature **normalization**  
  - **Redundant features** in the dataset  
  - The effect of **reversing GCN with DeepSet Layers**  
- Implemented and included final **configuration parameters**.
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

- Collaborated with Angelos on creating the **Optimal model** and **Transformer** models.  
- Investigated **overfitting symptoms** and proposed effective **regularization strategies**.  
- Explored and tested **alternative model architectures**.  
- Implemented an **optuna study** (optuna_study.py) to find the best hyperparameters for the optimal model

---

## Angelos

- Investigated **cylindrical symmetry** in the input data and its modeling implications.
- Collaborated with Moritz on evaluating the **optimal model**.  
- Tested various sequences of **GCN** and **Linear layers**, identifying the most effective combinations.  
- Investigated different activation functions 
- Code cleanup / Docstrings
