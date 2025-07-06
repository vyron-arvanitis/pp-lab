# Work Distribution

This document summarizes the contributions of each team member in the lab project.

---

## Vyron

- Wrote the introduction to the **Belle II experiment** and the problem tackled in this lab.  
- Evaluated the **Transformer model**.  
- Investigated:  
  - Feature **normalization**  
  - **Redundant features** in the dataset  
  - The effect of **reversing feature order**  
- Implemented and included final **configuration parameters**.

---

## Paul

- Wrote the introduction to **Deep Sets** and **Graph Neural Networks**.  
- Implemented and evaluated:  
  - **DeepSet model** 
  - **DeepSet + GCN model**   
  - **DeepSet + GCN model**
- Refactored:  
  - All models to use a unified `OutputLayer` class  
  - The main notebook into a standalone Python script (`workbook.py`)  
- Investigated the impact of increasing the number of **GCN** and **Linear layers**.

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


---

## Aggelos

- Investigated **spherical symmetry** in the input data and its modeling implications.  
- Collaborated with Moritz on evaluating the **Avenger (optimal)** model.  
- Tested various sequences of **GCN** and **Linear layers**, identifying the most effective combinations.  
- Contributed to **architectural improvements** based on approximate symmetries in the dataset.
