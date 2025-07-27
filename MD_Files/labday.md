# Your Tasks for the lab days
---

## Lab day 1

**Task 1: Task-distribution file, organisation of gitlab**  
Alongside your documentation, please submit a file which shows who of you has contributed which part of the tasks.
Throughout the lab you work in individual branches but your final submission of all relevant files should be together in one branch marked as submission.

**Task 2: Use the PDG ids of the particles as an additional feature (group work)**
Fit the DeepSet example from [dataset_and_models.ipynb](dataset_and_models.ipynb) again, but use both the PDG ids and the rest of the particle features as input. The Section [Embedding layers and multiple inputs](dataset_and_models.ipynb#Embedding-layers-and-multiple-inputs) shows an example how to do this. How much does this improve the performance? You can use [workbook.ipynb](workbook.ipynb) as a starting point.

**Task 3: Develop a Graph Network (group work)**
Encorporate `GCN` layers into a model similar to the DeepSet example from [dataset_and_models.ipynb](dataset_and_models.ipynb), e.g. by replacing one or multiple of the per-item `Linear` layers by `GCN` layers. Note: The model needs the adjacency matrices as an additional input! You can implement this in a jupyter notebook or a python script.

---

## Homework between lab day 1 and lab day 2

Can you find a configuration with better performance? You may need to run on a slightly larger dataset, e.g. use all row groups `[0, 1, 2, 3]` (400k events) when loading the data.

**Further tasks (individual)**
Make sure to contribute at least to one way to potentially improve the model or making a comparison. Discuss in the group and distribute tasks. Some suggestions:
* Does it help to increase the number of layers - both `Linear` and `GCN`. Which sequence of `GCN` and `Linear` layers works best?
* Is it possible to use a standard MLP as well? How?
* How important is the masking in this case?
* Likely your model will show signs of overfitting. How can this be avoided?
* By taking the average over all hidden states we may loose information on the size of the event (number of particles). How could this information be included?
* Do we need to transform the values for the particle features?
* Are there any redundant features in the dataset that could be removed?
* Could one make use of some (approximate) symmetries in the data?
* Both the aggregation over neighbors in the graph and over all items in the set are done with a sum - weighting all contributions equally. Could this be improved?
* Can you inspect the embedding layer for the PDG ids? E.g. are similar particles embedded to similar vectors?
* Ideas for different model architectures?

---

## Lab day 2

**Task 4: Evaluation (group work)**  
Evaluate the performance of the most promising models on an independent test data set (provided to you on the last day). 

**Task 5: Estimate the speedup**  
The network returns for each event that gets passed through it a number between 0 and 1 (because of the sigmoid activation in the last layer). This number can be interpreted as the "confidence" of the network that a given event will pass the skimming (1 being the highest and 0 the lowest). In order to decide which events make it to the detector simulation, one needs to set a threshold such that all events with a higher output value get passed on and the rest are dropped. This threshold should be chosen such that the speedup gained by using the NN is at a maximum. In addition to the assumptions listed in the following the speedup should only depend on the false and true positive rate (`fpr`, `tpr`). These on the other hand only depend on the threshold, such that you can plot the speedup against the threshold (or one of `fpr`, `tpr`) and determine a maximum.
First though, you should come up with an expression for the speedup, assuming that:

* The detector simulation and reconstruction (the parts you want to skip) take 100 times the computing time of the event generation (which needs to be run regardless): $t_\mathrm{sim} = 100 \cdot t_\mathrm{gen}$
* In the original simulation chain $5\%$ of events get selected by the skimming.
* You can neglect the time for NN inference.
* The speedup factor will be given by the ratio of the time it takes without the neural network filter to simulate a certain number of events that pass the skim to the time it takes with the network for the **same** number of true positive events.
* Hints:

  * passing $X$ events through the simulation chain takes $X \cdot (t_\mathrm{gen} + t_\mathrm{sim})$ without the NN
  * passing $X'$ events through the chain with the NN in place takes $X' \cdot t_\mathrm{gen} + P \cdot t_\mathrm{sim}$ (with $P$ being the number of events selected by the NN)
  * $X'$ will generally be larger than $X$ in the presence of false negatives (events rejected by the NN that would have been accepted by the skim) since more events need to be produced then to reach the same number of true positive events.


---

# Hints for development

The introductory notebook is quite long. You don't want to rerun every cell in there each time you change a parameter (most of them are merely for demonstration purposes anyway). That's why we provided you with a [workbook](workbook.ipynb) that contains only the necessary code to start building and training the NNs. If you would rather work with a blank notebook, it makes sense to copy utility functions into a separate python module and import them in the new notebook. This way you can reuse them for multiple different tests that you may want to put in separate notebooks.
For longer running trainings and more systematic tests it may be useful to write a small python script instead of using notebooks.

Make sure to
* define your Model(s) with configurable hyperparameters as functions or classes
* be able to save/load your model and all states necessary to evaluate this later on an independent test data set
