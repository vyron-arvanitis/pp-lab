# Smart Background AI Lab course

**Abstract:** When searching for rare signals in particle collision experiments the estimation of background processes plays a crucial role. This requires a huge amount of computationally expensive Monte Carlo simulations. One aspect that makes this processes very inefficient is that events that do not pass the search selection criteria have to be run through the whole simulation chain since the selection quantities are only known after that.

The goal of this lab course is to predict which events pass a certain selection already early in the simulation chain, in particular before running an expensive detector simulation and event reconstruction. You will use simulated data from the Belle II experiment that contains decay trees of simulated particles, early in the simulation chain, only containing the physics process, but no interaction with the detector. Your prediction can be used e.g. for importance sampling of the more computationally expensive steps in the simulation chain.

Particle decays are stochastic processes: the types, numbers and properties of produced particles vary for each decay. In order to handle such a very variable feature list more sophisticated ML algorithms such as Graph-NN are applied.

**ML packages:** PyTorch  
**Further packages:** Numpy, Pandas, Awkward Array  
**Input data:** Variable-length lists of particle features with decay relations

**Preparation:** You are expected to work through this [preparation](preparation.md) document and the [dataset and models notebook](dataset_and_models.ipynb).

You are also expected to read through the following two guides:

* [**Guide for the lab days**](labday.md)
* [**Guide for the lab report**](labreport.md)
