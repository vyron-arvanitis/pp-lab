# Preparation

This document contains instructions and background information to prepare for your lab course.

## Computational environment preparation

- Connect to jupyterhub at: [jupyter.physik.uni-muenchen.de](https://jupyter.physik.uni-muenchen.de). Further instructions, in particular regarding the necessary two factor identification can be found [here](https://www.en.it.physik.uni-muenchen.de/bekanntmachungen/new-service-jupyterhub/index.html). Note that the more resources you request for your session the harder it is to allocate them. Hence it is recommended to use only resources you actually need. As one often does not know the resources needed start with small resources which typically give you almost immediate access. When your jupyter kernel keeps crashing, this is a sign that you are running out of memory. It's good practice to shutdown your jupyterhub server when you are done. You can do that by selecting `File`&rarr;`Hub Control Panel`&rarr;`Stop my Server`. For this course you should select the `python/3.10-2022.08` environment.
- Clone repository on remote machines and generate your personal branch (using the terminal, a terminal window can be opened within your running jupyterhub):
```
git clone <this-repository-url>
```

You can find the url by clicking on the Clone button on the gitlab repository:

![](figures/git_clone_url2.png)

Initially, everyone works in their own branch of this repository. To do this, you can run:
```
cd <code-directory>
git branch your_individual_branch_name
git checkout your_individual_branch_name
```
Ensure that you can push your work to the repository. For instance, you can push your changes as follows from a terminal:
```
# ensure you are working in the correct branch
# ensure you are in the directory of the repository
git add .
git commit -m "your message about your changes here"
git push
```
You should see your changes on our gitlab server. If you are new to git, also have a look at our [git tips](git_tips.md).

## Physics context

![](figures/SuperKEKB-BelleII.jpg)

In this lab course you are going to use simulated data of the [Belle II Experiment](https://www.belle2.org/) at the SuperKEKB accelerator in Tsukuba, Japan. SuperKEKB is a so called *B-Factory*, a particle accelerator/collider specialized to produce B-mesons which are then studied in the Belle II detector. To prepare for the lab course, research the answers to the following questions:

* What is a B-Factory?
* What advantages do e+/e- colliders have compared to pp colliders?
* What particles does a B-meson typically decay into?
* Which particles can we actually see in the detector?


## Problem description

The high luminosity and correspondingly the high rate of data allow for the analysis of very rare processes. But this also means we need large simulated samples to optimize analysis strategies and to produce statistical models of the data. Have a look through the following section of the Belle II software online book:

https://software.belle2.org/development/sphinx/online_book/fundamentals.html

and try to answer the following questions:

* What are the different stages a simulated event has to pass through before it can be compared to real data?
* Which stages are computationally most expensive?

To make this process computationally more efficient, the approach that should be studied in this lab course is to train a neural network to predict which events correspond to ones that are going to be selected finally already early in the simulation chain:

**Smart Background Simulation**

![](figures/MC_data_flow-keep-discard_NN.png)

This is especially relevant for estimating the rate of processes that do not correspond to the actual signal that is supposed to be studied - the *background*. Since we want to reduce the background as much as possible, naturally we discard a large amount of these events.

To train a *smart background filter* in this lab course we provide a labelled dataset where the label indicates whether or not an event passes the so called [FEI skim](https://software.belle2.org/development/sphinx/skim/doc/02-physics.html#module-skim.WGs.fei). Only events where at least one B-meson could be reconstructed pass the selection. In this dataset only neutral B-mesons are simulated and the reconstruction is performed by applying the [Full Event Interpretation](https://software.belle2.org/development/sphinx/analysis/doc/FullEventInterpretation.html#fulleventinterpretation) Algorithm to reconstruct hadronic decays:

![](figures/fei_hierarchy2.png)

This hierarchical reconstruction algorithm is also based on machine learning. Specifically it makes use of Boosted Decision Trees to predict the probabilities that intermediate reconstructed particles correspond to certain decays. Since this is all based on detector level information and due to the complexity of the algorithm it is difficult to predict which events will be reconstructible just from the generated particle without detector interaction alone. Therefore we need to resort to machine learning to solve our smart background simulation problem.

The FEI Skim is not targetted at a specific signal selection, but for many analysis a reconstructed hadronic *tag* B-meson is required, so this serves as a generic filter. Only around 5% of events survive this filter, so there is already a lot to be gained by increasing this number with a smart pre-filtering.

To learn more about the dataset, the nescessary preprocessing and possible building blocks for a machine learning model, continue with [dataset_and_models.ipynb](dataset_and_models.ipynb)

## References

To read more about the context of this lab course and to answer the questions posed before you can look through the following material:

Physics and Belle II:  
* PhD Thesis of James Kahn: https://edoc.ub.uni-muenchen.de/24013
(first development of the Smart Background method at Belle II, also the Introductory chapters to Belle II could be useful)
* Simulation Chain at Belle II: https://software.belle2.org/development/sphinx/online_book/fundamentals.html
* Paper "The Physics of the B Factories": https://arxiv.org/abs/1406.6311 (if you want to know more about B-factories)
* Paper "The Belle II Physics Book": https://arxiv.org/abs/1808.10567 (physics researched at Belle II)
* PDGlive (Particle Data Group): https://pdglive.lbl.gov/Viewer.action (information on particles and decays)

Basic information about Neural Networks and Deep Learning:
* 3Blue1Brown video series: https://www.3blue1brown.com/topics/neural-networks (graphic introduction to neural networks)
* IBM: https://www.ibm.com/topics/neural-networks (very short intro)
* Victor Zhou "Machine Learning for Beginners": https://victorzhou.com/blog/intro-to-neural-networks/ (more detailed lead-in)
* Wikipedia: https://en.wikipedia.org/wiki/Artificial_neural_network (also a good introduction)
* The Deep Learning Book: https://www.deeplearningbook.org/ (introduction to everything machine/deep learning)
* Kaggle Course "Intro to deep learning" https://www.kaggle.com/learn/intro-to-deep-learning (interactive online course, with excercises)

Special information about Deep Sets and GCNs:
* Article: Thomas Kipf, "Graph Convolutional Networks": https://tkipf.github.io/graph-convolutional-networks/
* Corresponding Paper: https://arxiv.org/abs/1609.02907 (interesting for the math-oriented)
* Paper "Deep Sets": https://arxiv.org/abs/1703.06114 (shows principle and power of Deep Sets)
* Explanation of embeddings in neural networks: https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526
* Explanation and implementation of masking and padding in Keras: https://keras.io/guides/understanding_masking_and_padding/

General resources:
* https://www.tensorflow.org/api_docs/python/tf/all_symbols (Tensorflow documentation)
* https://keras.io/api/ (Keras API reference)
* https://git-scm.com/docs (Git documentation)
* https://numpy.org/doc/stable/reference/index.html (numpy API reference)
* https://pandas.pydata.org/docs/reference/index.html (pandas API reference)
* https://matplotlib.org/stable/api/index.html (matplotlib reference)
* Python: just search, often answers on StackOverflow

Binary Classification: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers  
(definition of many important classification quantities like fpr, accuracy, etc.)

