# Lab report

Your submitted code and your documentation both are part of your submission and will be graded.

Below we list the requirements of what should be part of your lab report. When in doubt, please reach out to your advisor.

- A basic introduction about the Belle II experiment and the problem you try to solve in this lab course.[@vyron]
- A basic introduction about Deep Sets and Graph neural networks.[@paul]
- A summary of the types of networks you were examining.[@emanuele: mlp,mask, @aggelos,@moritz transformer]
- 1. --deepset **NO norm**[@paul]
- 1. --deepset GCN **norm**[@paul]
- 1. --deepset combined w GCN **norm** [@paul: fix bug]
- 1. --mlp[@emanuele: make it work] **norm**
- 1. --Transformer **norm** [@vyron: evaluate transformer]
- 1. --optimal[@avenger model] **norm**[@moritz,aggelos: evaluate it]
- 1. [TODO:@vyron] include a { "embed_dim": 6,"dropout_rate": 0.17,"num_heads": 4,"num_layers": 2,"units": 32,"num_features"} **THIS IS DONE**



- A summary describing how you approached the respective tasks.[@vyron: normalization,features,reversed,@moritz: overfitting,other architectures,@aggelos: spherical symmetry]
- A summary of the results and a discussion on possible issues with the approach taken in this lab course. In particular you should discuss a potential systematic bias that could be introduced by this method.

**Your repository:**
- Jupyter notebook(s) and/or scripts that contain all the code required to reproduce the results. Check that the code actually runs!
- Jupyter notebook(s) and/or scripts summarising the results of your respective trials and a discussion of the results. You can summarize all trials in one notebook or have different notebooks for different trials.
- Make sure that your axes in your plots are labeled.
- Ensure that all neural network models are in a function or class respectively. Each function or class has a meaningful docstring describing the class.
