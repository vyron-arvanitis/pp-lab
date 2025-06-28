# TODO: [@Paul] write the workbook in .py 
# TODO: [@Paul] make models work with OutputLayer Class -> create a seperate branch

# TODO: [@paul] Does it help to increase the number of layers - both Linear and GCN?

# TODO: [@aggelos]Which sequence of GCN and Linear layers works best?

# TODO: [@emanuele] Is it possible to use a standard MLP as well? How?

# TODO: [@emanuele] How important is the masking in this case?

# TODO: [@moritz] Likely your model will show signs of overfitting. How can this be avoided?

# TODO: [@all] By taking the average over all hidden states we may lose information on the size of the event (number of particles). How could this information be included?

# TODO: [@vyron]Do we need to transform the values for the particle features (scale 0-1 or normalization)?

# TODO: [@vyron] Are there any redundant features in the dataset that could be removed (this one is nice)?

# TODO: [@all] Could one make use of some (approximate) symmetries in the data (this one is nice)?

# TODO: [@all] Both the aggregation over neighbors in the graph and over all items in the set are done with a sum - weighting all contributions equally. Could this be improved (this one is nice)?

# TODO: [@emanuele]Can you inspect the embedding layer for the PDG ids? E.g. are similar particles embedded to similar vectors (this one is nice)?

# TODO:[@all, moritz] Ideas for different model architectures?

