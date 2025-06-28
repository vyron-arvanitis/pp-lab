# TODO 01: [@Paul] Write the workbook in .py 
# TODO 02: [@Paul] Make models work with OutputLayer class → create a separate branch

# TODO 03: [@paul] Does it help to increase the number of layers — both Linear and GCN?

# TODO 04: [@aggelos] Which sequence of GCN and Linear layers works best?

# TODO 05: [@emanuele] Is it possible to use a standard MLP as well? How?

# TODO 06: [@emanuele] How important is the masking in this case?

# TODO 07: [@moritz] Likely your model will show signs of overfitting. How can this be avoided?

# TODO 08: [@all] By taking the average over all hidden states we may lose information on the size of the event (number of particles). How could this information be included?

# TODO 09: [@vyron] Do we need to transform the values for the particle features (scale 0–1 or normalization)?

# TODO 10: [@vyron] Are there any redundant features in the dataset that could be removed? (this one is nice)

# TODO 11: [@all] Could one make use of some (approximate) symmetries in the data? (this one is nice)

# TODO 12: [@all] Both the aggregation over neighbors in the graph and over all items in the set are done with a sum — weighting all contributions equally. Could this be improved? (this one is nice)

# TODO 13: [@emanuele] Can you inspect the embedding layer for the PDG IDs? E.g. are similar particles embedded to similar vectors? (this one is nice)

# TODO 14: [@all, moritz] Ideas for different model architectures?
