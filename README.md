# OGB-Proteins dataset experiments

The goal of this project is to predict protein functions. 
We use the [OGB-Proteins](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins) dataset. 
The dataset contains 132.534 proteins (nodes) with 79.122.504 edges with 8 features. Nodes do not have features.
Each protein is associated with 112 functions (labels). 
The goal is to predict the functions of proteins that do not have labels. 
Proteins are from 8 species. The dataset is split into train, 
validation and test sets with 6 species in train split and 1 in validation and 1 in test.


The goal of this project is to train graph NN to predict protein functions:
- base existing model from torch geometrics
- implement from scratch model from paper
- implement sparse 3N attention version
- report and compare results
