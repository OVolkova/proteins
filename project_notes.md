
# Notes:

There are 3 models:
- GCN
- Attention like in transformer model for NLP
- 3N attention model that has attention for edges

If to train just as is for node classification, the best model is GCN: I think just because it trains faster.

For both attention models training for node classification is unstable with high fluctuations in loss.

Pretraining task includes:
- link prediction: Average Precision (AP) 
- node features prediction: Mean Squared Error (MSE)
- edge features prediction: Mean Squared Error (MSE)

Both attention models are better for link prediction task and edge classification task than GCN.
GCN is better on node features prediction.
Metrics for node features and for edge features are improving with the same speed for attention models.
Metric for link prediction is the best in 3N attention model now, but simple attention is close to it and going up and trained for fewer epochs.

GCN does not include any transformation of the input features for edges and each layer takes edges features as is.
Both attention model apply transformation to the input features for edges.
Simple attention model applies only Feed Forward Neural Network (FFNN) to the input features for edges.
3N attention model applies attention mechanism to the input features for edges.

# Next steps:
- rename repo and project
- GAT model as is and with transformation for edges like in simple attention model
- add fine-tuning for node classification task
- add another datasets for experiments:
  - on link prediction task
  - on graph prediction task
- check other models and techniques for graph problems
- check architecture for link prediction problems

