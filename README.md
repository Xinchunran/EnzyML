# EnzyML
The machine learning based method in predicting enzymes property

## Projects
# Step 1
1. benchmarking the models which using to predict binding affinity of protein-ligand
2. benchmarking the models which using to predict kcat

# Step 2  Huggingface AutoML as a major template of the EnzyML
1.  The current models do not have the Enzyme encoder.
2. The current models only screening the Gradient Boost, the Random forest 
3. A good workflow for the [substrates](https://robert.readthedocs.io/en/latest/Examples/full_workflow/full_workflow.html) (introduce by Heidi and developed by Juan and inspired by Dr. Paton)

The current model contain the algorithm catBoost, RF, XGBoost for protein sequence one hot encoding and the ESM embedding. Later we need to incorporate the graph embedding for the active sites  
