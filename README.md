# Higgs Boson Classification (PyTorch)

Binary classification predicting the existence of the Higgs Boson particle against background noise using a PyTorch neural net, evaluated with ROC–AUC and compared to classical ML baselines.

## What’s inside
- Data preprocessing + train/val/test split
- PyTorch model training (MLP)
- Evaluation: ROC–AUC, ROC curve, confusion matrix
- Baseline comparison (e.g., logistic regression)

## Results
- PyTorch MLP: ROC–AUC = 0.80
- Baseline (LogReg): ROC–AUC = 0.68

## Run
```bash
pip install -r requirements.txt
jupyter notebook
