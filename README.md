# Higgs Boson Classification (PyTorch)

Binary classification of simulated LHC collision events: **Higgs signal vs background**, using a PyTorch neural network, evaluated with ROC–AUC and compared to classical ML baselines.

## What’s inside
- Data preprocessing + train/val/test split
- PyTorch model training
- Evaluation: ROC–AUC, ROC curve, confusion matrix
- Baseline comparison (e.g., logistic regression)

## Results
- PyTorch MLP: ROC–AUC = 0.80
- Baseline (LogReg): ROC–AUC = 0.68

## Dataset
This project uses the [Higgs Boson dataset (HIGGS)](https://archive.ics.uci.edu/dataset/280/higgs). The dataset is not included in this repo due to size.

**To run the notebook:**
1. Download the dataset file as `HIGGS.csv.gz`.
2. Place it in the **repo root** (same folder level as `README.md`).
3. The notebook loads the first 1,000,000 rows by default.

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/HIGGS_BOSON_PYTORCH.ipynb
