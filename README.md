# Higgs Boson Classification (PyTorch)

In this project I build a binary classifier to distinguish simulated Higgs boson collision events from background noise using the public **HIGGS** dataset. I chose **PyTorch** to get hands-on experience with deep learning frameworks and to better understand an end-to-end ML workflow.

By implementing the neural network and training loop from scratch, I practiced forward passes, backpropagation, gradient-based optimization, and evaluation using standard classification metrics.

## What's inside
- Data loading and exploration (EDA)
- Feature preprocessing and scaling
- Logistic regression baseline
- Feed-forward neural network in PyTorch (custom training loop)
- Hyperparameter tuning (learning rate sweep)
- Evaluation: ROC–AUC, ROC curve, confusion matrix, classification report

## Results
- PyTorch neural network (ROC–AUC): **0.80**
- Baseline (logistic regression) (ROC–AUC): **0.68**

> Note: ROC–AUC values are reported from the split used in the notebook (train/validation/test). See the notebook for details.

## Dataset
This project uses the [Higgs Boson dataset (HIGGS)](https://archive.ics.uci.edu/dataset/280/higgs). The dataset is not included in this repo due to size.

To run the notebook:
1. Download the dataset file as `HIGGS.csv.gz` from Kaggle/UCI.
2. Place it in the **repo root** (same folder level as `README.md`).
3. The notebook loads the first **1,000,000** rows by default.

## Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/HIGGS_BOSON_PYTORCH.ipynb
