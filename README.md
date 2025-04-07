# OPENPACK

# SHL_2025_Qualifiers
# BUN_BO
This repository contains my solution for the **SHL_2025_Qualifiers** challenge, using the **OpenPack dataset S1** to predict the `operation` based on sensor data. **Note:** The `action` column is not used (or included) during training.

## Author
LeQuang

## Date
28/03/2025

## 1. Introduction

- **Objective:** Build a model to predict the `operation` based on sensor data.
- **Dataset:** [OpenPack dataset S1](https://open-pack.github.io/)
- 
  The dataset that I use could be downloaded from [Google Drive](https://drive.google.com/drive/u/2/folders/1aaJnRY8hiZWE5mJLxgonwlWk_gn8RZQ1).
- **Context:** This submission is for the qualifiers round by team **BUN-BO** for SHL 2025.

## 2. Repository Structure

```bash
.
├── notebooks/
│   ├── 1. Overview_data.ipynb      # Overview and initial visualization of the data.
│   ├── 2. Gen_data.ipynb           # Data generation/aggregation and initial preprocessing.
│   ├── 3. Dataset.ipynb            # Dataset creation: splitting into train/test/validation, statistical analysis.
│   └── 4. Train_val(File2_3).ipynb  # Model training (RF, LG, GB,KNN, XGBoost) and evaluation.
├── data/
│   ├── raw/                # Raw data (place the data downloaded from Google Drive here).
│   └── processed/          # Processed data after preprocessing.
├── src/                    # (Optional) Contains additional scripts if separated from notebooks.
├── requirements.txt        # List of required Python libraries. 
└── README.md
