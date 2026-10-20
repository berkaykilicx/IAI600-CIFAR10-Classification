
# IAI600 - CIFAR-10 Image Classification Project: MLP vs. CNN

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete implementation, experimental setup, and evaluation for the **AI and Machine Learning Lab Project**, focusing on image classification using the standard **CIFAR-10** dataset. The project strictly adheres to the course scope, emphasizing data understanding, rigorous preprocessing, basic model design, and justified performance comparison between a Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN).

---

## 📂 Repository Structure

The project is organized in a modular structure to ensure reproducible experiments and seamless collaboration among group members:

```text
📦 IAI600-CIFAR10-Classification
 ┣ 📂 config/
 ┃ ┣ 📜 preprocessing_config.json   # Normalization parameters & hyperparameters
 ┃ ┗ 📜 split_indices.npz           # Fixed stratified train/validation split indices
 ┣ 📂 models/
 ┃ ┣ 📜 cnn_best_model.pth          # Restored best CNN model weights (state_dict)
 ┃ ┣ 📜 cnn_history.json            # Training & validation loss/accuracy history
 ┃ ┗ 📜 cnn_test_predictions.npz    # Unseen test set predictions & metrics
 ┣ 📜 CIFAR10_Project.ipynb         # Main Jupyter Notebook (EDA, MLP, CNN, Evaluation)
 ┣ 📜 data_preparation.py           # Shared data loader & normalization module
 ┣ 🖼️ cnn_learning_curves.png       # Training convergence & overfitting analysis plot
 ┣ 🖼️ cnn_confusion_matrix.png      # 10x10 Test Set Confusion Matrix heatmap
 ┣ 🖼️ cnn_correct_vs_misclassified.png # Qualitative error analysis visualization
 ┣ 📜 .gitignore                    # Excludes large raw datasets and local caches
 ┗ 📜 README.md                     # Project documentation
