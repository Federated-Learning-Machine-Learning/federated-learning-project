# Federated Learning Project 2025 

## Project Summary

Federated Learning (FL) enables decentralized model training without sharing raw data, making it well-suited for privacy-sensitive applications. In this project, we investigate convergence challenges in FL and propose pTaLoS, a hybrid strategy that combines global model editing (TaLoS) with client-level personalization (pFedEdit).

Our approach is evaluated on ViT-S/16 pretrained with DINO and tested on CIFAR-100.
## 📁 Repository Structure

- `notebooks/`  
  Jupyter notebooks for running all experiments—centralized and federated—under different settings.

- `utils/`  
  Supporting Python modules used across notebooks, including:
  - `editing.py`, `strategies.py`, `clients.py`, etc.

- `theory/`: Contains research papers and project planning resources.
  - `papers/`: Research PDFs grouped by topic
  - `Project Guidelines.pdf`: Internal project roadmap and goals


## 🚀 Quick Start (Colab)

1. Open any notebook from the [`notebooks/`](./notebooks) folder in **Google Colab**.
2. Upload the required utility files from the [`utils/`](./utils) folder to your Colab session.
3. Run all cells. 

## 📈 Results

Training logs and visualizations are available on Weights & Biases:  
👉 [View Experiments](https://wandb.ai/polito-fl/federated-learning-project/overview)


## 📚 References

Key references and papers are included in the [`theory/papers/`](./theory/papers/) folder.
