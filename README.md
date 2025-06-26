# Federated Learning Project 2025

## Project Summary

Federated Learning (FL) enables decentralized model training without sharing raw data, making it well-suited for privacy-sensitive applications. In this project, we investigate convergence challenges in FL and propose pTaLoS, a hybrid strategy that combines global model editing (TaLoS) with client-level personalization (pFedEdit).

Our approach is evaluated on ViT-S/16 pretrained with DINO and tested on CIFAR-100.

## Repository Structure

- `notebooks/`  
  Jupyter notebooks for running all experiments—centralized and federated—under different settings. In particular
  
  - `PRODUCTION_CENTRALIZED.ipynb`: Centralized training **without** model editing.
  
  - `PRODUCTION_CENTRALIZED_Editing.ipynb`: Centralized training **with** model editing (TaLoS-based pruning and fine-tuning)
  
  - `PRODUCTION_FL.ipynb`: Federated Learning **without** model editing (standard FedAvg baseline).
  
  - `PRODUCTION_FL_Editing.ipynb`: Federated Learning **with** model editing, including TaLoS and pFedEdit strategies.
  
  Each notebook provides a configurable section where all relevant hyperparameters for the experiment can be specified.

- `utils/`  
  Supporting Python modules used across notebooks, including:
  
  - `editing.py`: Implements model editing and sensitivity-based pruning.
  - `strategies.py`: Aggregation strategies such as FedAvg and YOGI
  - `clients.py`: Custom client classes for FL, including support for pFedEdit and pruning.
  - `data_utils.py`: Provides functions to split datasets in **IID** and **non-IID** ways, crucial for simulating statistical heterogeneity in FL.
  - `data_preprocessing.py`: Includes image preprocessing utilities such as data augmentation, normalization, and dataset preparation for CIFAR-100.
  - `wandb_logger.py`: Contains the logic and setup for **logging metrics, losses, and hyperparameters** to **Weights & Biases**, allowing real-time monitoring of training and evaluation.

## Client Configuration: `TalosPFedEdit`

To test the `TalosPFedEdit` mode, simply use the `TalosPFedEdit` client in your experiment script and set the `mode` to `full` inside the `talos_config`.

## Quick Start (Colab)

1. Open any notebook from the [`notebooks/`](./notebooks) folder in **Google Colab**.
2. Upload the required utility files from the [`utils/`](./utils) folder to your Colab session.
3. Run all cells. 

## Results

Training logs and visualizations are available on Weights & Biases:  
 [View Experiments](https://wandb.ai/polito-fl/federated-learning-project/overview)

## References

Key references and papers are included in the [`theory/papers/`](./theory/papers/) folder.
