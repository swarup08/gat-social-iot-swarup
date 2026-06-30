# GAT-based Secure Social IoT Framework

![Status](https://img.shields.io/badge/Project-Research-blue)
![Milestone](https://img.shields.io/badge/Milestone-2%20Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyG](https://img.shields.io/badge/PyTorch%20Geometric-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Author

**Swarup Sarkar**

Research Area:
- Graph Neural Networks (GNN)
- Social Internet of Things (SIoT)
- Explainable AI (XAI)
- Reinforcement Learning
- Cyber Security

---

# Overview

This repository presents an end-to-end research framework for securing **Social Internet of Things (Social IoT)** networks using **Graph Attention Networks (GAT)**, **Graph Pruning**, **Reinforcement Learning (RL)** and **Explainable AI (XAI)**.

The framework first learns communication importance using Graph Attention Networks and then removes less important communication links to reduce botnet propagation while preserving the essential structure of the Social IoT network.

This repository is being developed milestone-wise as part of an ongoing research project.

---

# Research Objectives

The major objectives of this research are:

- Construct a realistic Social IoT graph
- Simulate botnet propagation
- Learn edge importance using Graph Attention Networks
- Analyze graph attention behaviour
- Perform static graph pruning
- Develop RL-based dynamic graph pruning
- Explain pruning decisions using Explainable AI
- Build a secure and trustworthy Social IoT framework

---

# Project Roadmap

| Milestone | Description | Status |
|------------|-------------|--------|
| Milestone-1 | Social IoT Graph Construction | ✅ Completed |
| Milestone-2 | GAT Training & Static Graph Pruning | ✅ Completed |
| Milestone-3 | Reinforcement Learning Based Dynamic Pruning | ⏳ Upcoming |
| Milestone-4 | Explainable AI & Comparative Evaluation | ⏳ Upcoming |

---

# System Architecture

```
                 Social IoT Graph
                        │
                        ▼
          PyTorch Geometric Dataset
                        │
                        ▼
              Graph Attention Network
                        │
                        ▼
          Attention-based Edge Importance
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 Threshold Pruning              Top-K Pruning
         │                             │
         └──────────────┬──────────────┘
                        ▼
            Botnet Spread Evaluation
                        │
                        ▼
      Reinforcement Learning (Milestone-3)
                        │
                        ▼
 Explainable AI & Final Evaluation (Milestone-4)
```

---

# Current Project Structure

```
gat-social-iot-swarup/

│
├── data/
│
├── results/
│   ├── models/
│   └── plots/
│
├── src/
│
│   ├── analysis/
│   │      attention_degree_analysis.py
│   │      attention_correlation_analysis.py
│   │
│   ├── graph/
│   │      social_iot_graph.py
│   │
│   ├── simulator/
│   │      botnet_simulator.py
│   │
│   ├── models/
│   │      social_iot_to_pyg.py
│   │      gat_social_iot.py
│   │      edge_importance_from_gat.py
│   │      threshold_pruning.py
│   │      topk_pruning.py
│   │      pruning_evaluation.py
│   │
│   └── utils/
│          data_loader.py
│          save_social_iot_dataset.py
│          pyg_to_networkx.py
│
├── README.md
└── requirements.txt
```

---

# Milestone-1 Summary

Successfully completed:

- Python environment setup
- PyTorch Geometric installation
- Synthetic Social IoT graph generation
- Node feature generation
- Edge feature generation
- Degree feature integration
- Graph statistics computation
- Degree distribution analysis
- Botnet propagation simulator
- Toy GAT verification

Milestone-1 establishes a reproducible Social IoT graph generation and simulation framework.

---

# Milestone-2 Summary

Milestone-2 focuses on learning graph attention and using it for secure graph pruning.

## Completed Components

### Dataset Preparation

- Social IoT dataset generation
- Frozen PyTorch Geometric dataset
- Dataset loading utilities

### Graph Attention Network

- Multi-head Graph Attention Network
- Node classification
- Best model checkpoint saving
- Reproducible training pipeline

### Attention Analysis

Implemented:

- Edge importance extraction
- Attention statistics
- Pearson correlation analysis
- Attention histogram
- Degree vs Attention scatter plot

### Static Graph Pruning

Implemented two pruning strategies:

#### Threshold-based Pruning

- Removes edges below a selected attention percentile.

#### Top-K Pruning

- Keeps only the strongest K outgoing edges for each node.

### Botnet Evaluation

Compared:

- Original Graph
- Threshold-Pruned Graph
- Top-K Pruned Graph

using botnet propagation simulations.

---

# Experimental Results

## GAT Training

- Multi-head GAT successfully trained.
- Best model automatically saved.

---

## Attention Analysis

Successfully generated:

- Edge importance scores
- Attention statistics
- Attention histogram
- Degree-attention correlation
- Scatter plot

---

## Static Pruning Results

| Method | Original | Remaining | Removed |
|---------|----------|-----------|----------|
| Threshold (20%) | 124 | 99 | 25 |
| Top-K (K=3) | 124 | 100 | 24 |

---

## Botnet Evaluation

Botnet propagation experiments demonstrate that:

- Threshold pruning slows malware propagation.
- Top-K pruning preserves important communication while reducing infection spread.
- Attention-guided pruning improves network robustness.

---

# How to Run

## 1. Activate Environment

```bash
venv\Scripts\activate
```

## 2. Train GAT

```bash
python -m src.models.gat_social_iot
```

## 3. Extract Edge Importance

```bash
python -m src.models.edge_importance_from_gat
```

## 4. Attention Correlation Analysis

```bash
python -m src.analysis.attention_correlation_analysis
```

## 5. Threshold Pruning

```bash
python -m src.models.threshold_pruning
```

## 6. Top-K Pruning

```bash
python -m src.models.topk_pruning
```

## 7. Botnet Evaluation

```bash
python -m src.models.pruning_evaluation
```

---

# Key Contributions

- Synthetic Social IoT graph generation
- Graph Attention based node classification
- Attention-guided edge importance estimation
- Statistical attention analysis
- Threshold-based graph pruning
- Top-K graph pruning
- Botnet propagation evaluation
- Modular and reusable research codebase

---

# Future Work

## Milestone-3

- Reinforcement Learning Environment
- Graph Pruning Agent
- Reward Function Design
- PPO/DQN Training
- Dynamic Graph Pruning

## Milestone-4

- Explainable AI
- SHAP Analysis
- Comparative Evaluation
- Research Paper Preparation

---

# Repository Status

✅ Milestone-1 Completed

✅ Milestone-2 Completed

⏳ Milestone-3 Under Development

⏳ Milestone-4 Planned

---

# Citation

If you find this repository useful in your research, please cite it appropriately after the publication of the associated research work.

---

# License

This project is intended for academic and research purposes.
