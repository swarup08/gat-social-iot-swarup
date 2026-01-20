# GAT-based Secure Social IoT Framework

**Author:** Swarup Sarkar  
**Version:** Milestone-1 Baseline  
**Last Updated:** January 2026  

---

## Overview

This project implements a **Graph Attention Network (GAT) + Reinforcement Learning (RL) + Explainable AI (XAI)**  
framework to secure **Social Internet of Things (Social IoT)** networks against botnet propagation using
intelligent edge pruning.

The main goal is to dynamically identify and prune risky edges in the Social IoT graph while preserving
essential communication and service functionality.

---

## Project Objectives

- Construct a Social IoT graph with heterogeneous nodes and edges  
- Simulate botnet propagation using a probabilistic infection model  
- Learn edge importance using a Graph Attention Network (GAT)  
- Apply Reinforcement Learning (RL) for adaptive edge pruning  
- Use Explainable AI (XAI) to provide transparent security justifications  

---

## Project Structure

```
gat-social-iot-swarup/
│
├── data/                 # Datasets (ignored in Git for large files)
│
├── src/
│   ├── graph/            # Social IoT graph generator and analysis
│   ├── simulator/        # Botnet propagation simulator
│   ├── models/           # GAT and RL models (future milestones)
│   └── utils/            # Utility and helper functions
│
├── experiments/          # Experimental scripts and results
│
├── toy_gat_test.py       # Toy GAT verification using Cora dataset
├── README.md
└── .gitignore
```

---

## How to Run Milestone-1 Experiments

### 1. Activate the virtual environment
```
venv\Scripts\activate
```

---

### 2. Run the Social IoT Graph Generator
```
python src\graph\social_iot_graph.py
```

This will:
- Generate a synthetic Social IoT graph  
- Print node and edge statistics  
- Display the network topology  
- Display the degree distribution plot  
- Print basic graph statistics (average degree, density, etc.)

---

### 3. Run the Botnet Propagation Simulator
```
python -m src.simulator.botnet_simulator
```

This will:
- Initialize infected seed nodes  
- Simulate botnet propagation over time  
- Print infected node count at each timestep  
- Plot the infection curve (Time vs Infected Nodes)

---

### 4. Verify GAT Installation (Toy Example)
```
python toy_gat_test.py
```

This will:
- Load the Cora citation dataset  
- Train a Toy GAT model  
- Print training, validation, and test accuracy  
- Verify PyTorch and PyTorch Geometric installation  

---

## Milestone-1 Status

Milestone-1 (Weeks 1–2) is fully completed. It includes:

- Python environment setup with PyTorch and PyTorch Geometric  
- Social IoT graph generator with node and edge features  
- Degree feature integration and degree distribution analysis  
- Basic graph statistics computation  
- Botnet propagation simulator with infection curve visualization  
- Successful verification of GAT using a Toy GAT model  

---

This milestone establishes a complete and reproducible baseline system for developing
GAT-based, RL-driven, and explainable security mechanisms in Social IoT networks.
