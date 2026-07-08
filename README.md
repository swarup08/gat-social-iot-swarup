# GAT-based Secure Social IoT Framework using Graph Attention Networks, Deep Reinforcement Learning and Explainable AI

![Research](https://img.shields.io/badge/Research-Q1%20Journal%20Project-blue)
![Status](https://img.shields.io/badge/Status-Milestone--4%20Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Latest-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

# Overview

This repository presents a complete research framework for securing the **Social Internet of Things (SIoT)** against botnet propagation using **Graph Attention Networks (GAT)**, **Deep Reinforcement Learning (DQN)**, and **Explainable Artificial Intelligence (XAI)**.

The proposed framework models SIoT devices as a graph where nodes represent smart entities and edges represent social relationships or communication links. A Graph Attention Network learns the importance of communication links through attention mechanisms. These learned attention scores guide an intelligent Deep Reinforcement Learning agent that dynamically prunes high-risk communication links while preserving overall network connectivity.

Unlike traditional static pruning approaches, the proposed framework continuously adapts its pruning strategy according to the current network condition, making it suitable for defending against dynamic botnet propagation in large-scale Social IoT environments.

To improve transparency, an Explainable AI module summarizes pruning decisions by identifying influential communication links and important nodes responsible for security decisions.

---

# Research Motivation

The rapid growth of the Social Internet of Things has introduced highly interconnected smart environments where malware and botnets can propagate rapidly through social relationships among devices.

Conventional graph pruning techniques rely on fixed thresholds or heuristic rules, which often fail to adapt to evolving attack patterns.

This research investigates whether Graph Attention Networks can identify critical communication links and whether Deep Reinforcement Learning can dynamically optimize graph pruning decisions while maintaining network connectivity.

The ultimate objective is to build an intelligent, adaptive, and explainable defense mechanism for securing Social IoT networks against botnet propagation.

---

# Key Contributions

This repository provides the following research contributions:

- Graph Attention Network for Social IoT node representation learning
- Attention-based communication link importance estimation
- Threshold-based static graph pruning
- Deep Reinforcement Learning (DQN) for adaptive edge pruning
- Explainable AI module for edge-level and node-level pruning explanations
- Botnet propagation simulator for security evaluation
- Comparative evaluation against baseline pruning strategies
- Comprehensive ablation studies
- Robustness analysis under different attack scenarios
- Scalability analysis using different network sizes and graph densities
- Quality figures and reproducible experiments

---

# Research Workflow

```

Social IoT Graph Construction

↓

PyTorch Geometric Dataset

↓

Graph Attention Network (GAT)

↓

Attention-based Edge Importance

↓

Threshold Pruning (Baseline)

↓

Deep Reinforcement Learning (DQN)

↓

Dynamic Edge Pruning

↓

Botnet Propagation Simulation

↓

Explainable AI (XAI)

↓

Experimental Evaluation

↓

Results

```

---

# Project Architecture

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
        ┌───────────────┴────────────────┐
        │                                │
        ▼                                ▼
Threshold Pruning              Dynamic RL Pruning
        │                                │
        └───────────────┬────────────────┘
                        ▼
           Botnet Propagation Evaluation
                        │
                        ▼
           Explainable Artificial Intelligence
                        │
                        ▼
        Figures • Tables • Experimental Results

```

---

# Repository Structure

```

gat-social-iot-swarup/

│

├── data/
│   └── Synthetic Social IoT datasets

│

├── experiments/
│   ├── compare_methods.py
│   ├── robustness_test.py
│   ├── reward_weight_ablation.py
│   ├── noisy_feature_ablation.py
│   ├── network_size_stress_test.py
│   ├── graph_density_stress_test.py
│   ├── ablation_gat_heads.py
│   ├── ablation_gat_layers.py
│   └── generate_final_figures.py

│

├── results/
│   ├── figures/
│   ├── history/
│   ├── logs/
│   ├── models/
│   ├── plots/
│   └── tables/

│

├── src/
│   ├── analysis/
│   ├── graph/
│   ├── models/
│   ├── rl/
│   ├── simulator/
│   ├── utils/
│   └── xai/

│

├── tests/

│

├── README.md

└── requirements.txt

```

---

# Current Project Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| Milestone-1 | Social IoT Graph Construction | ✅ Completed |
| Milestone-2 | Graph Attention Learning & Static Pruning | ✅ Completed |
| Milestone-3 | Deep Reinforcement Learning for Dynamic Pruning | ✅ Completed |
| Milestone-4 | Explainable AI, Ablation Studies and Experimental Evaluation | ✅ Completed |

---

# Author

**Swarup Sarkar**

Research Interests:

- Graph Neural Networks (GNN)
- Graph Attention Networks (GAT)
- Social Internet of Things (SIoT)
- Reinforcement Learning
- Explainable Artificial Intelligence (XAI)
- Cyber Security
- Trust Management

---

# Installation

## Prerequisites

The project has been developed and tested using the following software versions.

| Software | Version |
|----------|---------|
| Python | 3.11 |
| PyTorch | 2.x |
| PyTorch Geometric | Latest |
| NetworkX | Latest |
| NumPy | Latest |
| Pandas | Latest |
| Matplotlib | Latest |
| Scikit-learn | Latest |

---

## Clone the Repository

```bash
git clone https://github.com/<your-github-username>/gat-social-iot-swarup.git

cd gat-social-iot-swarup
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The framework uses a synthetic Social Internet of Things (SIoT) graph generated using NetworkX and converted into a PyTorch Geometric dataset.

Each node represents a smart IoT entity, while edges represent communication or social relationships between devices.

### Node Features

- Device Type
- Degree
- Trust Score
- Activity Score
- Battery Level
- Vulnerability Label

### Edge Features

- Communication Protocol
- Bandwidth
- Latency
- Learned GAT Attention Score

The dataset is automatically generated and loaded using:

```text
src/utils/data_loader.py
```

---

# Project Workflow

The overall execution pipeline is shown below.

```
Generate Social IoT Graph

↓

Convert to PyTorch Geometric Dataset

↓

Train Graph Attention Network

↓

Extract Attention Scores

↓

Threshold-based Pruning

↓

Dynamic Reinforcement Learning Pruning

↓

Botnet Propagation Simulation

↓

Explainability Analysis

↓

Generate Figures and Tables

↓

Research Report
```

---

# Running the Project

## Step 1

Train the Graph Attention Network

```bash
python -m src.models.gat_social_iot
```

Output

```
results/models/best_gat_social_iot.pth
```

---

## Step 2

Extract Edge Importance

```bash
python -m src.models.edge_importance_from_gat
```

Output

```
Attention Scores
```

---

## Step 3

Run Threshold-based Pruning

```bash
python -m src.models.threshold_pruning
```

---

## Step 4

Train the Deep Reinforcement Learning Agent

```bash
python -m src.rl.train_dqn
```

Output

```
results/models/dqn_dynamic_pruning.pth
```

---

## Step 5

Evaluate Dynamic RL

```bash
python -m experiments.evaluate_dynamic_rl
```

---

## Step 6

Compare All Methods (single-graph sanity check)

```bash
python -m experiments.compare_methods
```

Output

```
results/tables/comparison_results.csv
```

This runs every method once on a single generated graph. It is a fast
smoke test for checking the pipeline end-to-end, **not** the reported
result — a single graph's outcome is not statistically meaningful. See
Step 6a/6b below for the authoritative multi-graph comparison.

---

## Step 6a

Multi-Graph Evaluation

Repeats the full method comparison across many independently generated
graph instances and reports mean +/- std per method, so results aren't
driven by one lucky/unlucky graph.

```bash
python -m experiments.multi_graph_evaluation
```

Output

```
results/tables/multi_graph_raw_results.csv
results/tables/multi_graph_summary.csv
```

---

## Step 6b

Paired Comparison vs Random

Graph-to-graph topology variance dominates the between-method
differences in the pooled summary above, so this re-analyzes the same
raw results as a per-graph paired comparison against Random pruning
(paired t-test, Wilcoxon signed-rank, Shapiro-Wilk normality check).

```bash
python -m experiments.paired_comparison
```

Output

```
results/tables/paired_comparison_vs_random.csv
```

**`multi_graph_summary.csv` and `paired_comparison_vs_random.csv` are
the authoritative comparison results** (n=40 graphs); treat
`comparison_results.csv` as an illustrative single-run example only.

---

## Step 7

Run Robustness Analysis

```bash
python -m experiments.robustness_test
```

---

## Step 8

Run GAT Head Ablation

```bash
python -m experiments.ablation_gat_heads
```

Output

```
results/tables/gat_heads_ablation.csv
```

---

## Step 9

Run GAT Layer Ablation

```bash
python -m experiments.ablation_gat_layers
```

---

## Step 10

Run Reward Weight Ablation

```bash
python -m experiments.reward_weight_ablation
```

---

## Step 11

Run Noisy Feature Ablation

```bash
python -m experiments.noisy_feature_ablation
```

---

## Step 12

Run Network Size Stress Test

```bash
python -m experiments.network_size_stress_test
```

---

## Step 13

Run Graph Density Stress Test

```bash
python -m experiments.graph_density_stress_test
```

---

## Step 14

Generate XAI Summaries

```bash
python -m src.xai.edge_summary

python -m src.xai.node_summary
```

---

## Step 15

Generate Figures

```bash
python -m experiments.generate_final_figures
```

Outputs

```
results/figures/
```

- infection_curve_comparison.png
- performance_comparison.png
- xai_top_edges.png

---

# Reproducibility

All experiments in this repository are fully reproducible.

The generated CSV files, trained models, plots, and figures are automatically stored inside the `results` directory.

Each experiment can be executed independently without modifying the source code.

Random seeds are fixed throughout the implementation to improve reproducibility.

---

# Experimental Results

The proposed framework was extensively evaluated using multiple baseline methods, ablation studies, robustness analyses, and stress tests to assess its effectiveness in mitigating botnet propagation within Social IoT networks.

---

# Method Comparison

Three edge pruning strategies were evaluated.

| Method | Description |
|----------|-------------|
| No Pruning | Original Social IoT graph without any edge removal |
| Threshold Pruning | Static attention-based edge pruning using a fixed threshold |
| Dynamic RL Pruning | Adaptive edge pruning using a Deep Q-Network (DQN) agent |

The Dynamic RL approach achieved a better balance between malware mitigation and network connectivity than static threshold pruning.

---

# Performance Summary

The framework evaluates each pruning strategy using the following metrics.

- Infection Ratio
- Connectivity Ratio
- Number of Removed Edges
- Total Reward (RL)

Experimental results are automatically exported to:

```
results/tables/multi_graph_summary.csv           (authoritative: mean +/- std over 40 graphs)
results/tables/paired_comparison_vs_random.csv   (authoritative: paired significance tests vs Random)
results/tables/comparison_results.csv            (single-graph sanity check only)
```

---

# Graph Attention Network (GAT) Ablation Study

To evaluate the robustness of the Graph Attention Network, multiple architectural configurations were investigated.

### Multi-Head Attention

Different attention head configurations were evaluated.

- 2 Heads
- 4 Heads
- 8 Heads

Experimental results are stored in:

```
results/tables/gat_heads_ablation.csv
```

---

### Layer Ablation

Different GAT architectures with varying numbers of graph attention layers were evaluated to study their influence on node classification performance.

The objective was to determine whether deeper architectures improve representation learning within Social IoT graphs.

---

# Reinforcement Learning Reward Ablation

Different reward configurations were investigated to analyze how the RL agent balances security and network utility.

The following reward strategies were evaluated.

- Security-Focused
- Balanced
- Utility-Focused

Results are stored in

```
results/tables/reward_weight_ablation.csv
```

---

# Noisy Feature Ablation

The robustness of the Graph Attention Network was further evaluated by injecting synthetic noise into node features.

Noise Levels:

- 0%
- 5%
- 10%

The experiment demonstrates the sensitivity of GAT performance to imperfect node representations.

Results:

```
results/tables/noisy_feature_ablation.csv
```

---

# Stress Testing

To evaluate scalability and robustness, multiple stress tests were conducted.

### Network Size Analysis

Different graph sizes were investigated.

- 40 Nodes
- 80 Nodes
- 120 Nodes

Metrics observed:

- Number of Edges
- Final Infection
- Infection Ratio

Results:

```
results/tables/network_size_stress_test.csv
```

---

### Graph Density Analysis

The communication density of the Social IoT graph was varied.

Graph densities evaluated:

- 0.05
- 0.08
- 0.12

Observed metrics:

- Number of Edges
- Infection Ratio
- Final Infection

Results:

```
results/tables/graph_density_stress_test.csv
```

---

# Explainable AI (XAI)

To improve transparency, the framework includes an Explainable AI module that summarizes graph pruning decisions.

Two complementary analyses are provided.

## Edge-Level Explanation

Frequently pruned communication links are identified using learned GAT attention scores.

Output:

```
results/tables/edge_summary.csv
```

---

## Node-Level Explanation

Each node is analyzed to determine

- Frequently pruned neighbors
- Average attention score
- Average reward
- Number of pruned edges

Output:

```
results/tables/node_summary.csv
```

---

# Generated Figures

### Figure 1

**Botnet Infection under Different Edge Pruning Strategies**

Illustrates infection behavior for

- No Pruning
- Static Threshold Pruning
- Dynamic RL Pruning

---

### Figure 2

**Performance Comparison of Edge Pruning Strategies**

Compares

- Infection Ratio
- Connectivity Ratio
- Removed Edges

---

### Figure 3

**Top-10 Most Influential Edges Selected for Pruning by GAT Attention**

Visualizes the communication links receiving the highest attention scores and selected for pruning.

All generated figures are stored in

```
results/figures/
```

---

# Experimental Highlights

The proposed framework demonstrates several important observations.

- Dynamic RL pruning reduces infection while preserving higher network connectivity than static pruning.
- Attention-guided edge selection effectively identifies high-risk communication links.
- The GAT architecture remains stable across different attention-head configurations.
- The framework remains robust under noisy node features.
- Scalability experiments demonstrate consistent behavior across different network sizes and graph densities.
- Explainable AI modules improve transparency by identifying influential edges and node-level pruning patterns.

---
# Results Directory

All generated outputs are automatically organized inside the `results` directory.

```
results/

├── figures/
│   ├── infection_curve_comparison.png
│   ├── performance_comparison.png
│   └── xai_top_edges.png
│
├── history/
│   └── Training history and experiment logs
│
├── logs/
│   └── pruning_log.csv
│
├── models/
│   ├── best_gat_social_iot.pth
│   ├── dqn_dynamic_pruning.pth
│   ├── gat_heads_2.pth
│   ├── gat_heads_4.pth
│   └── gat_heads_8.pth
│
├── plots/
│   └── Intermediate visualization outputs
│
└── tables/
    ├── comparison_results.csv            (single-graph sanity check)
    ├── multi_graph_raw_results.csv       (authoritative: per-graph raw results, n=40)
    ├── multi_graph_summary.csv           (authoritative: mean +/- std, n=40)
    ├── paired_comparison_vs_random.csv   (authoritative: paired significance tests)
    ├── gat_heads_ablation.csv
    ├── reward_weight_ablation.csv
    ├── noisy_feature_ablation.csv
    ├── network_size_stress_test.csv
    ├── graph_density_stress_test.csv
    ├── edge_summary.csv
    └── node_summary.csv
```

---

# Reproducibility

The repository has been designed to ensure reproducible experimentation.

Key reproducibility measures include:

- Fixed random seeds
- Modular implementation
- Automatic result generation
- CSV-based experiment logging
- Automatic model checkpoint saving
- Figure generation
- Independent experiment scripts

All major experiments can be reproduced using the provided commands without modifying the source code.

---

# Project Highlights

The proposed framework integrates multiple AI techniques into a unified Social IoT security pipeline.

Key components include:

- Graph Neural Networks
- Graph Attention Networks
- Deep Reinforcement Learning
- Explainable Artificial Intelligence
- Graph Pruning
- Botnet Propagation Simulation
- Experimental Evaluation
- Ablation Studies
- Stress Testing

The project demonstrates how adaptive graph pruning can effectively reduce malware propagation while preserving network connectivity.

---

# Current Limitations

Although the proposed framework demonstrates promising results, several limitations remain.

- The experiments are conducted on a synthetic Social IoT dataset.
- The current reinforcement learning agent is based on Deep Q-Network (DQN).
- Dynamic botnet behaviors and adaptive adversaries are not explicitly modeled.
- Real-world Social IoT datasets have not yet been evaluated.
- Multi-agent reinforcement learning has not been explored.

These limitations provide opportunities for future research.

---

# Future Research Directions

Possible extensions of this work include:

- Graph Transformer Networks
- Multi-Agent Reinforcement Learning
- Trust-aware Graph Neural Networks
- Federated Learning for SIoT
- Continual Learning
- Zero-Day Attack Detection
- Explainable Reinforcement Learning
- Real-world Social IoT datasets
- Temporal Graph Neural Networks
- Large Language Models for autonomous security decision support

---

# Repository Status

| Component | Status |
|-----------|--------|
| Social IoT Graph Generation | ✅ Completed |
| GAT Training | ✅ Completed |
| Attention Analysis | ✅ Completed |
| Static Graph Pruning | ✅ Completed |
| Dynamic RL Pruning | ✅ Completed |
| Explainable AI | ✅ Completed |
| Comparative Evaluation | ✅ Completed |
| Ablation Studies | ✅ Completed |
| Stress Testing | ✅ Completed |
| Figures | ✅ Completed |
| Documentation | ✅ Completed |

---

# Acknowledgements

This project was developed as part of ongoing doctoral research in the field of:

- Graph Neural Networks
- Social Internet of Things
- Reinforcement Learning
- Explainable Artificial Intelligence
- Cyber Security

The implementation emphasizes reproducible research practices, modular software design, quality experimentation.

---

# License

This repository is released for academic and research purposes.

You are welcome to use, modify, and extend this work with appropriate citation.

---

# Contact

**Swarup Sarkar**

Assistant Professor  
Department of Computer Science and Engineering

Research Interests

- Graph Neural Networks
- Social Internet of Things
- Reinforcement Learning
- Explainable Artificial Intelligence
- Cyber Security

---

# Final Remarks

This repository presents an end-to-end research framework for securing Social Internet of Things networks through the integration of Graph Attention Networks, Deep Reinforcement Learning, and Explainable Artificial Intelligence.

The implementation includes reproducible experiments, extensive ablation studies, robustness evaluations, stress testing, and quality visualizations, making it suitable for academic research, benchmarking, and future extensions in intelligent graph security.

**Thank you for visiting this repository.**