\# GAT-based Secure Social IoT Framework



This project implements a Graph Attention Network (GAT) + Reinforcement Learning (RL) + Explainable AI (XAI)

framework to secure Social Internet of Things (Social IoT) networks against botnet propagation using

intelligent edge pruning. The goal is to dynamically identify and prune risky edges in the Social IoT

graph while preserving essential communication and service functionality.



Author: Swarup Sarkar  

Version: Milestone-1 Baseline  

Last Updated: January 2026  



--------------------------------------------------

Project Objectives

--------------------------------------------------

\- Construct a Social IoT graph with heterogeneous nodes and edges.

\- Simulate botnet propagation over the graph using a probabilistic infection model.

\- Learn edge importance using a Graph Attention Network (GAT).

\- Apply Reinforcement Learning (RL) to adaptively prune edges for security–utility trade-off.

\- Use Explainable AI (XAI) techniques to justify pruning decisions in a transparent way.



--------------------------------------------------

Project Structure

--------------------------------------------------

gat-social-iot-swarup/

│

├── data/                 # Datasets (ignored in Git for large files)

│

├── src/

│   ├── graph/            # Social IoT graph generator and analysis

│   ├── simulator/        # Botnet propagation simulator

│   ├── models/           # GAT and RL models (to be added in later milestones)

│   └── utils/            # Utility and helper functions

│

├── experiments/          # Experimental scripts and results

│

├── toy\_gat\_test.py       # Toy GAT verification using Cora dataset

├── README.md

└── .gitignore



--------------------------------------------------

How to Run the Milestone-1 Experiments

--------------------------------------------------



1\. Activate the virtual environment:

&nbsp;  

&nbsp;  venv\\Scripts\\activate



2\. Run the Social IoT Graph Generator:



&nbsp;  python src\\graph\\social\_iot\_graph.py



&nbsp;  This will:

&nbsp;  - Generate a synthetic Social IoT graph

&nbsp;  - Print node and edge statistics

&nbsp;  - Show the network topology plot

&nbsp;  - Show the degree distribution plot

&nbsp;  - Print basic graph statistics (average degree, max degree, density, etc.)



3\. Run the Botnet Propagation Simulator:



&nbsp;  python -m src.simulator.botnet\_simulator



&nbsp;  This will:

&nbsp;  - Initialize infected seed nodes

&nbsp;  - Simulate botnet propagation over time

&nbsp;  - Print the number of infected nodes at each timestep

&nbsp;  - Plot the infection curve (Time vs Infected Nodes)



4\. Verify GAT Installation with a Toy Example:



&nbsp;  python toy\_gat\_test.py



&nbsp;  This will:

&nbsp;  - Load the Cora citation network dataset

&nbsp;  - Train a Toy GAT model

&nbsp;  - Print training, validation, and test accuracy

&nbsp;  - Verify that PyTorch and PyTorch Geometric are working correctly



--------------------------------------------------

Milestone-1 Status

--------------------------------------------------

Milestone-1 (Weeks 1–2) is fully completed. It includes:



\- Version-controlled repository with clean project structure  

\- Python environment setup with PyTorch and PyTorch Geometric  

\- Social IoT graph generator with node and edge features  

\- Degree feature integration and degree distribution analysis  

\- Basic graph statistics computation  

\- Botnet propagation simulator with infection curve visualization  

\- Successful verification of GAT using a Toy GAT model  



