# High-Fidelity Simulation Platform for Microrobot Swarm Behaviors and Control

This repository contains the code associated with our manuscript:

**"High-Fidelity Simulation Platform for Scalable Learning of Microrobot Swarm Behaviors and Control"**

---

## Overview

Microrobot swarms exhibit complex collective behaviors that enable applications such as targeted delivery and micromanipulation. However, learning such behaviors in physical systems is challenging due to strong inter-particle interactions, environmental disturbances, and limited access to scalable training data.

This repository provides a high-fidelity simulation platform for learning microrobot swarm behaviors and control, along with demonstration code for training and deployment.

---

## Important Note

Due to the coupling with hardware systems (imaging system and electromagnetic actuation setup), the full physical experimental pipeline cannot be directly reproduced.

However, we provide:

- Simulation environments for reproducible experiments
- Reinforcement learning training scripts
- Pre-trained models and deployment examples

These components allow users to reproduce the learning process and evaluate trained policies in simulation.

---

## Requirements

- Python 3.8+
- Windows / Linux
- Unity3D

Install dependencies:
```bash
pip install -r requirements.txt
```
---

## Sections

1. Training_env
2. Deployment_demo
3. PID_demo

---

### 1. Training_env

This folder contains simulation environments for four representative tasks:

- delivery
- navigation
- slope_motion
- shape_control

It also includes corresponding training scripts for reproducible reinforcement learning experiments.

After running the training programs, log files and trained models will be automatically generated and saved in:

- `logs_graphs/` training curves and logs
- `logs_models/` trained policy models

---

### 2. Deployment_demo

This folder provides testing environments, pre-trained models, and example scripts for all four tasks.

After execution, the trained models can be directly loaded to perform the corresponding tasks. The system will automatically generate recorded output videos demonstrating the swarm behaviors.

---

### 3. PID_demo

This folder includes classical PID control experiments for 3D motion tasks.

These experiments serve as a baseline for comparison with reinforcement learning-based control methods.

---

## Simulation Environment

The simulation environment is developed in Unity3D and models:

- Inter-particle interactions
- Swarm-environment coupling
- Collective swarm dynamics


