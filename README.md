# High-Fidelity Simulation Platform for Microrobot Swarm Behaviors and Control

This repository contains the code associated with our manuscript:

**"High-Fidelity Simulation Platform for Scalable Learning of Microrobot Swarm Behaviors and Control"**

---

## Overview

Microrobot swarms exhibit complex collective behaviors that enable applications such as targeted delivery and micromanipulation. However, learning such behaviors in physical systems is challenging due to limited data availability and experimental constraints.

This repository provides a high-fidelity simulation platform for learning microrobot swarm behaviors and control, together with demonstration code for training and deployment.

---

## Important Note

Due to the coupling with hardware systems (imaging system and electromagnetic actuation setup), the full physical experimental pipeline cannot be directly reproduced.

However, we provide:

- Simulation environments for reproducible experiments
- Reinforcement learning training scripts
- Demonstration code for deployment policies

These components allow users to reproduce the learning process and evaluate trained policies in simulation.

---

## Requirements

- Python 3.8+
- Windows / Linux
- Unity3D (for simulation environment)

Install dependencies:

```bash
pip install -r requirements.txt

## Sections

1. Training_env
2. Deployment_demo
3. PID_demo

### 1. Training_env
This folder contains environment files for four types of tasks: delivery, navigation, slope_motion, and shape_control, along with corresponding reproducible training example scripts. After running the program, training log files and model files will be generated and saved in the automatically created logs_graghs and logs_models folders respectively.

### 2. Deployment_demo
This folder provides test environments for the four tasks, pre-trained task models, and test example code. After execution, the trained models can be invoked to complete corresponding tasks, and a recorded output video will be generated automatically.

### 3. PID_demo
This folder includes PID control tests for 3D motion tasks, which serve as the baseline for comparison with reinforcement learning methods.

