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
