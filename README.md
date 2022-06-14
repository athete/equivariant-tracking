# Equivariant Charged Particle Tracking
[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/ameya1101/equivariant-tracking)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code written for the IRIS-HEP Fellowship project: **[Equivariant Graph Neural Networks for Particle Tracking](https://iris-hep.org/fellows/ameya1101.html)**.

---

## **Project Description**

Tracking devices, or trackers, at the LHC record hits of charged particles produced in a collision event. The task of track reconstruction involves clustering tracker hits to reconstruct the trajectories of these particles. The sparse nature of tracking data makes graph neural networks (GNNs) well-suited to particle tracking applications. The goal of this project is to develop a GNN for particle tracking by explicitly incorporating E(3)-equivariance into the model. Incorporating physically meaningful symmetries into the GNN can reduce the number of parameters and potentially reduce training and inference times for the model, while retaining the expressive power of non-equivariant GNNs.
