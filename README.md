# Equivariant Charged Particle Tracking
[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/ameya1101/equivariant-tracking)

[![PyPI version](https://img.shields.io/badge/python-3.9-blue)](https://img.shields.io/badge/python-3.9-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code written for the IRIS-HEP Fellowship project: **[Equivariant Graph Neural Networks for Particle Tracking](https://iris-hep.org/fellows/ameya1101.html)**.

**IRIS-HEP Fellowship Presentation:** 
[Equivariant Graph Neural Networks for Charged Particle Tracking](https://indico.cern.ch/event/1199559/), Ameya Thete, 19 October 2022. ([Recording]())

**Poster:** 
[Equivariant Graph Neural Networks for Charged Particle Tracking](https://indi.to/Gh2Fs), 21st International Workshop on Advanced Computing Analysis Techniques in Physics Research (ACAT), Bari, Italy. 26 October 2022.

---
## **Project Description**

Tracking devices, or trackers, at the LHC record hits of charged particles produced in a collision event. The task of track reconstruction involves clustering tracker hits to reconstruct the trajectories of these particles. The sparse nature of tracking data makes graph neural networks (GNNs) well-suited to particle tracking applications. The goal of this project is to develop a GNN for particle tracking by explicitly incorporating SO(2)-equivariance into the model. Incorporating physically meaningful symmetries into the GNN can reduce the number of parameters and potentially reduce training and inference times for the model, while retaining the expressive power of non-equivariant GNNs.


## **Acknowledgements**

This work was supported by IRIS-HEP through the U.S. National Science Foundation (NSF) under Cooperative Agreement OAC-1836650 and by Department of Energy grant DE-SC0007968. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility located at Lawrence Berkeley National Laboratory, operated under Contract No. DE-AC02-05CH11231 using 2022 NERSC award ERCAP-0021226.

---
## **Authors**

* Ameya Thete
* Daniel Murnane