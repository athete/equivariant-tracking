# Worklog

A log of the tasks completed over the week, updated weekly.

## Week 1 (06 Jun - 12 Jun)
- Studied GNNs
    * [Microsoft Research Talk (Allamanis)](https://youtu.be/zCEYiCxrL_0)
    * [Theoretical Foundations of Graph Neural Networks (Veličković)](https://youtu.be/uF53xsT7mjc)
    * [GNNs Beyond Permutation Equivariance (Veličković)](https://youtu.be/aCUOAkOqNoU)
- Looked at particle tracking GNNs
    * [Graph Neural Networks for High Luminosity Track Reconstruction (Murnane)](https://cds.cern.ch/record/2809959)
    * [Graph Neural Networks for Particle Reconstruction in High Energy Physics Detectors](https://arxiv.org/abs/2003.11603)
    * [Graph Neural Networks for Particle Tracking](https://drive.google.com/file/d/11NDxKukSEMRctrWFu3DV-UJHUJtrnsXs/view?usp=sharing)
- Read [DeZoort et al](https://arxiv.org/abs/2103.16701)

## Week 2 (13 Jun - 19 Jun)
- Implemented the Interaction Network pipeline to familiarize myself with GNNs 
    * Graph Construction
    * GNN Training
- Setup NESRC account
    * Ran a test "Hello, World!" job

## Week 3 (20 Jun - 26 Jun)
- Adapted the LorentzNet architecture to a tracking task (used the Euclidean group instead) -- called it EuclidNet
- Generated hitgraphs using the geometric and pre-clustering methods
- Wrote training scripts (need to perform a sanity check though)

## Week 4 (27 Jun - 3 Jul)
- Tested training scripts on GPU and CPU
- Turns out EuclidNet wasn't SO(3) equivariant, changed that :( 
- Generated geometric hitgraphs for $p_T = 0.8, 1.3, 1.5, 2.0$ GeV

## Week 5 (4 Jul - 10 Jul)
- Trained the _corrected_ EuclidNet on geometric and pre-clustering hitgraphs for $p_T = 1.5$ GeV
    * This architecture is extremely simple: uses only inner products and norms constructed from node features
- Ran a quick equivariance test on EuclidNet blocks
    * Deviation from equivariance $\sim O(10^{-6} - 10^{-5})$ 
    * The deviation is periodic for some reason with a period of $\pi/2$. This needs to be investigated.
- tl;dr geometric better than pre-clustering. ~95-96% accuracy
    * Accuracy is a false metric for this task, need to look at in the context of purity (will do that next week)/
- Wrote a script that computes metrics for the last stage in the pipeline, needs testing however.

# Week 6 (11 Jul - 18 Jul)
- Trained EuclidNet on the remaining $p_T$ hitgraphs. 
- Found a library called [EMLP](https://github.com/mfinzi/equivariant-MLP) to generate equivariant MLP layers
    * This would let us use vector features to generate messages :)
    * Modifying EuclidNet to use EMLPs to generate message vectors. 