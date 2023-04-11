# GraphAC

Here we provide the source code for the paper "Graph Attention for Automated Audio Captioning" submitted to [IEEE Signal Processing Letter]()

## Model Structure

The model structure of the proposed GraphAC is provided in the file path [model](./model/), and the dependent module parts (i.e., PANNs, GAT and the SpecAugment operation) are provided in the file path [modules](./modules/).

## Examples

The example of predicted captions is presented at [captions_of_examples.md](captions_of_examples.md), accompanying Figure 2 of the paper "Graph Attention for Automated Audio Captioning". It demonstrates that the proposed method can accurately capture and caption the long-time dependent information.
