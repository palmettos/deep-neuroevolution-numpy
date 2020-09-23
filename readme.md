# Introduction
This is a very minimal implementation of [deep neuroevolution](https://arxiv.org/abs/1712.06567?source=post_page---------------------------) utilizing numpy and DEAP. An XOR example is provided to verify correctness.

# Requirements
- python 3.x
- numpy
- DEAP

# Notes
- In the XOR example, truncation selection is used. However, the generation's best are selected from a [double tournament](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.72.9008&rep=rep1&type=pdf) with a tunable parsimony pressure parameter that helps mitigate excessive growth of genomes
- More mutation operators are experimentally implemented: add, drop and mutate seed value
- The seed values are just unsigned 64-bit integers. The original paper describes a method to control the size of the seed values, but I have not implemented this yet
- Only FC feedforward layers are currently supported, but I'm planning on adding more types of layers