import numpy as np
from pprint import PrettyPrinter


def mutate_existing_seed(genome, index, seed):
    genome[index] = seed
    return genome


def mutate_add_seed(genome, index, seed):
    genome.append(seed)
    return genome


def mutate_drop_seed(genome, index, seed):
    if len(genome) < 2:
        return mutate_existing_seed(genome, 0, seed)
    del genome[index]
    return genome


def mutate_genome(genome, mutate_options):
    rng = np.random.default_rng()
    mutate_func = rng.choice(mutate_options)
    index = rng.integers(len(genome))
    seed = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    return mutate_func(genome, index, seed)


def apply_phenotype_mutation(seed, phenotype):
    rng = np.random.default_rng(seed)
    for layer in phenotype:
        weight_noise = rng.normal(0., 0.1, layer['weights'].shape)
        layer['weights'] += weight_noise
        bias_noise = rng.normal(0., 0.1, layer['biases'].shape)
        layer['biases'] += bias_noise
    return phenotype


def initialize(genome, layer_defs):
    rng = np.random.default_rng(genome[0])

    n_layers = len(layer_defs)
    phenotype = []
    for prev, curr in zip(range(0, n_layers-1), range(1, n_layers)):
        prev_layer_node_count = layer_defs[prev]['nodes']
        curr_layer_node_count = layer_defs[curr]['nodes']
        layer = {
            'name': layer_defs[curr]['name'],
            'weights': rng.normal(0., 1/prev_layer_node_count, (prev_layer_node_count, curr_layer_node_count)),
            'biases': np.zeros(curr_layer_node_count),
            'activation': layer_defs[curr]['activation']
        }
        phenotype.append(layer)

    if len(genome) > 1:
        for seed in genome[1:]:
            phenotype = apply_phenotype_mutation(seed, phenotype)
    
    return phenotype


def forward_pass(phenotype, inputs):
    layer_out = inputs
    for layer in phenotype:
        z = layer_out.dot(layer['weights']) + layer['biases']
        layer_out = layer['activation'](z)
    return layer_out
