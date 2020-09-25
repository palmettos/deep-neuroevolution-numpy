from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
from encoding import encoding
import math

pop_size = 500
elitism = 5
survival_threshold = 0.4
mutate_options = [
    encoding.mutate_add_seed,
    encoding.mutate_drop_seed,
    encoding.mutate_existing_seed
]

input_data = [
    {
        'input': (1., 1.),
        'expected': 0.
    },
    {
        'input': (1., 0.),
        'expected': 1.
    },
    {
        'input': (0., 1.),
        'expected': 1.
    },
    {
        'input': (0., 0.),
        'expected': 0.
    }
]


def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def eval(genome, layer_defs):
    nn = encoding.initialize(genome, layer_defs)
    fitness = 4.
    for sample in input_data:
        output = encoding.forward_pass(nn, np.array(sample['input']))
        fitness -= (sample['expected'] - output) ** 2
    return fitness


layer_defs = [
    {
        'name': 'input',
        'nodes': 2
    },
    {
        'name': 'encoder_hidden',
        'nodes': 2,
        'activation': relu
    },
    {
        'name': 'output',
        'nodes': 1,
        'activation': relu
    }
]

creator.create('fitness', base.Fitness, weights=(1.,))
creator.create('individual', list, fitness=creator.fitness)

toolbox = base.Toolbox()
rng = np.random.default_rng()

toolbox.register(
    'individual',
    tools.initRepeat,
    creator.individual,
    lambda: rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64),
    1
)

toolbox.register('population', tools.initRepeat, list, toolbox.individual, pop_size)
toolbox.register('evaluate', eval, layer_defs=layer_defs)
toolbox.register('mutate', encoding.mutate_genome, mutate_options=mutate_options)

pop = toolbox.population()
best_fitness = 0.
best_genome = None

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fitness in zip(pop, fitnesses):
    ind.fitness.values = [fitness]

gen = 0
while best_fitness < 3.95:
    gen += 1

    truncated = tools.selDoubleTournament(
        individuals=pop,
        k=math.floor(len(pop) * survival_threshold),
        fitness_size=5,
        parsimony_size=1.5,
        fitness_first=True
    )

    next_pop = truncated[:elitism]
    while len(next_pop) < pop_size:
        parent = toolbox.clone(random.choice(truncated))
        next_pop.append(toolbox.mutate(parent))

    pop = next_pop

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fitness in zip(pop, fitnesses):
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = toolbox.clone(ind)
        ind.fitness.values = [fitness]

    genome_length = np.array(list(map(lambda genome: len(genome), pop)))
    avg_length = np.mean(genome_length)

    print(f'generation: {gen}, best_fitness: {best_fitness}')
    print(f'best_genome: {best_genome}')
    print(f'avg_length: {avg_length}')
    print('-'*80)

nn = encoding.initialize(best_genome, layer_defs)
for input in input_data:
    print(f'input: {input["input"]}')
    output = encoding.forward_pass(nn, np.array(input['input']))
    print(f'output: {output}')