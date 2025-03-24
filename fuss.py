"""
FUSS - Fitness Uniform Selection proposed by Marcus Hutter (who knew he was interested in genetic algorithms?)

http://www.hutter1.net/ai/pfuss.htm

He replaces classic tournament selection with selecting for individuals on a uniform fitness distribution, using fitness as a proxy for diversity. He found that this maintains population diversity and improves results for certain problems (however not all problems, and also has the flaw of not increasing diversity of the fitter individuals)
"""

import torch
from einx import get_at

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
MUTATION_RATE = 0.05
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])

# derived constants

gene_length = len(GOAL)
gene_midpoint = gene_length // 2
target_gene = encode(GOAL)

keep_fittest_len = int(POP_SIZE * FRAC_FITTEST_SURVIVE)
num_children = POP_SIZE - keep_fittest_len
num_mutate = MUTATION_RATE * gene_length

# genetic algorithm

generation = 1

pool = torch.randint(0, 255, (POP_SIZE, gene_length))

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    fitnesses = 1. / torch.square(pool - target_gene).sum(dim = -1)

    indices = fitnesses.sort(descending = True).indices
    pool, fitnesses = pool[indices], fitnesses[indices]

    # keep the fittest

    pool, fitnesses = pool[:keep_fittest_len], fitnesses[:keep_fittest_len]

    # display every generation

    for gene, fitness in zip(pool, fitnesses):
        print(f"{decode(gene)} ({fitness.item():.3f})")

    # solved if any fitness is inf

    if (fitnesses == float('inf')).any():
        break

    # FUSS - fitness uniform selection

    sorted_fitness, sorted_gene_indices = fitnesses.sort(dim = -1)

    sorted_fitness = sorted_fitness - sorted_fitness[0]
    sorted_fitness_cdf = sorted_fitness.cumsum(dim = -1)
    sorted_fitness_cdf = sorted_fitness_cdf / sorted_fitness_cdf[-1]

    rand = torch.rand((2, num_children))
    rand_parent_sorted_gene_ids = torch.searchsorted(sorted_fitness_cdf, rand)

    parent_ids = sorted_gene_indices[rand_parent_sorted_gene_ids - 1]

    parents = get_at('[p] d, ... -> ... d', pool, parent_ids)

    # cross over recombination of parents

    parent1, parent2 = parents
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)

    pool = torch.cat((pool, children))

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    pool = torch.where(mutate_mask, pool + noise, pool)
    pool.clamp_(0, 255)

    generation += 1
