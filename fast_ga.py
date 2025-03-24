"""
Fast Genetic Algorithm - https://arxiv.org/abs/1703.03334

faster convergence by drawing mutation length from power law distribution
"""

import torch
from einx import get_at, less

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25
POWER_LAW_BETA = 1.1

assert POWER_LAW_BETA > 1.

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
num_tournament_contenders = int(keep_fittest_len * FRAC_TOURNAMENT)
num_children = POP_SIZE - keep_fittest_len

# power law cdf

half_gene_length = gene_length // 2
power_law_cdf = torch.linspace(1, half_gene_length, half_gene_length).pow(-POWER_LAW_BETA).cumsum(dim = -1)
power_law_cdf = power_law_cdf / power_law_cdf[-1]

assert num_tournament_contenders >= 2

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

    # deterministic tournament selection - let top 2 winners become parents

    contender_ids = torch.randn((num_children, keep_fittest_len)).argsort(dim = -1)[..., :num_tournament_contenders]
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
    top2_winners = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices
    parents = get_at('p [t] g, p w -> p w g', participants, top2_winners)

    # cross over recombination of parents

    parent1, parent2 = parents.unbind(dim = 1)
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)

    pool = torch.cat((pool, children))

    # mutate genes in population

    # 1. sample from power law cdf

    num_mutate = torch.searchsorted(power_law_cdf, torch.rand(pool.shape[0]))

    # 2. get mutation mask from sampled mutation lengths

    mutate_mask = less('i j, i', torch.randn(pool.shape).argsort(dim = -1), num_mutate)

    # 3. mutate

    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    pool = torch.where(mutate_mask, pool + noise, pool)
    pool.clamp_(0, 255)

    generation += 1
