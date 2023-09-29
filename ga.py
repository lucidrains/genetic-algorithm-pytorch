"""
Genetic Algorithm - formalized by John H. Holland in 1992, but has been talked about since 1960-70s

https://www.researchgate.net/figure/Hollands-canonical-genetic-algorithm-Holland-1992_fig4_221174380
"""

import torch

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
MUTATION_RATE = 0.04
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25
ELITE_FRAC = 0.05

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
num_elite = int(ELITE_FRAC * POP_SIZE)
num_repro_and_mutate = keep_fittest_len - num_elite
num_tournament_contenders = int(num_repro_and_mutate * FRAC_TOURNAMENT)
num_children = POP_SIZE - keep_fittest_len
num_mutate = MUTATION_RATE * gene_length

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

    # elites can pass directly to next generation

    elites, pool = pool[:num_elite], pool[num_elite:]
    elites_fitnesses, fitnesses = fitnesses[:num_elite], fitnesses[num_elite:]

    # deterministic tournament selection - let top 2 winners become parents

    contender_ids = torch.randn((num_children, num_repro_and_mutate)).argsort(dim = -1)[..., :num_tournament_contenders]
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
    top2_winners = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices
    top2_winners = top2_winners.unsqueeze(-1).expand(-1, -1, gene_length)
    parents = participants.gather(1, top2_winners)

    # cross over recombination of parents

    parent1, parent2 = parents.unbind(dim = 1)
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)

    pool = torch.cat((pool, children))

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    pool = torch.where(mutate_mask, pool + noise, pool)
    pool.clamp_(0, 255)

    # add back the elites

    pool = torch.cat((elites, pool))

    generation += 1
