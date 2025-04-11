"""
Genetic Algorithm

but allowing for N > 2 parents, sidestepping a limitation of biology
"""

import torch
from einx import get_at, equal, where

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
MUTATION_RATE = 0.04
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25
ELITE_FRAC = 0.05

NUM_PARENTS = 4         # 4 parents
NUM_WORST_PARENTS = 1   # allow for non-fit individuals to breed for diversity reasons, in same vein as the queenbee mating strategy - todo, select based on the uniform fitness conclusions from Marcus Hutter

TOTAL_PARENTS = NUM_PARENTS + NUM_WORST_PARENTS

assert TOTAL_PARENTS >= 2

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
num_repro_and_mutate = keep_fittest_len
num_tournament_contenders = int(num_repro_and_mutate * FRAC_TOURNAMENT)
num_children = POP_SIZE - keep_fittest_len
num_mutate = MUTATION_RATE * gene_length

assert num_tournament_contenders >= NUM_PARENTS

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

    contender_ids = torch.randn((num_children, num_repro_and_mutate)).argsort(dim = -1)[..., :num_tournament_contenders]
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]

    top_individuals = tournaments.topk(NUM_PARENTS, dim = -1, largest = True, sorted = False).indices
    worst_individuals = tournaments.topk(NUM_WORST_PARENTS, dim = -1, largest = False, sorted = False).indices

    parents = get_at('p [t] g, p w -> p w g', participants, top_individuals)

    if NUM_WORST_PARENTS > 0:
        worst_parents = get_at('p [t] g, p w -> p w g', participants, worst_individuals)
        parents = torch.cat((parents, worst_parents), dim = 1)

    # crossover recombination of parents using uniform crossover, and allowing for any number of parents. there are no limits insilico

    pop, genome_length = parents.shape[0], parents.shape[-1]

    genome_assignments_from_parents = torch.randint(0, TOTAL_PARENTS, (pop, genome_length))
    parent_ids = torch.arange(TOTAL_PARENTS)
    genome_selection = equal('p g, w -> p w g', genome_assignments_from_parents, parent_ids)

    children = (genome_selection * parents).sum(dim = 1).long() # randomly select codes from the parents

    pool = torch.cat((pool, children))

    # elites can pass directly to next generation

    elites, pool = pool[:num_elite], pool[num_elite:]
    elites_fitnesses, fitnesses = fitnesses[:num_elite], fitnesses[num_elite:]

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    pool = torch.where(mutate_mask, pool + noise, pool)
    pool.clamp_(0, 255)

    # add back the elites

    pool = torch.cat((elites, pool))

    generation += 1
