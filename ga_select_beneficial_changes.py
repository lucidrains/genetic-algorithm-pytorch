"""
Genetic Algorithm

sidestepping biology, do N mutations and select for the one with most benefit

also do a few children and allow only the top one to survive, this one more biologically plausible
"""

import torch
from einx import get_at

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
MUTATION_RATE = 0.04
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25
ELITE_FRAC = 0.05

NUM_MUTATIONS = 20            # do 20 mutations and take the top
NUM_CHILDREN_PER_PARENTS = 4  # take top child

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

def fitness_fn(gene, target_gene, eps = 0.):
    return 1. / (gene - target_gene).pow(2).sum(dim = -1).clamp(min = eps)

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    fitnesses = fitness_fn(pool, target_gene)

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
    parents = get_at('p [t] g, p w -> p w g', participants, top2_winners)

    # uniform cross over recombination of parents

    parent1, parent2 = parents.unbind(dim = 1)
    crossover_mask = torch.randint(0, 2, (NUM_CHILDREN_PER_PARENTS, *parent1.shape)).bool()
    children = torch.where(crossover_mask, parent1, parent2)

    fitness_for_children = fitness_fn(children, target_gene, eps = 1e-5)
    top_child_id = fitness_for_children.argmax(dim = 0)

    selected_children = get_at('[c] p g, p -> p g', children, top_child_id)

    pool = torch.cat((pool, selected_children))

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate
    noise = torch.randint(0, 2, (NUM_MUTATIONS, *pool.shape)) * 2 - 1
    mutated_pool = pool + noise

    fitness_for_mutations = fitness_fn(mutated_pool, target_gene, eps = 1e-5)
    best_mutation_index = fitness_for_mutations.argmax(dim = 0)

    selected_mutated_pool = get_at('[m] p g, p -> p g', mutated_pool, best_mutation_index)

    pool = torch.where(mutate_mask, selected_mutated_pool, pool)
    pool.clamp_(0, 255)

    # add back the elites

    pool = torch.cat((elites, pool))

    generation += 1
