"""
Genetic Algorithms with Islands: periodic migration for diversity, and island resets for populations stuck in local minimum

https://link.springer.com/chapter/10.1007/BFb0027170
"""

import torch
from torch import cat
from einx import get_at
from einops import rearrange

# constants

GOAL = 'Attention is all you need'

ISLANDS = 5
POP_SIZE = 100
MUTATION_RATE = 0.04
FRAC_FITTEST_SURVIVE = 0.25
FRAC_TOURNAMENT = 0.25
ELITE_FRAC = 0.05

MIGRATE_EVERY = 50
FRAC_MIGRATE = 0.1
NUM_MIGRANTS = int(POP_SIZE * FRAC_MIGRATE)

ISLAND_RESET_EVERY = 100
NUM_ISLANDS_RESET = 1
ISLAND_RESET_TOURNAMENT_SIZE = 25

DISPLAY_TOP_ISLAND_POP = 10 # number of top individuals per island to display

# helpers

def divisible_by(num, den):
    return (num % den) == 0

def batch_randperm(shape):
    return torch.randn(shape).argsort(dim = -1)

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])

# derived constants

gene_length = len(GOAL)
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

islands = torch.randint(0, 255, (ISLANDS, POP_SIZE, gene_length))

def fitness_fn(islands, target):
    return 1. / (islands - target).pow(2).sum(dim = -1)

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    island_fitnesses = fitness_fn(islands, target_gene)

    indices = island_fitnesses.sort(descending = True, dim = -1).indices

    islands = get_at('i [p1] g , i p2 -> i p2 g', islands, indices)
    island_fitnesses = get_at('i [p1], i p2 -> i p2', island_fitnesses, indices)

    # keep the fittest

    islands, island_fitnesses = islands[:, :keep_fittest_len], island_fitnesses[:, :keep_fittest_len]

    # display every generation

    for island_id, (pool, fitnesses) in enumerate(zip(islands, island_fitnesses)):
        print(f'\nisland {island_id + 1}:\n')

        for gene, fitness in zip(pool[:DISPLAY_TOP_ISLAND_POP], fitnesses[:DISPLAY_TOP_ISLAND_POP]):
            print(f"{decode(gene)} ({fitness.item():.3f})")

    # solved if any fitness is inf

    if (island_fitnesses == float('inf')).any():
        break

    # deterministic tournament selection - let top 2 winners become parents

    contender_ids = torch.randn((ISLANDS, num_children, num_repro_and_mutate)).argsort(dim = -1)[..., :num_tournament_contenders]

    participants = get_at('i [p] g, i c t -> i c t g', islands, contender_ids)
    tournament_results = get_at('i [f], i c tf -> i c tf', island_fitnesses, contender_ids)

    top2_winners = tournament_results.topk(2, dim = -1, largest = True, sorted = False).indices

    parents = get_at('i p [t] g, i p w -> w i p g', participants, top2_winners)

    # cross over recombination of parents

    parent1, parent2 = parents
    uniform_crossover_mask = torch.randint(0, 2, parent1.shape).bool()
    children = torch.where(uniform_crossover_mask, parent1, parent2)

    islands = cat((islands, children), dim = 1)

    # mutate genes in population

    mutate_mask = batch_randperm(islands.shape) < num_mutate
    noise = torch.randint(0, 2, islands.shape) * 2 - 1
    islands = torch.where(mutate_mask, islands + noise, islands)
    islands.clamp_(0, 255)

    # migration

    if divisible_by(generation, MIGRATE_EVERY):
        # migrants are randomly chosen for now

        island_rand_order = batch_randperm((ISLANDS, POP_SIZE))
        islands = get_at('i [p1] g, i p2 -> i p2 g', islands, island_rand_order)

        migrants, islands = islands[:, :NUM_MIGRANTS], islands[:, NUM_MIGRANTS:]
        migrants = torch.roll(migrants, 1, dims = 0)

        islands = cat((migrants, islands), dim = 1)

    # island reset strategy
    # purportedly effective in recent LLM inference time search papers - funsearch and mind-evolution

    if divisible_by(generation, ISLAND_RESET_EVERY):
        island_fitnesses = fitness_fn(islands, target_gene)

        # just take average of island fitnesses for now

        average_island_fitnesses = island_fitnesses.mean(dim = -1)
        sort_indices = average_island_fitnesses.sort(dim = -1).indices

        islands = islands[sort_indices]

        islands = islands[NUM_ISLANDS_RESET:] # only keep the best performing islands
        island_fitnesses = island_fitnesses[NUM_ISLANDS_RESET:]

        # repopulate the number of island resets with children from randomly selected individuals from other islands

        all_individuals = rearrange(islands, 'i p g -> (i p) g')
        all_fitnesses = rearrange(island_fitnesses, 'i f -> (i f)')

        num_individuals = all_individuals.shape[0]

        # perform reliable tournament strategy for repopulating islands

        tournament_participant_ids = batch_randperm((NUM_ISLANDS_RESET, POP_SIZE, POP_SIZE))[..., :ISLAND_RESET_TOURNAMENT_SIZE]

        participants = get_at('[global_pop] g, i p t -> t i p g', all_individuals, tournament_participant_ids)
        participant_fitnesses = get_at('[global_pop], i p t -> t i p', all_fitnesses, tournament_participant_ids)

        parent_indices = participant_fitnesses.topk(2, dim = 0, largest = True, sorted = False).indices

        parents = get_at('[p1] g, parents i p2 -> parents i p2 g', all_individuals, parent_indices)
        parent1, parent2 = parents

        uniform_crossover_mask = torch.randint(0, 2, parent1.shape).bool()
        new_islands = torch.where(uniform_crossover_mask, parent1, parent2)

        islands = cat((islands, new_islands))     

    generation += 1
