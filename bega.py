"""
Queen-bee evolution for genetic algorithms - Jung 2003

Inspired by evolution of bees, the fittest solution is designated the "queen", and the rest of the population contends to mate with it. The strong exploitation is balanced by a higher than normal mutation rate.
For some problems, the paper claims convergence at 2-3 orders of magnitude faster

https://www.researchgate.net/publication/3385719_Queen-bee_evolution_for_genetic_algorithms
"""

import torch
from einops import repeat

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 100
MUTATION_PROB = 0.04
STRONG_MUTATION_RATE = 0.1
STRONG_MUTATION_PROB = 0.25
NUM_TOURNAMENT_PARTICIPANTS = 25

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])

# derived constants

gene_length = len(GOAL)
gene_midpoint = gene_length // 2
target_gene = encode(GOAL)

strong_mutate_pool_size = STRONG_MUTATION_RATE * POP_SIZE
num_code_mutate = MUTATION_PROB * gene_length
strong_num_code_mutate = STRONG_MUTATION_PROB * gene_length

# queen bee genetic algorithm

generation = 1

pool = torch.randint(0, 255, (POP_SIZE, gene_length))

queen = queen_fitness = None

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    fitnesses = 1. / torch.square(pool - target_gene).sum(dim = -1)

    indices = fitnesses.sort(descending = True).indices
    pool, fitnesses = pool[indices], fitnesses[indices]

    # display every generation

    if queen is not None:
        print("queen:")
        print(f"{decode(queen)} ({queen_fitness.item():.3f})\n")

    for gene, fitness in zip(pool, fitnesses):
        print(f"{decode(gene)} ({fitness.item():.3f})")

    # solved if any cost is 0

    if (fitnesses == float('inf')).any():
        break
    
    # if one of the children has a better fitness than queen, that child becomes the new queen
    # and the queen replaces the worst bee in the population, kept around for at least one generation more

    if queen is not None and queen_fitness < fitnesses[0]:
        pool = torch.cat((pool, queen[None, :]), dim = 0)
        fitnesses = torch.cat((fitnesses, queen_fitness[None]), dim = 0)
        queen = queen_fitness = None

    # separate the queen bee from the rest of the population

    if queen is None:
        queen, pool = pool[0], pool[1:]
        queen_fitness, fitnesses = fitnesses[0], fitnesses[1:]

    # deterministic tournament selection - let top winner become parent with queen

    contender_ids = torch.randn((POP_SIZE - 1, POP_SIZE - 1)).argsort(dim = -1)[..., :NUM_TOURNAMENT_PARTICIPANTS]
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
    top_winner = tournaments.topk(1, dim = -1, largest = True, sorted = False).indices
    top_winner = repeat(top_winner, '... -> ... g', g = gene_length)
    parents = participants.gather(1, top_winner).squeeze(1)

    # cross over all chosen drones with the queen

    queen_parents = repeat(queen, '... -> p ...', p = POP_SIZE - 1)
    pool = torch.cat((queen_parents[:, :gene_midpoint], parents[:, gene_midpoint:]), dim = -1)

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_code_mutate
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    mutated_pool = torch.where(mutate_mask, pool + noise, pool)

    strong_mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < strong_num_code_mutate
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    strong_mutated_pool = torch.where(strong_mutate_mask, pool + noise, pool)

    strong_mutate_pool_mask = torch.randn(POP_SIZE - 1).argsort(dim = -1) < strong_mutate_pool_size

    pool = torch.where(strong_mutate_pool_mask[:, None], strong_mutated_pool, mutated_pool)
    pool.clamp_(0, 255)

    # increment generation

    generation += 1
