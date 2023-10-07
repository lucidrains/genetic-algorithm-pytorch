"""
Bee Colonies Genetic Algorithm

Here we simulate different colonies to maintain diversity. At each generation, one allow a small subset of bees from each colony to immigrate to another
"""

import torch
from einops import repeat, rearrange

# constants

GOAL = 'Attention is all you need'

COLONIES = 10
POP_SIZE = 250
MUTATION_PROB = 0.05
STRONG_MUTATION_PROB = 0.15
NUM_TOURNAMENT_PARTICIPANTS = 25
MIGRATE_EVERY = 5
FRAC_BEES_MIGRANTS = 0.1

# encode and decode functions

def exists(v):
    return v is not None

def encode(s):
    return torch.tensor([ord(c) for c in s])

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])

def calc_fitness(genes, target):
    return 1. / (genes - target).square().sum(dim = -1)

# derived constants

gene_length = len(GOAL)
target_gene = encode(GOAL)

num_code_mutate = MUTATION_PROB * gene_length
strong_num_code_mutate = STRONG_MUTATION_PROB * gene_length

num_bees_migrate = int((POP_SIZE - 1) * FRAC_BEES_MIGRANTS)

# queen bee genetic algorithm

generation = 1

colonies = torch.randint(0, 255, (COLONIES, POP_SIZE - 1, gene_length))
colonies_arange = torch.arange(COLONIES)[..., None]

queens = torch.randint(0, 255, (COLONIES, gene_length))
queen_fitnesses = calc_fitness(queens, target_gene)

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    colony_fitnesses = calc_fitness(colonies, target_gene)

    indices = colony_fitnesses.sort(descending = True).indices
    colonies, colony_fitnesses = colonies[colonies_arange, indices], colony_fitnesses[colonies_arange, indices]

    # display every generation

    for i, (pool, fitnesses) in enumerate(zip(colonies[:, :10], colony_fitnesses[:, :10])):
        print(f'\ncolony {i + 1}:\n')

        if exists(queens):
            queen, queen_fitness = queens[i], queen_fitnesses[i]
            print(f"Q: {decode(queen)} ({queen_fitness.item():.3f})\n")

        for gene, fitness in zip(pool, fitnesses):
            print(f"{decode(gene)} ({fitness.item():.3f})")
    
    # if one of the children has a better fitness than queen, that child becomes the new queen
    # and the queen replaces the worst bee in the population, kept around for at least one generation more

    has_new_queen = colony_fitnesses[:, 0] > queen_fitnesses

    pop_arange = torch.arange(POP_SIZE)
    pop_arange_with_offset = pop_arange + has_new_queen[:, None]

    colonies = torch.cat((
        queens[:, None, :],
        colonies,
        queens[:, None, :]
    ), dim = -2)

    colony_fitnesses = torch.cat((
        queen_fitnesses[:, None],
        colony_fitnesses,
        queen_fitnesses[:, None]
    ), dim = -1)

    colonies = colonies[colonies_arange, pop_arange_with_offset]
    colony_fitnesses = colony_fitnesses[colonies_arange, pop_arange_with_offset]

    queens, colonies = colonies[:, 0], colonies[:, 1:]
    queen_fitnesses, colony_fitnesses = colony_fitnesses[:, 0], colony_fitnesses[:, 1:]

    # solved if any fitness is inf

    if (queen_fitnesses == float('inf')).any():
        print(f'\nsolved at generation {generation}')
        break

    # deterministic tournament selection - let top winner become parent with queen

    colonies_arange_ = colonies_arange[..., None]
    contender_ids = torch.randn((COLONIES, POP_SIZE - 1, POP_SIZE - 1)).argsort(dim = -1)[..., :NUM_TOURNAMENT_PARTICIPANTS]
    participants, tournaments = colonies[colonies_arange_, contender_ids], colony_fitnesses[colonies_arange_, contender_ids]
    top_winner = tournaments.topk(1, dim = -1, largest = True, sorted = False).indices
    top_winner = repeat(top_winner, '... -> ... g', g = gene_length)
    parents = participants.gather(2, top_winner).squeeze(2)

    # potential parents with queen is strongly mutated ("Mutant Bee")

    strong_mutate_mask = torch.randn(parents.shape).argsort(dim = -1) < strong_num_code_mutate
    noise = torch.randint(0, 2, parents.shape) * 2 - 1
    mutated_parents = torch.where(strong_mutate_mask, parents + noise, parents)
    mutated_parents.clamp_(0, 255)

    # do a random 50% genetic code mix instead of contiguous crossover at midpoint

    rand_mix_mask = torch.randn(mutated_parents.shape).argsort(dim = -1) < (gene_length // 2)

    colonies = torch.where(
        rand_mix_mask,
        rearrange(queens, 'c ... -> c 1 ...'),
        mutated_parents
    )

    # mutate genes in population

    mutate_mask = torch.randn(colonies.shape).argsort(dim = -1) < num_code_mutate
    noise = torch.randint(0, 2, colonies.shape) * 2 - 1

    colonies = torch.where(mutate_mask, colonies + noise, colonies)
    colonies.clamp_(0, 255)

    # allow a subset of bees to migrate to adjacent colonies

    if not (generation % MIGRATE_EVERY) and num_bees_migrate > 0:
        colonies, migrant_colonies = colonies[:, :-num_bees_migrate], colonies[:, -num_bees_migrate:]
        migrant_colonies = torch.roll(migrant_colonies, 1, dims = 0)
        colonies = torch.cat((colonies, migrant_colonies), dim = 1)

    # increment generation

    generation += 1
