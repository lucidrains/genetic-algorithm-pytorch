import torch

# constants

GOAL = 'Attention is all you need'

POP_SIZE = 25
MUTATION_RATE = 0.05
FRAC_FITTEST_SURVIVE = 0.1

# helpers

def join(arr, sep = ''):
    return sep.join(arr)

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])

def decode(t):
    return join([chr(i) for i in t.tolist()])

def display_gene_pool(pool, costs):
    for gene, cost in zip(pool, costs):
        print(f"{decode(gene)} ({cost.item()})")

def calc_costs(pool, target_gene):
    return torch.square(pool - target_gene).sum(dim = -1)

def sort_gene_pool_by_costs(pool, costs):
    indices = costs.sort().indices
    return pool[indices], costs[indices]

# init gene pool

gene_length = len(GOAL)
gene_midpoint = gene_length // 2
target_gene = encode(GOAL)

pool_shape = (POP_SIZE, gene_length)
pool = torch.randint(0, 255, pool_shape)

keep_fittest_len = int(POP_SIZE * FRAC_FITTEST_SURVIVE)
num_children = POP_SIZE - keep_fittest_len
num_mutate = MUTATION_RATE * gene_length

# genetic algorithm

generation = 1
solved = False

while not solved:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness (inverse costs)

    costs = calc_costs(pool, target_gene)
    pool, costs = sort_gene_pool_by_costs(pool, costs)

    # keep the fittest

    pool, costs = pool[:keep_fittest_len], costs[:keep_fittest_len]

    # cross over recombination of fittest

    rand_parents = torch.randn((num_children, keep_fittest_len)).argsort(dim = -1)[..., :2]
    parent1_indices, parent2_indices = rand_parents.unbind(dim = -1)
    parent1, parent2 = pool[parent1_indices], pool[parent2_indices]
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)

    pool = torch.cat((pool, children), dim = 0)

    # mutate genes in population

    mutate_mask = torch.randn(pool_shape).argsort(dim = -1) < num_mutate
    ones = torch.ones(pool_shape).int()
    noise = torch.where(torch.randn(pool_shape) > 0, ones, -ones)
    pool = torch.where(mutate_mask, pool + noise, pool)
    pool.clamp_(0, 255)

    costs = calc_costs(pool, target_gene)

    # solved if any cost is 0

    solved = (costs == 0).any()

    if solved:
        pool, costs = sort_gene_pool_by_costs(pool, costs)

    # display every generation

    display_gene_pool(pool, costs)

    generation += 1
