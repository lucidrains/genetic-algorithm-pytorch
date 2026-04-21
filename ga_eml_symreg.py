# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "einx",
#     "einops",
#     "fire",
#     "accelerate"
# ]
# ///

from __future__ import annotations

# citations
#
# EML Operator: "All elementary functions from a single operator" - Andrzej Odrzywolek (arXiv:2603.21852)
# Queen/Mutant Bee GA: "Queen-bee and Mutant-bee evolution for genetic algorithms" - S. Jung (2007)
# Fast GA: "Fast Genetic Algorithm" - Doerr et al. (arXiv:1703.03334)

import fire
from accelerate import Accelerator

from einx import get_at, less, where
from einops import rearrange, repeat

import torch
from torch import cat

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def batch_randperm(shape, device = None):
    return torch.randn(shape, device = device).argsort(dim = -1)

# eml functions

def generate_paper_target(d):
    if d < 2:
        return "x"

    s = "x"
    for _ in range(d - 1):
        s = f"x - ln({s})"
    return s

def eml(x, y, ln_min = 1e-20):
    return torch.exp(x) - torch.log(y.clamp(min = ln_min))

def evaluate_target_string(target_str, x_vals):
    context = {
        'x': x_vals,
        'log': torch.log,
        'ln': torch.log,
        'exp': torch.exp,
        'eml': eml
    }
    return eval(target_str, {"__builtins__": {}}, context)

def decode_tree(tree_array):
    def _decode(idx):
        if idx >= len(tree_array):
            return "ERROR"

        token = tree_array[idx]

        if token == 0:
            return "1"
        elif token == 1:
            return "x"
        elif token == 2:
            left = 2 * idx + 1
            right = 2 * idx + 2
            return f"eml({_decode(left)}, {_decode(right)})"

        return "ERROR"

    return _decode(0)

# vectorized evaluator

def evaluate_population(islands, tree_depth, x_vals):
    device = islands.device

    # pre-fill with 1.0 or x based on token (0 or 1)

    node_values = where('i p g, , n -> i p g n', islands == 0, 1., x_vals)

    # bottom-up evaluation

    for d in reversed(range(tree_depth)):
        start_idx = 2 ** d - 1
        end_idx = 2 ** (d + 1) - 1

        parent_indices = torch.arange(start_idx, end_idx, device = device)
        left_indices = 2 * parent_indices + 1
        right_indices = 2 * parent_indices + 2

        left_vals = node_values[:, :, left_indices, :]
        right_vals = node_values[:, :, right_indices, :]

        eml_vals = eml(left_vals, right_vals)

        node_values[:, :, parent_indices, :] = where(
            'i p k, i p k n, i p k n -> i p k n',
            islands[:, :, parent_indices] == 2,
            eml_vals,
            node_values[:, :, parent_indices, :]
        )

    return node_values[:, :, 0, :]

# genetic algorithm

def run_ga(
    target: str | None = None,
    depth: int | None = None,
    tree_depth: int | None = None,
    num_points = 100,
    pop_size = 1000,
    num_islands = 10,
    generations = 500,
    power_law_beta = 1.1,
    strong_mutation_rate = 0.25,
    frac_fittest_survive = 0.25,
    frac_tournament = 0.10,
    elite_frac = 0.05,
    migrate_every = 250,
    frac_migrate = 0.1,
    max_elite_age = 5,
    cpu = False,
    seed = 42
):
    torch.manual_seed(seed)

    assert not (exists(target) and exists(depth)), 'cannot pass in both depth and target'

    if not exists(target) and not exists(depth):
        depth = 3

    if exists(depth):
        target = generate_paper_target(depth)
        tree_depth = default(tree_depth, depth + 2)
    else:
        tree_depth = default(tree_depth, 5)

    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    print(f"using device: {device}")

    num_nodes = 2 ** (tree_depth + 1) - 1
    num_internal = 2 ** tree_depth - 1

    x_vals = torch.linspace(3.0, 5.0, num_points, device = device)

    try:
        y_target = evaluate_target_string(target, x_vals)
    except Exception as e:
        print(f"failed to parse or evaluate target string: {e}")
        return

    # derived constants

    keep_fittest_len = int(pop_size * frac_fittest_survive)
    num_elite = int(pop_size * elite_frac)
    num_tournament_contenders = max(2, int(keep_fittest_len * frac_tournament))
    num_children = pop_size - keep_fittest_len
    num_migrants = int(pop_size * frac_migrate)
    strong_num_mutate = int(strong_mutation_rate * num_nodes)

    # power law cdf for fast ga mutation

    power_law_cdf = torch.linspace(1, num_nodes, num_nodes, device = device).pow(-power_law_beta).cumsum(dim = -1)
    power_law_cdf = power_law_cdf / power_law_cdf[-1]

    # initialize population

    islands = torch.randint(0, 3, (num_islands, pop_size, num_nodes), device = device)
    islands[..., num_internal:] = islands[..., num_internal:].clamp(0, 1)

    ages = torch.zeros((num_islands, pop_size), dtype=torch.long, device=device)

    # fitness function

    def fitness_fn(pop_islands):
        y_pred = evaluate_population(pop_islands, tree_depth, x_vals)
        mse = (y_pred - y_target).pow(2).mean(dim = -1)

        inf_tensor = torch.tensor(float('inf'), device = device)
        mse = where('i p, , i p -> i p', torch.isnan(mse) | torch.isinf(mse), inf_tensor, mse)
        return 1. / (mse + 1e-8)

    # logging

    print(f"\ntarget: {target}")

    if exists(depth):
        print(f"target depth: {depth}")

    print(f"tree depth: {tree_depth}")
    print(f"population: {num_islands} islands x {pop_size} individuals")
    print("\n")

    # evolution loop

    for gen in range(1, generations + 1):
        island_fitnesses = fitness_fn(islands)

        # sort population by fitness within islands

        indices = island_fitnesses.sort(descending = True, dim = -1).indices
        islands = get_at('i [p1] g , i p2 -> i p2 g', islands, indices)
        island_fitnesses = get_at('i [f1], i f2 -> i f2', island_fitnesses, indices)
        ages = get_at('i [a1], i a2 -> i a2', ages, indices)

        ages += 1

        best_island_idx = island_fitnesses[:, 0].argmax()
        best_fitness = island_fitnesses[best_island_idx, 0].item()
        best_mse = (1. / best_fitness) - 1e-8

        if gen == 1 or divisible_by(gen, 10):
            print(f"generation {gen} | best mse: {best_mse:.6f}")

        # solved if mse is near zero

        if best_mse < 1e-7:
            print(f"\nexact match found at gen {gen}! stopping early.")
            break

        # keep the fittest

        elites = islands[:, :keep_fittest_len]
        elite_ages = ages[:, :keep_fittest_len]

        # queen bee strategy - tournament selection

        contender_ids = batch_randperm((num_islands, num_children, pop_size), device = device)[..., :num_tournament_contenders]

        participants = get_at('i [p] g, i c t -> i c t g', islands, contender_ids)
        tournament_results = get_at('i [f], i c tf -> i c tf', island_fitnesses, contender_ids)

        # drone is top 1 winner from tournament

        top1_winner = tournament_results.topk(1, dim = -1, largest = True, sorted = False).indices
        drone = get_at('i p [t] g, i p 1 -> i p g', participants, top1_winner)

        # queen is the best individual of the island

        queen = repeat(islands[:, 0], 'i g -> i c g', c = num_children)

        # strong mutation on the drone (mutant bee)

        strong_mutate_mask = batch_randperm(drone.shape, device = device) < strong_num_mutate
        noise = torch.randint(0, 3, drone.shape, device = device)
        mutated_drone = where('i p g, i p g, i p g -> i p g', strong_mutate_mask, noise, drone)
        mutated_drone[..., num_internal:] = mutated_drone[..., num_internal:].clamp(0, 1)

        # crossover

        uniform_mask = torch.randint(0, 2, mutated_drone.shape, device = device).bool()

        cut_points = torch.randint(0, num_nodes, (num_islands, num_children, 1), device = device)
        indices = torch.arange(num_nodes, device = device)
        traditional_mask = indices < cut_points

        # alternate islands between uniform and traditional (1-point) crossover

        is_uniform = (torch.arange(num_islands, device = device) % 2 == 0)
        crossover_mask = where('i, i c g, i c g -> i c g', is_uniform, uniform_mask, traditional_mask)

        children = where('i p g, i p g, i p g -> i p g', crossover_mask, mutated_drone, queen)
        children_ages = torch.zeros((num_islands, num_children), dtype=torch.long, device=device)

        islands = cat((elites, children), dim = 1)
        ages = cat((elite_ages, children_ages), dim = 1)

        # mutate genes via fast ga power law (protect young elites)

        rand_probs = torch.rand((num_islands, pop_size), device = device)
        num_mutate = torch.searchsorted(power_law_cdf, rand_probs)
        mutate_mask = less('i p g, i p -> i p g', batch_randperm(islands.shape, device = device), num_mutate)

        is_elite = torch.arange(pop_size, device=device) < num_elite
        is_elite = rearrange(is_elite, 'p -> 1 p 1')
        is_too_old = ages >= max_elite_age
        is_too_old = rearrange(is_too_old, 'i p -> i p 1')
        
        protect_mask = is_elite & ~is_too_old
        mutate_mask = mutate_mask & ~protect_mask

        noise = torch.randint(0, 3, islands.shape, device = device)
        islands = where('i p g, i p g, i p g -> i p g', mutate_mask, noise, islands)
        islands[..., num_internal:] = islands[..., num_internal:].clamp(0, 1)

        has_mutated = mutate_mask.any(dim=-1)
        ages = where('i p, , i p -> i p', has_mutated, 0, ages)

        # island migration

        if divisible_by(gen, migrate_every):
            island_rand_order = batch_randperm((num_islands, pop_size), device = device)
            islands = get_at('i [p1] g, i p2 -> i p2 g', islands, island_rand_order)
            ages = get_at('i [a1], i a2 -> i a2', ages, island_rand_order)

            migrants, remaining = islands[:, :num_migrants], islands[:, num_migrants:]
            migrants = torch.roll(migrants, 1, dims = 0)
            islands = cat((migrants, remaining), dim = 1)

            migrant_ages, remaining_ages = ages[:, :num_migrants], ages[:, num_migrants:]
            migrant_ages = torch.roll(migrant_ages, 1, dims = 0)
            ages = cat((migrant_ages, remaining_ages), dim = 1)

    # final evaluation

    island_fitnesses = fitness_fn(islands)
    flat_islands = rearrange(islands, 'i p g -> (i p) g')
    flat_fitnesses = rearrange(island_fitnesses, 'i f -> (i f)')

    best_idx = flat_fitnesses.argmax()
    best_mse = (1. / flat_fitnesses[best_idx].item()) - 1e-8
    best_tree = flat_islands[best_idx].tolist()

    print("\n--- training complete ---")
    print(f"global best mse: {best_mse:.6f}")
    print(f"discovered expression: {decode_tree(best_tree)}")

if __name__ == "__main__":
    fire.Fire(run_ga)
