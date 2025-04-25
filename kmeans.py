import torch
from torch import tensor

import einx
from einops import einsum, rearrange, repeat, reduce

from sklearn.datasets import make_blobs

points, _ = make_blobs(
  n_samples = 500,
  n_features = 16,
  centers = 10,
  cluster_std = 1.0
)

points = torch.from_numpy(points).float()

# metric, using silhouette score

def silhouette_score(clusters, points):
    dist = torch.cdist(clusters, points)
    top2_closest_dist = dist.topk(2, dim = -2, sorted = True, largest = False).values
    first, second = reduce(top2_closest_dist, '... top2 centers -> top2 ...', 'mean')
    return (second - first) / torch.max(first, second)

# regular kmeans

@torch.inference_mode()
def kmeans(points, num_clusters):
    num_points, dim = points.shape

    # init centers

    centers = torch.randn(num_clusters, dim)

    while True:
        # assign points to a center

        dist = torch.cdist(points, centers)

        center_ids = dist.argmin(dim = -1)
        expanded_center_ids = repeat(center_ids, 'n -> n d', d = dim)

        # average intra cluster distance

        next_centers = torch.zeros_like(centers).scatter_reduce_(0, expanded_center_ids, points, 'mean')

        # account for empty clusters by not changing

        is_empty_cluster = einx.not_equal('c, n -> c n', torch.arange(num_clusters), center_ids).all(dim = -1)

        next_centers = einx.where('c, c d, c d', is_empty_cluster, centers, next_centers)

        # stop if no changes between iterations

        if torch.allclose(centers, next_centers, atol = 1e-6):
            break

        centers.copy_(next_centers)

    return centers

# kmeans with genetic algorithm

@torch.inference_mode()
def genetic_kmeans(
    points,
    num_clusters,
    pop_size = 50,
    num_generations = 100,
    frac_fittest_selected = 0.25,
    frac_elites = 0.1,
    mutation_strength = 0.05
):
    num_points, dim = points.shape
    keep_fittest = int(frac_fittest_selected * pop_size)

    num_elites = int(frac_elites * pop_size)
    has_elites = num_elites > 0

    # init centers

    pop_centers = torch.randn(pop_size, num_clusters, dim)

    # generations

    for _ in range(num_generations):

        fitness = silhouette_score(pop_centers, points)

        # natural selection

        fitness, sel_indices = fitness.topk(keep_fittest, dim = -1)

        pop_centers = pop_centers[sel_indices]

        # tournament and crossover

        tourn_ids = torch.randn((pop_size - keep_fittest, keep_fittest)).argsort(dim = -1)

        competitor_fitness = fitness[tourn_ids]

        tourn_winner_indices = competitor_fitness.topk(2, dim = -1).indices

        parent1, parent2 = pop_centers[tourn_winner_indices].unbind(dim = 1)

        crossover_mask = torch.randint(0, 2, parent1.shape[:2]).bool()

        child = einx.where('n c, n c d, n c d', crossover_mask, parent1, parent2)

        pop_centers = torch.cat((pop_centers, child))

        # mutation

        if has_elites:
            elites, pop_centers = pop_centers[:num_elites], pop_centers[num_elites:]

        pop_centers += torch.randn_like(pop_centers) * points.std() * mutation_strength

        if has_elites:
            pop_centers = torch.cat((elites, pop_centers))

    fitness = silhouette_score(pop_centers, points)

    best_index = fitness.argmax(dim = -1).item()

    return pop_centers[best_index]

# running and evaluating

num_trials = 10
avg_kmeans = 0.
avg_genetic = 0.

for _ in range(num_trials):
    clusters = kmeans(points, 5)

    kmeans_score = silhouette_score(clusters, points)

    clusters = genetic_kmeans(points, 5)

    genetic_score = silhouette_score(clusters, points)

    avg_kmeans += kmeans_score / num_trials
    avg_genetic += genetic_score / num_trials

# print scores

print(f'\nscores over {num_trials} trials: (higher better)')

print(f'kmeans: {avg_kmeans:.4f}')
print(f'genetic kmeans: {avg_genetic:.4f}')
