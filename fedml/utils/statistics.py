import numpy as np
import torch.nn.functional as tf
import torch
import math
from collections import Counter


def get_state_dict_size(state_dict, elem_size=None):
    total_size = 0
    for key, tensor in state_dict.items():
        size = tensor.numel() * (tensor.element_size() if elem_size is None else elem_size)
        total_size += size
    return total_size


def compute_probs(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    return [count / total_count for label, count in label_counts.items()]


def calculate_entropy(probabilities):
    # to prevent log(0) error
    return -np.sum(probabilities * np.log2(np.array(probabilities, copy=False) + 1e-10))


def calculate_gini(probabilities):
    return 1 - np.sum(np.array(probabilities, copy=False) ** 2)


def cos_distance(x, y, dim=1):
    return (1 - torch.cosine_similarity(x, y, dim=dim)) / 2


def pairwise_cos_sim(embeds, batch_size=64):
    embeds = tf.normalize(embeds, p=2, dim=1)
    embeds_transpose = embeds.t()
    num_samples = embeds.size(0)
    sim_mat = torch.zeros((num_samples, num_samples), device=embeds.device)
    # Compute pairwise cosine similarity in a batched manner
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_embeddings = embeds[start_idx:end_idx]
        # Compute cosine similarity for the current batch with all embeddings
        batch_similarity = torch.mm(batch_embeddings, embeds_transpose)
        # Place the batch results in the similarity matrix
        sim_mat[start_idx:end_idx, :] = batch_similarity
    return sim_mat


def mean_pairwise_cos_dist(embeds, batch_size=64):
    num_samples = embeds.size(0)
    distance_matrix = (1 - pairwise_cos_sim(embeds)) / 2
    # Sum the upper triangular part of the distance matrix to avoid duplicate pairs
    total_distance = torch.sum(torch.triu(distance_matrix, diagonal=1))
    # Calculate the number of pairs in the upper triangular part (excluding diagonal)
    num_pairs = num_samples * (num_samples - 1) / 2
    return total_distance.item() / num_pairs if num_pairs > 0 else 0.0


def entropy_of_cos_distances(embeds):
    distances = (1 - pairwise_cos_sim(embeds)) / 2
    distances = torch.triu(distances, diagonal=1)
    distances = distances[distances > 0]
    # Normalize distances to 0-1 range
    distances_normalized = (distances - torch.min(distances)) / (torch.max(distances) - torch.min(distances))
    # Compute histogram
    hist = torch.histc(distances_normalized, bins=10, min=0, max=1)
    hist = hist / hist.sum()
    hist = hist + 1e-10  # Avoid log(0)
    return calculate_entropy(hist.cpu().detach().numpy())


def consine_schedule(cur_iter, total_iter, max_val, min_val, alpha=1.0):
    cos_ann = (math.cos(math.pi * cur_iter / total_iter) + 1) / 2
    return min_val + (max_val - min_val) * cos_ann * alpha
