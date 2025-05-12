import numpy as np
import random
import torch
from typing import List
from torch.utils.data import Dataset, Subset, random_split

def iid_split(dataset: Dataset, num_clients: int) -> List[Subset]:
    """
    Split a dataset into approximately equal IID (Independent and Identically Distributed) subsets.

    Each client receives a random subset of the dataset, preserving the global label distribution.

    Args:
        dataset (Dataset): The complete dataset (e.g., CIFAR-100 training set).
        num_clients (int): Number of client datasets to generate.

    Returns:
        List[Subset]: A list of `Subset` objects, one for each client.
    """
    data_size = len(dataset)
    indices = np.random.permutation(data_size)  # Shuffle all data indices

    # Determine number of samples per client (handle remainder)
    split_sizes = [data_size // num_clients] * num_clients
    for i in range(data_size % num_clients):
        split_sizes[i] += 1

    return random_split(dataset, split_sizes)


def non_iid_split(
    dataset: Dataset,
    num_clients: int,
    nc: int,
    num_classes: int = 100
) -> List[Subset]:
    """
    Split a dataset into non-IID subsets across clients, allowing for sampling with replacement.

    Args:
        dataset (Dataset): The complete dataset to be split.
        num_clients (int): Number of clients to simulate.
        nc (int): Number of distinct classes to assign to each client (with replacement).
        num_classes (int): Total number of classes in the dataset (default: 100 for CIFAR-100).

    Returns:
        List[Subset]: A list of `Subset` objects, each corresponding to one client.

    Raises:
        ValueError: If any client ends up with an empty dataset.
    """
    # Step 1: Index samples by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Step 2: Create shards per class (with replacement)
    shards_per_class = []
    for indices in class_indices:
        if len(indices) > 0:
            # Randomly shuffle and allow for repeated sampling
            shards = list(np.array_split(indices, max(1, len(indices) // num_clients)))
            shards_per_class.append(shards)
        else:
            shards_per_class.append([])

    # Step 3: Assign Nc classes (with replacement) to each client
    client_data_indices = [[] for _ in range(num_clients)]
    available_classes = [cls for cls, shards in enumerate(shards_per_class) if shards]

    for client_id in range(num_clients):
        # Sample with replacement if needed
        chosen_classes = random.choices(available_classes, k=nc)

        for cls in chosen_classes:
            if shards_per_class[cls]:
                shard = random.choice(shards_per_class[cls])
                client_data_indices[client_id].extend(shard)

    # Step 4: Wrap in Subset and validate non-empty assignment
    client_subsets = []
    for idx_list in client_data_indices:
        if not idx_list:
            raise ValueError("Client received 0 samples. Check non-IID sharding logic or dataset size.")
        client_subsets.append(Subset(dataset, idx_list))

    return client_subsets
