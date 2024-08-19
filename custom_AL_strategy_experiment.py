import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, HeterophilousGraphDataset, WikipediaNetwork, Amazon, WebKB, Actor
from torch_geometric.nn import SAGEConv
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Define datasets
datasets = {
    'Cora': Planetoid(root='/tmp/Cora', name='Cora'),
    'CiteSeer': Planetoid(root='/tmp/CiteSeer', name='CiteSeer'),
    'PubMed': Planetoid(root='/tmp/PubMed', name='PubMed'),
    'Computers': Amazon(root='/tmp/Amazon', name='Computers'),
    'Photo': Amazon(root='/tmp/Amazon', name='Photo'),
    'Chameleon': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='chameleon'),
    'Squirrel': WikipediaNetwork(root='/tmp/WikipediaNetwork', name='squirrel'),
    'Cornell': WebKB(root='/tmp/WebKB', name='Cornell'),
    'Texas': WebKB(root='/tmp/WebKB', name='Texas'),
    'Wisconsin': WebKB(root='/tmp/WebKB', name='Wisconsin'),
    'Actor': Actor(root='/tmp/Actor'),
    'Questions': HeterophilousGraphDataset(root='/tmp/HeterophilousGraphDataset', name='questions'),
    'Roman-empire': HeterophilousGraphDataset(root='/tmp/roman-empire', name='roman-empire'),
    'CLUSTER': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='CLUSTER'),
    'PATTERN': GNNBenchmarkDataset(root='/tmp/GNNBenchmark', name='PATTERN')
}

def get_dataset_info(dataset):
    data = dataset[0]
    num_features = data.num_features if hasattr(data, 'num_features') else data.x.size(1)
    num_classes = max(data.y).item() + 1 if hasattr(data, 'y') else dataset.num_classes
    return num_features, num_classes

class GraphSAGE(torch.nn.Module):    
    def __init__(self, num_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 64)
        self.conv2 = SAGEConv(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, data, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        return correct.sum().item() / mask.sum().item()

def get_neighborhood_vector(node, edge_index, predictions, num_classes):
    neighbors = edge_index[1][edge_index[0] == node]
    neighbor_preds = predictions[neighbors]
    return torch.bincount(neighbor_preds, minlength=num_classes).float() / len(neighbors)

def random_active_learning(data, initial_labeled, n_queries, step_size, num_features, num_classes):
    num_nodes = data.num_nodes
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_mask[initial_labeled] = True

    accuracies = []
    n_labeled = []

    for i in range(n_queries):
        n_labeled.append(labeled_mask.sum().item())
        
        model = GraphSAGE(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(200):
            train(model, optimizer, data, labeled_mask)
        
        accuracy = test(model, data, ~labeled_mask)
        accuracies.append(accuracy)

        unlabeled_indices = torch.where(~labeled_mask)[0]
        new_labeled = np.random.choice(unlabeled_indices.numpy(), step_size, replace=False)
        labeled_mask[new_labeled] = True

    return n_labeled, accuracies

def custom_active_learning(data, initial_labeled, n_queries, step_size, num_features, num_classes):
    num_nodes = data.num_nodes
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_mask[initial_labeled] = True

    accuracies = []
    n_labeled = []

    for i in range(n_queries):
        n_labeled.append(labeled_mask.sum().item())
        
        model = GraphSAGE(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(200):
            train(model, optimizer, data, labeled_mask)
        
        accuracy = test(model, data, ~labeled_mask)
        accuracies.append(accuracy)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            predictions = out.argmax(dim=1)

        neighborhood_vectors = torch.stack([get_neighborhood_vector(n, data.edge_index, predictions, num_classes) for n in range(num_nodes)])

        unlabeled_indices = torch.where(~labeled_mask)[0]
        scores = torch.zeros(len(unlabeled_indices))

        for idx, node in enumerate(unlabeled_indices):
            labeled_vectors = neighborhood_vectors[labeled_mask]
            
            node_vector = neighborhood_vectors[node].numpy().reshape(1, -1)
            labeled_vectors_np = labeled_vectors.numpy()
            
            if np.isnan(node_vector).any() or np.isinf(node_vector).any() or np.isnan(labeled_vectors_np).any() or np.isinf(labeled_vectors_np).any():
                dissimilarity = 1.0
            else:
                if np.all(node_vector == 0) or np.all(labeled_vectors_np == 0):
                    dissimilarity = 1.0
                else:
                    similarity = cosine_similarity(node_vector, labeled_vectors_np)
                    dissimilarity = 1 - similarity.mean()
            
            if np.isnan(dissimilarity):
                dissimilarity = 1.0
            
            proportion = (neighborhood_vectors == neighborhood_vectors[node]).all(dim=1).float().mean().item()
            scores[idx] = dissimilarity + proportion

        new_labeled = unlabeled_indices[scores.topk(step_size).indices]
        labeled_mask[new_labeled] = True

    return n_labeled, accuracies

def entropy_active_learning(data, initial_labeled, n_queries, step_size, num_features, num_classes):
    num_nodes = data.num_nodes
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_mask[initial_labeled] = True

    accuracies = []
    n_labeled = []

    for i in range(n_queries):
        n_labeled.append(labeled_mask.sum().item())
        
        model = GraphSAGE(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(200):
            train(model, optimizer, data, labeled_mask)
        
        accuracy = test(model, data, ~labeled_mask)
        accuracies.append(accuracy)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            probs = out.exp()

        entropies = -(probs * probs.log()).sum(dim=1)
        unlabeled_indices = torch.where(~labeled_mask)[0]
        entropies = entropies[unlabeled_indices]
        
        new_labeled = unlabeled_indices[entropies.topk(step_size).indices]
        labeled_mask[new_labeled] = True

    return n_labeled, accuracies

def diversity_active_learning(data, initial_labeled, n_queries, step_size, num_features, num_classes):
    num_nodes = data.num_nodes
    labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_mask[initial_labeled] = True

    accuracies = []
    n_labeled = []

    for i in range(n_queries):
        n_labeled.append(labeled_mask.sum().item())
        
        model = GraphSAGE(num_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(200):
            train(model, optimizer, data, labeled_mask)
        
        accuracy = test(model, data, ~labeled_mask)
        accuracies.append(accuracy)

        unlabeled_indices = torch.where(~labeled_mask)[0]
        unlabeled_features = data.x[unlabeled_indices]
        labeled_features = data.x[labeled_mask]
        
        similarities = cosine_similarity(unlabeled_features, labeled_features)
        diversity_scores = 1 - similarities.max(axis=1)
        
        new_labeled = unlabeled_indices[np.argsort(diversity_scores)[-step_size:]]
        labeled_mask[new_labeled] = True

    return n_labeled, accuracies

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a directory to save the results
os.makedirs("results", exist_ok=True)

# Initialize results table
results_table = pd.DataFrame(columns=['Random', 'Custom', 'Entropy', 'Diversity'])

# Run active learning strategies for each dataset
# Run active learning strategies for each dataset
for dataset_name, dataset in datasets.items():
    try:
        print(f"Processing dataset: {dataset_name}")
        data = dataset[0]
        num_features, num_classes = get_dataset_info(dataset)
        
        # Initial random selection
        initial_labeled = np.random.choice(data.num_nodes, 3, replace=False)

        # Run all active learning strategies
        n_labeled_random, accuracies_random = random_active_learning(data, initial_labeled, n_queries=50, step_size=1, num_features=num_features, num_classes=num_classes)
        n_labeled_custom, accuracies_custom = custom_active_learning(data, initial_labeled, n_queries=50, step_size=1, num_features=num_features, num_classes=num_classes)
        n_labeled_entropy, accuracies_entropy = entropy_active_learning(data, initial_labeled, n_queries=50, step_size=1, num_features=num_features, num_classes=num_classes)
        n_labeled_diversity, accuracies_diversity = diversity_active_learning(data, initial_labeled, n_queries=50, step_size=1, num_features=num_features, num_classes=num_classes)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(n_labeled_random, accuracies_random, marker='o', label='Random Strategy')
        plt.plot(n_labeled_custom, accuracies_custom, marker='s', label='Custom Strategy')
        plt.plot(n_labeled_entropy, accuracies_entropy, marker='^', label='Entropy Strategy')
        plt.plot(n_labeled_diversity, accuracies_diversity, marker='D', label='Diversity Strategy')
        plt.xlabel('Number of Labeled Nodes')
        plt.ylabel('Accuracy')
        plt.title(f'Comparison of Active Learning Strategies on {dataset_name} Dataset')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"results/{dataset_name}_comparison.png")
        plt.close()

        # Save final accuracies to the results table
        results_table.loc[dataset_name] = [
            accuracies_random[-1],
            accuracies_custom[-1],
            accuracies_entropy[-1],
            accuracies_diversity[-1]
        ]
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {str(e)}")
        print(f"Skipping dataset {dataset_name}")
        continue

# Save the results table
results_table.to_csv("results/final_accuracies.csv")
print("Results saved in the 'results' directory.")