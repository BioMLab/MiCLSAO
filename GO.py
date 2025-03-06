import numpy as np
from collections import defaultdict
import time
import logging
from argument import arg_parse

args = arg_parse()
id_file = args.id
relation_file = args.relation
ic_file = args.ic

def load_features(feature):
    return np.loadtxt(feature, delimiter=",")

def load_id(id_file):
    id = {}
    with open(id_file, "r") as f:
        for line in f:
            term, idx = line.strip().split()
            id[term] = int(idx)
    return id

def load_relations(relation_file):
    relations = defaultdict(list)
    with open(relation_file, "r") as f:
        for line in f:
            child, relation, parent = line.strip().split()
            relations[child].append((relation, parent)) 
    return relations

def load_ic(ic_file):
    ic_values = {}
    with open(ic_file, "r") as f:
        for line in f:
            term, idx, ic = line.strip().split()
            ic_values[int(idx)] = float(ic) 
    return ic_values


ancestor_cache = {}
def preprocess_ancestors(relations):
    def dfs(term, visited, ancestors):
        if term in visited:
            return ancestors[term]
        visited.add(term)
        all_ancestors = set()
        for rel, parent in relations.get(term, []):
            if rel in ["is_a", "part_of"]:
                all_ancestors.add(parent)
                all_ancestors.update(dfs(parent, visited, ancestors))
        ancestors[term] = all_ancestors
        return all_ancestors

    visited = set()
    ancestors = {}
    for term in relations:
        dfs(term, visited, ancestors)
    return ancestors

def preprocess_descendants(relations):
    descendants_cache = defaultdict(set)
    for child, parents in relations.items():
        for rel, parent in parents:
            if rel in ["is_a", "part_of"]:
                descendants_cache[parent].add(child)
    def dfs(term, visited, descendants):
        if term in visited:
            return descendants[term]
        visited.add(term)
        all_descendants = set(descendants[term])  
        for child in descendants[term]:
            all_descendants.update(dfs(child, visited, descendants))
        descendants[term] = all_descendants 
        return all_descendants

    visited = set()
    for term in list(descendants_cache.keys()): 
        dfs(term, visited, descendants_cache)

    return descendants_cache

def preprocess_children(relations):
    children_cache = defaultdict(set)
    for child, parents in relations.items():
        for rel, parent in parents:
            if rel in ["is_a", "part_of"]:
                children_cache[parent].add(child)
    return children_cache


def compute_ic_weighted_mean_vectorized(indices, features, ic_values):
    if not indices:
        return np.zeros(features.shape[1])
    
    weights = np.array([ic_values.get(idx, 0) for idx in indices])
    if weights.sum() == 0:
        return features[indices].mean(axis=0)
    weighted_features = features[indices] * weights[:, None]
    
    weighted_mean = weighted_features.sum(axis=0) / (weights.sum() + 1e-6)
    
    return weighted_mean

def compute_projection(t_feature, c_feature):
    F = (t_feature + c_feature) / 2  
    F_norm = np.linalg.norm(F) + 1e-6
    t_proj = np.dot(t_feature, F) / (F_norm ** 2) * F 
    c_proj = np.dot(c_feature, F) / (F_norm ** 2) * F  
    return t_proj, c_proj


logging.info("Loading data...")
start_time = time.time()
features = load_features(feature)
id = load_id(id_file)
id2term = {v: k for k, v in id.items()}
relations = load_relations(relation_file)
ic_values = load_ic(ic_file)
logging.info("Data loaded successfully.")

logging.info("Precomputing ancestors...")
ancestor_cache = preprocess_ancestors(relations)
logging.info("Ancestor precomputation completed.")

logging.info("Precomputing descendants...")
descendants_cache = preprocess_descendants(relations)
logging.info("Descendant precomputation completed.")

logging.info("Precomputing children...")
children_cache = preprocess_children(relations)
logging.info("Children precomputation completed.")

def update_features(features, id, relations, ic_values, max_epochs=50, epsilon=1e-6):
    term_indices = {term: idx for term, idx in id.items()}
    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        previous_features = features.copy()

        for term, idx in id.items():
            ancestors = ancestor_cache.get(term, set())
            ancestor_indices = [term_indices[ancestor] for ancestor in ancestors if ancestor in term_indices]
            ancestor_mean = compute_ic_weighted_mean_vectorized(ancestor_indices, features, ic_values)

            parents = {parent for rel, parent in relations.get(term, []) if rel in ["is_a", "part_of"]}
            parent_indices = [term_indices[parent] for parent in parents if parent in term_indices]
            parent_mean = compute_ic_weighted_mean_vectorized(parent_indices, features, ic_values)

            descendants = descendants_cache.get(term, set())
            descendant_indices = [term_indices[descendant] for descendant in descendants if descendant in term_indices]
            descendant_mean = compute_ic_weighted_mean_vectorized(descendant_indices, features, ic_values)

            children = children_cache.get(term, set())
            child_indices = [term_indices[child] for child in children if child in term_indices]
            child_mean = compute_ic_weighted_mean_vectorized(child_indices, features, ic_values)

            t_feature = features[idx]
            for c_mean in [ancestor_mean, parent_mean, descendant_mean, child_mean]:
                if np.linalg.norm(c_mean) < 1e-6:
                    continue
                t_proj, c_proj = compute_projection(t_feature, c_mean)
                Xn_t = c_proj  
                alpha = np.linalg.norm(t_feature) / (np.linalg.norm(t_feature) + np.linalg.norm(Xn_t) + 1e-6)
                t_feature = alpha * t_feature + (1 - alpha) * Xn_t

            features[idx] = t_feature

        max_diff = np.max(np.abs(features - previous_features))
        epoch_end_time = time.time()

        logging.info(f"Epoch {epoch + 1}: max feature difference = {max_diff}, time taken = {epoch_end_time - epoch_start_time:.2f} seconds.")

    return features

logging.info("Starting feature adjustment...")
updated_features = update_features(features, id, relations, ic_values)