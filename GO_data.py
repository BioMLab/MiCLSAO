from collections import defaultdict
import math
def calculate_max_depth_and_total_terms(parent_dict, children_dict):
    depth_cache = {}
    def compute_depth(term):
        if term in depth_cache:
            return depth_cache[term]
        if not parent_dict[term]:
            depth_cache[term] = 1
            return 1
        max_depth = 1 + max(compute_depth(parent) for parent in parent_dict[term])
        depth_cache[term] = max_depth
        return max_depth
    
    terms = set(parent_dict.keys()).union(set(children_dict.keys()))
    max_depth = max(compute_depth(term) for term in terms)
    total_terms = len(terms)
    return max_depth, total_terms

def parse_go_relationships(file_path):
    parent_dict = defaultdict(set)
    children_dict = defaultdict(set)
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                child = parts[0]
                relationship = parts[1]
                parent = parts[2]
                if relationship == "is_a" or relationship == "part_of":
                    parent_dict[child].add(parent)
                    children_dict[parent].add(child)

    return parent_dict, children_dict

def compute_ancestors(term, parent_dict, ancestors_cache):
    if term in ancestors_cache:
        return ancestors_cache[term]
    
    ancestors = set(parent_dict[term])
    for parent in parent_dict[term]:
        ancestors.update(compute_ancestors(parent, parent_dict, ancestors_cache))
    
    ancestors_cache[term] = ancestors
    return ancestors

def compute_descendants(term, children_dict, descendants_cache):
    if term in descendants_cache:
        return descendants_cache[term]
    
    descendants = set(children_dict[term])
    for child in children_dict[term]:
        descendants.update(compute_descendants(child, children_dict, descendants_cache))
    
    descendants_cache[term] = descendants
    return descendants

def calculate_ic(terms, parent_dict, children_dict, max_nodes):
    depth_cache = {}
    ancestors_cache = {}
    descendants_cache = {}

    def compute_depth(term):
        if term in depth_cache:
            return depth_cache[term]
        if not parent_dict[term]:
            depth_cache[term] = 1
            return 1
        max_depth = 1 + max(compute_depth(parent) for parent in parent_dict[term])
        depth_cache[term] = max_depth
        return max_depth

    ic_values = {}
    for term in terms:
        depth = compute_depth(term)
        ancestors = compute_ancestors(term, parent_dict, ancestors_cache)
        descendants = compute_descendants(term, children_dict, descendants_cache)
        
        specificity = depth * math.log(len(ancestors) + 1)
        coverage = 1 - (math.log(sum(1 / (compute_depth(desc) + 1) for desc in descendants) + 1) / math.log(max_nodes))
        
        ic = specificity * coverage
        ic_values[term] = ic

    return ic_values

def read_term2id(file_path):
    term2id = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                go_id, term_id = parts[0], parts[1]
                term2id[go_id] = term_id
    return term2id

def merge_ic_with_term2id(term2id, ic_values, output_path):
    with open(output_path, 'w') as file:
        for go_term, term_id in term2id.items():
            ic_value = ic_values.get(go_term, "NA") 
            file.write(f"{go_term}\t{term_id}\t{ic_value}\n")

def calculate_max_depth_and_total_terms(parent_dict, children_dict):
    depth_cache = {}
    def compute_depth(term):
        if term in depth_cache:
            return depth_cache[term]
        if not parent_dict[term]:
            depth_cache[term] = 1
            return 1
        max_depth = 1 + max(compute_depth(parent) for parent in parent_dict[term])
        depth_cache[term] = max_depth
        return max_depth
    
    terms = set(parent_dict.keys()).union(set(children_dict.keys()))
    max_depth = {term: compute_depth(term) for term in terms}
    total_terms = len(terms)
    return max_depth, total_terms

def parse_go_relationships(file_path):
    parent_dict = defaultdict(set)
    children_dict = defaultdict(set)
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                child = parts[0]
                relationship = parts[1]
                parent = parts[2]
                if relationship == "is_a" or relationship == "part_of":
                    parent_dict[child].add(parent)
                    children_dict[parent].add(child)

    return parent_dict, children_dict

def compute_descendants(term, children_dict, descendants_cache, visited):
    if term in descendants_cache:
        return descendants_cache[term]
    if term in visited:
        return 0  

    visited.add(term)
    total_descendants = 1  
    for child in children_dict.get(term, []):
        if child not in visited:
            total_descendants += compute_descendants(child, children_dict, descendants_cache, visited)
    
    descendants_cache[term] = total_descendants
    return total_descendants

def read_term2id(file_path):
    term2id = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                go_id, term_id = parts[0], parts[1]
                term2id[go_id] = term_id
    return term2id

def merge_ic_with_term2id(term2id, max_depth_dict, output_path):
    with open(output_path, 'w') as file:
        for go_term, term_id in term2id.items():
            max_depth = max_depth_dict.get(go_term, "NA")  
            file.write(f"{go_term}\t{term_id}\t{max_depth}\n")
