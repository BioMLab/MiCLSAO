import random
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import random

def read_ids(file_path):
    ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            clean_line = line.strip()
            ids.add(clean_line)
    return ids

def filter_coexpress_file(input_file_path, output_file_path, valid_ids):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            parts = line.strip().split('\t')
            if parts[0] in valid_ids and parts[1] in valid_ids:
                output_file.write(line)

def filter_interaction_file(input_file_path, output_file_path, valid_ids):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if line.strip().split('\t')[0] in valid_ids:
                output_file.write(line)

def filter_fasta(fasta_path, ids, output_path):
    ids_to_find = set(ids) 
    with open(fasta_path, 'r') as fasta_file, open(output_path, 'w') as output_file:
        write_seq = False
        for line in fasta_file:
            if line.startswith('>'):
                ensg_id = line.split('|')[1].strip()
                if ensg_id in ids_to_find:
                    ids_to_find.remove(ensg_id)
                    write_seq = True
                    output_file.write(line)
                else:
                    write_seq = False
            elif write_seq:
                output_file.write(line)
            if not ids_to_find:
                break

def read_go_terms(file_path):
    with open(file_path, 'r') as file:
        go_terms = [line.strip() for line in file]
    return go_terms

def parse_go_relations(file_path):
    child_to_parents = defaultdict(set)
    all_terms = set()

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            child, relation, parent = parts[0], parts[1], parts[2]
            if relation in {'is_a', 'part_of'}:
                child_to_parents[child].add(parent)
                all_terms.update([child, parent])

    return child_to_parents, all_terms

def build_ancestor_dict(child_to_parents, all_terms):
    term_to_ancestors = defaultdict(set)

    for term in all_terms:
        term_to_ancestors[term].add(term)

    for term in all_terms:
        queue = deque([term])
        while queue:
            current = queue.popleft()
            if current in child_to_parents:
                for parent in child_to_parents[current]:
                    if parent not in term_to_ancestors[term]:
                        term_to_ancestors[term].add(parent)
                        queue.append(parent)

    return term_to_ancestors
def expand_lncrna_go_associations(lncrna_go_file, term_to_ancestors):
    expanded_associations = defaultdict(set)

    with open(lncrna_go_file, 'r') as file:
        for line in file:
            lncrna, go_term = line.strip().split()
            if go_term in term_to_ancestors:
                expanded_associations[lncrna].update(term_to_ancestors[go_term])

    return expanded_associations

def create_ancestor_matrix(go_terms, term_to_ancestors):
    num_terms = len(go_terms)
    ancestor_matrix = np.zeros((num_terms, num_terms), dtype=int)
    
    term_index = {term: idx for idx, term in enumerate(go_terms)}
    
    for term, ancestors in term_to_ancestors.items():
        if term in term_index:
            term_idx = term_index[term]
            for ancestor in ancestors:
                if ancestor in term_index:
                    ancestor_idx = term_index[ancestor]
                    ancestor_matrix[term_idx, ancestor_idx] = 1
    
    return ancestor_matrix


def write_associations_to_file(expanded_associations, output_file):
    with open(output_file, 'w') as file:
        for lncrna, go_terms in expanded_associations.items():
            for go_term in go_terms:
                file.write(f"{lncrna}\t{go_term}\n")

    input_file_path = 'lnc-GFP.txt'
    lncrna_output_path = 'lncRNAname.txt'
    go_output_path = 'GOname.txt'
    association_output_path = 'GFP_lncRNA-GO.txt'
    lncrna_set = set()
    go_set = set()
    associations = []

    with open(input_file_path, 'r') as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                lncrna, go_terms = parts
                lncrna_set.add(lncrna)
                for go_term in go_terms.split(','):
                    go_set.add(go_term)
                    associations.append(f"{lncrna}\t{go_term}")

    with open(lncrna_output_path, 'w') as lncrna_file:
        for lncrna in sorted(lncrna_set):
            lncrna_file.write(f"{lncrna}\n")
    with open(go_output_path, 'w') as go_file:
        for go in sorted(go_set):
            go_file.write(f"{go}\n")

    with open(association_output_path, 'w') as assoc_file:
        for assoc in associations:
            assoc_file.write(f"{assoc}\n")




def main():
    ids_path = 'ids.txt'
    coexpress_path = 'coexpress.txt'
    miRNA_path = 'miRNA.txt'
    protein_path = 'protein.txt'

    lncRNA_ids = read_ids(ids_path)

    filter_coexpress_file(coexpress_path, 'coexpress', lncRNA_ids)
    filter_interaction_file(miRNA_path, 'miRNA', lncRNA_ids)
    filter_interaction_file(protein_path, 'protein', lncRNA_ids)
    go_file = 'GO'
    lncrna_go_file = 'lncrna_go'
    output_file = 'output'
    root_terms = ['GO:0008150', 'GO:0005575', 'GO:0003674']

    child_to_parents, all_terms = parse_go_relations(go_file)
    term_to_ancestors = build_ancestor_dict(child_to_parents, all_terms)
    expanded_associations = expand_lncrna_go_associations(lncrna_go_file, term_to_ancestors)
    write_associations_to_file(expanded_associations, output_file)


if __name__ == "__main__":
    main()
