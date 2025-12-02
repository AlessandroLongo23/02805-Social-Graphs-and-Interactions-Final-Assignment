import networkx as nx 
import numpy as np

def create_graph_from_matrix(matrix, row_characters, col_characters):
    G = nx.DiGraph()
    for i, row_character in enumerate(row_characters):
        for j, col_character in enumerate(col_characters):
            if not np.isnan(matrix[i, j]):
                G.add_edge(row_character, col_character, weight=matrix[i, j])

    # 4. Calculate charisma (Bayesian average) for each character and add it to the node attribute 'charisma'
    # S_adj = (C * m + Σs) / (C + n)
    # where:
    #   C = confidence parameter (avg incoming edges per character)
    #   m = global average sentiment
    #   Σs = sum of incoming sentiment scores
    #   n = total number of incoming edges for the character
    
    # Collect all incoming sentiments for each character
    character_incoming = {}  # {character: [list of sentiment scores]}
    all_sentiments = []
    
    for u, v, data in G.edges(data=True):
        sentiment = data.get('weight', 0.0)
        if v not in character_incoming:
            character_incoming[v] = []
        character_incoming[v].append(sentiment)
        all_sentiments.append(sentiment)
    
    # Calculate global statistics for Bayesian Average
    global_mean = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
    
    # Calculate C: average number of incoming edges per character
    num_characters = len(character_incoming) if character_incoming else 1
    total_incoming_edges = len(all_sentiments)
    C = total_incoming_edges / num_characters if num_characters > 0 else 1.0
    
    # Calculate charisma (Bayesian average) for each character
    for node in G.nodes():
        if node in character_incoming:
            incoming_scores = character_incoming[node]
            n = len(incoming_scores)
            sum_s = sum(incoming_scores)
            
            # Bayesian Average
            charisma = (C * global_mean + sum_s) / (C + n) if (C + n) > 0 else 0.0
            G.nodes[node]['charisma'] = charisma
        else:
            # Character with no incoming edges gets charisma = 0
            G.nodes[node]['charisma'] = 0.0

    return G