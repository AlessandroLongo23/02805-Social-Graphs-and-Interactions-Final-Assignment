import pandas as pd
import networkx as nx
import powerlaw
import numpy as np
import matplotlib.colors as mcolors


def get_dialogues(path_name="data/dialogues.csv"):
    return pd.read_csv(path_name)

def create_graph(path_name="data/dialogues.csv", edge_type="direct"):
    # Load dataset
    all_dialogues = get_dialogues(path_name)

    # Create graph
    G = nx.DiGraph()

    # Add dialogues as edges between characters
    for _, row in all_dialogues.iterrows():
        speaker_raw = row['Speaker(s)']
        addressee_raw = row['Addressee(s)']
        about_raw = row['About Character(s)']
        if pd.notna(speaker_raw):
            for speaker in [speaker.strip() for speaker in str(speaker_raw).split(';')]:
                G.add_node(speaker)
        if pd.notna(addressee_raw):
            for addressee in [addressee.strip() for addressee in str(addressee_raw).split(';')]:
                G.add_node(addressee)
        if pd.notna(about_raw):
            for about in [about.strip() for about in str(about_raw).split(';')]:
                G.add_node(about)

    G = add_edges(G, path_name="data/dialogues.csv", edge_type=edge_type)
        
    return G


def create_sentiment_network(DG, relationship_data, type="undirected"):
    # 1. Build lookup dict from relationship_data for fast access
    pair_lookup = {}
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                pair_lookup[(speaker, addressee)] = item['adjusted_sentiment']
    
    # 2. Create graph with combined sentiment data
    if type == "undirected":
        G = nx.Graph()
    elif type == "directed":
        G = nx.DiGraph()
    
    # Add all nodes from the directed graph
    G.add_nodes_from(DG.nodes())
    
    # Process edges: for each undirected pair, get both directional sentiments
    processed_pairs = set()
    
    if type == "undirected":
        for u, v in DG.edges():
            # Create canonical pair key (sorted tuple) to avoid duplicates
            pair_key = tuple(sorted([u, v]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Get sentiment in both directions (default to 0 if not exists)
            sentiment_u_to_v = pair_lookup.get((u, v), 0.0)
            sentiment_v_to_u = pair_lookup.get((v, u), 0.0)

            weight_u_to_v = DG[u][v].get('weight', 0) if DG.has_edge(u, v) else 0
            weight_v_to_u = DG[v][u].get('weight', 0) if DG.has_edge(v, u) else 0
            
            # Add undirected edge with both sentiment values
            G.add_edge(
                u, v, 
                sentiment_ab=sentiment_u_to_v, 
                sentiment_ba=sentiment_v_to_u,
                weight=weight_u_to_v + weight_v_to_u
            )
    elif type == "directed":
        for u, v in DG.edges():
            sentiment = DG[u][v].get('sentiment_ab', 0.0)
            weight = DG[u][v].get('weight', 0) if DG.has_edge(u, v) else 0
            G.add_edge(u, v, sentiment=sentiment, weight=weight)

    return G


def add_edges(G, path_name="data/dialogues.csv", edge_type="direct"):
    all_dialogues = get_dialogues(path_name)

    for _, row in all_dialogues.iterrows():
        speaker_raw = row['Speaker(s)']
        addressee_raw = row['Addressee(s)']
        about_raw = row['About Character(s)']

        dialogue = row['Dialogue']
        if edge_type == 'direct': 
            if pd.notna(speaker_raw) and pd.notna(addressee_raw):
                speaker_raw_list = [speaker.strip() for speaker in str(speaker_raw).split(';')]
                addressee_raw_list = [addressee.strip() for addressee in str(addressee_raw).split(';')]

                for speaker in speaker_raw_list:
                    for addressee in addressee_raw_list:
                        if G.has_edge(speaker, addressee):
                            G[speaker][addressee]['weight'] += len(dialogue)
                            G[speaker][addressee]['dialogues'].append(dialogue)
                        else:
                            G.add_edge(speaker, addressee, weight=len(dialogue), dialogues=[dialogue])

        elif edge_type == 'indirect':
            if pd.notna(speaker_raw) and pd.notna(about_raw):
                speaker_raw_list = [speaker.strip() for speaker in str(speaker_raw).split(';')]
                about_raw_list = [about.strip() for about in str(about_raw).split(';')]

                for speaker in speaker_raw_list:
                    for about in about_raw_list:
                        if G.has_edge(speaker, about):
                            G[speaker][about]['weight'] += len(dialogue)
                            G[speaker][about]['dialogues'].append(dialogue)
                        else:
                            G.add_edge(speaker, about, weight=len(dialogue), dialogues=[dialogue])
    
    # Normalize weights
    max_weight = 0
    for u, v, data in G.edges(data=True):
        if data['weight'] > max_weight:
            max_weight = data['weight']

    if max_weight > 0:
        for u, v, data in G.edges(data=True):
            data['normalized_weight'] = data['weight'] / max_weight

    return G


def get_normalized_weights(G):
    max_weight = 0
    for _, _, data in G.edges(data=True):
        if data['weight'] > max_weight:
            max_weight = data['weight']

    if max_weight > 0:
        for _, _, data in G.edges(data=True):
            data['normalized_weight'] = data['weight'] / max_weight
    return [data['weight'] for _, _, data in G.edges(data=True)]


def print_graph_data(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}\n")

    print(f"Min in_degree: {min([d for n, d in G.in_degree()])}")
    print(f"Average in_degree: {np.mean([d for n, d in G.in_degree()])}")
    print(f"Max in_degree: {max([d for n, d in G.in_degree()])}\n")

    print(f"Min out_degree: {min([d for n, d in G.out_degree()])}")
    print(f"Average out_degree: {np.mean([d for n, d in G.out_degree()])}")
    print(f"Max out_degree: {max([d for n, d in G.out_degree()])}\n")

    print(f"Average weight: {np.mean([G[u][v]['weight'] for u, v in G.edges()])}")
    avg_num_dialogues = np.mean([len(G[u][v]['dialogues']) for u, v in G.edges()])
    max_num_dialogues = max([len(G[u][v]['dialogues']) for u, v in G.edges()]) if G.number_of_edges() > 0 else 0
    print(f"Average number of dialogues: {avg_num_dialogues}")
    print(f"Maximum number of dialogues between two characters: {max_num_dialogues}")


def fit_powerlaw(degree_sequence):
    """
    Fit a power law distribution to the degree sequence.
    """
    degree_fit = powerlaw.Fit(degree_sequence, discrete=True, xmin=1, verbose=False)
    alpha = degree_fit.alpha
    xmin = degree_fit.xmin
    return alpha, xmin


def assortativity_analysis(G):
    r = nx.degree_assortativity_coefficient(G, weight="weight")
    print(f"Degree Assortativity Coefficient (r): {r:.4f}")


def get_continuous_edge_style(sentiment_ab, sentiment_ba):
    """
    Returns (color, width) for an edge based on bidirectional sentiment.
    Inputs are expected to be roughly in range [-1, 1].
    """
    
    # 1. Define your archetypal colors
    c_love = np.array(mcolors.to_rgb('green'))
    c_hate = np.array(mcolors.to_rgb('red'))
    c_conflict = np.array(mcolors.to_rgb('purple'))
    
    # 2. Calculate the "Forces"
    # Love: Both positive (Sum is positive)
    force_love = max(0, sentiment_ab + sentiment_ba)
    
    # Hate: Both negative (Sum is negative)
    force_hate = max(0, -(sentiment_ab + sentiment_ba))
    
    # Conflict: Disagreement magnitude
    force_conflict = abs(sentiment_ab - sentiment_ba)
    
    # 3. Mix the colors
    # We sum the weighted vectors and normalize
    total_force = force_love + force_hate + force_conflict
    
    if total_force == 0:
        final_color = (0.5, 0.5, 0.5) # Grey if absolutely no sentiment
    else:
        # Weighted average of the three base colors
        mixed_rgb = (force_love * c_love + 
                     force_hate * c_hate + 
                     force_conflict * c_conflict) / total_force
        final_color = mixed_rgb

    # 4. Calculate Continuous Width (Magnitude/Intensity)
    # Using Euclidean distance so (1,1) and (1,-1) have equal 'power'
    # We scale it slightly so very faint relationships are visible
    raw_magnitude = np.sqrt(sentiment_ab ** 2 + sentiment_ba ** 2)
    final_width = 1 + (raw_magnitude * 5)

    return final_color, final_width