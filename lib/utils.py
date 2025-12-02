"""
Utility functions for ASoIaF Network Analysis.
Provides graph creation, data loading, and analysis utilities.
"""
import pandas as pd
import networkx as nx
import powerlaw
import numpy as np
from typing import Tuple, Dict, List, Optional, Any


# =============================================================================
# DATA LOADING
# =============================================================================

def get_dialogues(path_name: str = "data/dialogues.csv") -> pd.DataFrame:
    """
    Load dialogue data from CSV file.
    
    Args:
        path_name: Path to the dialogues CSV file
    
    Returns:
        DataFrame containing dialogue data
    """
    return pd.read_csv(path_name)


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_graph(
    path_name: str = "data/dialogues.csv",
    edge_type: str = "direct"
) -> nx.DiGraph:
    """
    Create a directed graph from dialogue data.
    
    Args:
        path_name: Path to the dialogues CSV file
        edge_type: Type of edges to create:
            - 'direct': Speaker → Addressee
            - 'indirect': Speaker → About Character
    
    Returns:
        NetworkX DiGraph with weighted edges and dialogue lists
    """
    all_dialogues = get_dialogues(path_name)
    G = nx.DiGraph()
    
    # Add all characters as nodes
    for _, row in all_dialogues.iterrows():
        # Add speakers
        if pd.notna(row['Speaker(s)']):
            for speaker in _parse_character_list(row['Speaker(s)']):
                G.add_node(speaker)
        
        # Add addressees
        if pd.notna(row['Addressee(s)']):
            for addressee in _parse_character_list(row['Addressee(s)']):
                G.add_node(addressee)
        
        # Add "about" characters
        if pd.notna(row['About Character(s)']):
            for about in _parse_character_list(row['About Character(s)']):
                G.add_node(about)
    
    # Add edges based on edge_type
    G = _add_edges(G, all_dialogues, edge_type)
    
    return G


def _parse_character_list(raw_str: str) -> List[str]:
    """
    Parse a semicolon-separated character string.
    
    Args:
        raw_str: Raw string with semicolon-separated names
    
    Returns:
        List of stripped character names
    """
    return [char.strip() for char in str(raw_str).split(';')]


def _add_edges(
    G: nx.DiGraph,
    all_dialogues: pd.DataFrame,
    edge_type: str
) -> nx.DiGraph:
    """
    Add edges to graph based on dialogue data.
    
    Args:
        G: NetworkX DiGraph
        all_dialogues: DataFrame with dialogue data
        edge_type: 'direct' or 'indirect'
    
    Returns:
        Graph with edges added
    """
    for _, row in all_dialogues.iterrows():
        dialogue = row['Dialogue']
        speaker_raw = row['Speaker(s)']
        
        if edge_type == 'direct':
            addressee_raw = row['Addressee(s)']
            if pd.notna(speaker_raw) and pd.notna(addressee_raw):
                speakers = _parse_character_list(speaker_raw)
                addressees = _parse_character_list(addressee_raw)
                
                for speaker in speakers:
                    for addressee in addressees:
                        _add_or_update_edge(G, speaker, addressee, dialogue)
        
        elif edge_type == 'indirect':
            about_raw = row['About Character(s)']
            if pd.notna(speaker_raw) and pd.notna(about_raw):
                speakers = _parse_character_list(speaker_raw)
                abouts = _parse_character_list(about_raw)
                
                for speaker in speakers:
                    for about in abouts:
                        _add_or_update_edge(G, speaker, about, dialogue)
    
    # Normalize weights
    _normalize_edge_weights(G)
    
    return G


def _add_or_update_edge(
    G: nx.DiGraph,
    source: str,
    target: str,
    dialogue: str
) -> None:
    """
    Add a new edge or update existing edge with dialogue.
    
    Args:
        G: NetworkX DiGraph
        source: Source node
        target: Target node
        dialogue: Dialogue text
    """
    if G.has_edge(source, target):
        G[source][target]['weight'] += len(dialogue)
        G[source][target]['dialogues'].append(dialogue)
    else:
        G.add_edge(source, target, weight=len(dialogue), dialogues=[dialogue])


def _normalize_edge_weights(G: nx.DiGraph) -> None:
    """
    Normalize edge weights to [0, 1] range in-place.
    
    Args:
        G: NetworkX DiGraph
    """
    if G.number_of_edges() == 0:
        return
    
    max_weight = max(data['weight'] for _, _, data in G.edges(data=True))
    
    if max_weight > 0:
        for _, _, data in G.edges(data=True):
            data['normalized_weight'] = data['weight'] / max_weight


# =============================================================================
# SENTIMENT NETWORK CREATION
# =============================================================================

def create_sentiment_network(
    DG: nx.DiGraph,
    relationship_data: Dict,
    network_type: str = "undirected"
) -> nx.Graph:
    """
    Create a sentiment network from relationship data.
    
    Args:
        DG: Directed graph with dialogue data
        relationship_data: Dict of dicts with sentiment data
        network_type: 'undirected' or 'directed'
    
    Returns:
        Graph with sentiment edge attributes
    """
    # Build lookup dict for fast access
    pair_lookup = {}
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                pair_lookup[(speaker, addressee)] = item['adjusted_sentiment']
    
    # Create appropriate graph type
    G = nx.Graph() if network_type == "undirected" else nx.DiGraph()
    G.add_nodes_from(DG.nodes())
    
    if network_type == "undirected":
        processed_pairs = set()
        
        for u, v in DG.edges():
            pair_key = tuple(sorted([u, v]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            sentiment_u_to_v = pair_lookup.get((u, v), 0.0)
            sentiment_v_to_u = pair_lookup.get((v, u), 0.0)
            
            weight_u_to_v = DG[u][v].get('weight', 0) if DG.has_edge(u, v) else 0
            weight_v_to_u = DG[v][u].get('weight', 0) if DG.has_edge(v, u) else 0
            
            G.add_edge(
                u, v,
                sentiment_ab=sentiment_u_to_v,
                sentiment_ba=sentiment_v_to_u,
                weight=weight_u_to_v + weight_v_to_u
            )
    else:
        for u, v in DG.edges():
            sentiment = DG[u][v].get('sentiment_ab', 0.0)
            weight = DG[u][v].get('weight', 0)
            G.add_edge(u, v, sentiment=sentiment, weight=weight)
    
    return G


def create_graph_from_matrix(
    matrix: np.ndarray,
    row_characters: List[str],
    col_characters: List[str]
) -> nx.DiGraph:
    """
    Create a directed graph from a matrix representation.
    
    Args:
        matrix: 2D numpy array with edge weights
        row_characters: List of row character names
        col_characters: List of column character names
    
    Returns:
        NetworkX DiGraph with charisma node attributes
    """
    G = nx.DiGraph()
    
    for i, row_char in enumerate(row_characters):
        for j, col_char in enumerate(col_characters):
            if not np.isnan(matrix[i, j]):
                G.add_edge(row_char, col_char, weight=matrix[i, j])
    
    # Calculate charisma for each character using Bayesian average
    _calculate_charisma(G)
    
    return G


def _calculate_charisma(G: nx.DiGraph) -> None:
    """
    Calculate charisma (Bayesian average) for each node based on incoming sentiment.
    
    Args:
        G: NetworkX DiGraph with weight attributes
    """
    # Collect incoming sentiments
    character_incoming = {}
    all_sentiments = []
    
    for u, v, data in G.edges(data=True):
        sentiment = data.get('weight', 0.0)
        if v not in character_incoming:
            character_incoming[v] = []
        character_incoming[v].append(sentiment)
        all_sentiments.append(sentiment)
    
    if not all_sentiments:
        for node in G.nodes():
            G.nodes[node]['charisma'] = 0.0
        return
    
    # Calculate global statistics
    global_mean = sum(all_sentiments) / len(all_sentiments)
    num_characters = len(character_incoming) if character_incoming else 1
    C = len(all_sentiments) / num_characters
    
    # Calculate charisma for each character
    for node in G.nodes():
        if node in character_incoming:
            scores = character_incoming[node]
            n = len(scores)
            sum_s = sum(scores)
            charisma = (C * global_mean + sum_s) / (C + n) if (C + n) > 0 else 0.0
        else:
            charisma = 0.0
        G.nodes[node]['charisma'] = charisma


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

def print_graph_data(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Print and return basic graph statistics.
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        Dictionary with graph statistics
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'in_degree': {
            'min': min(in_degrees),
            'avg': np.mean(in_degrees),
            'max': max(in_degrees)
        },
        'out_degree': {
            'min': min(out_degrees),
            'avg': np.mean(out_degrees),
            'max': max(out_degrees)
        }
    }
    
    if G.number_of_edges() > 0:
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        dialogue_counts = [len(G[u][v]['dialogues']) for u, v in G.edges()]
        
        stats['weight'] = {
            'avg': np.mean(weights),
            'max': max(weights)
        }
        stats['dialogues'] = {
            'avg': np.mean(dialogue_counts),
            'max': max(dialogue_counts)
        }
    
    # Print formatted output
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}\n")
    
    print(f"Min in_degree: {stats['in_degree']['min']}")
    print(f"Average in_degree: {stats['in_degree']['avg']:.4f}")
    print(f"Max in_degree: {stats['in_degree']['max']}\n")
    
    print(f"Min out_degree: {stats['out_degree']['min']}")
    print(f"Average out_degree: {stats['out_degree']['avg']:.4f}")
    print(f"Max out_degree: {stats['out_degree']['max']}\n")
    
    if 'weight' in stats:
        print(f"Average weight: {stats['weight']['avg']:.2f}")
        print(f"Average number of dialogues: {stats['dialogues']['avg']:.2f}")
        print(f"Maximum number of dialogues: {stats['dialogues']['max']}")
    
    return stats


# =============================================================================
# POWER-LAW ANALYSIS
# =============================================================================

def fit_powerlaw(degree_sequence: List[int]) -> Tuple[float, float]:
    """
    Fit a power law distribution to a degree sequence.
    
    Args:
        degree_sequence: List of degree values
    
    Returns:
        Tuple of (alpha exponent, xmin value)
    """
    degree_fit = powerlaw.Fit(degree_sequence, discrete=True, xmin=1, verbose=False)
    return degree_fit.alpha, degree_fit.xmin


# =============================================================================
# ASSORTATIVITY ANALYSIS
# =============================================================================

def assortativity_analysis(G: nx.DiGraph) -> float:
    """
    Compute and print degree assortativity coefficient.
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        Assortativity coefficient
    """
    r = nx.degree_assortativity_coefficient(G, weight="weight")
    print(f"Degree Assortativity Coefficient (r): {r:.4f}")
    
    # Provide interpretation
    if r > 0.1:
        print("  → Assortative network: high-degree nodes connect to high-degree nodes")
    elif r < -0.1:
        print("  → Disassortative network: high-degree nodes connect to low-degree nodes")
    else:
        print("  → Neutral assortativity")
    
    return r


# =============================================================================
# WEIGHT NORMALIZATION
# =============================================================================

def get_normalized_weights(G: nx.Graph) -> List[float]:
    """
    Get normalized weights for all edges and update graph.
    
    Args:
        G: NetworkX Graph
    
    Returns:
        List of normalized weight values
    """
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if not weights:
        return []
    
    max_weight = max(weights)
    
    if max_weight > 0:
        for _, _, data in G.edges(data=True):
            data['normalized_weight'] = data['weight'] / max_weight
    
    return weights


# =============================================================================
# DEGREE EXTRACTION
# =============================================================================

def get_degree_sequences(G: nx.DiGraph) -> Tuple[List[int], List[int], List[int]]:
    """
    Extract degree sequences from a directed graph.
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        Tuple of (total_degrees, in_degrees, out_degrees)
    """
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    degrees = [d for _, d in G.degree()]
    
    return degrees, in_degrees, out_degrees


def get_nonzero_degrees(
    G: nx.DiGraph
) -> Tuple[List[int], List[int], List[int]]:
    """
    Get degree sequences with zero values filtered out (for log-log plots).
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        Tuple of filtered (total_degrees, in_degrees, out_degrees)
    """
    degrees, in_degrees, out_degrees = get_degree_sequences(G)
    
    return (
        [d for d in degrees if d > 0],
        [d for d in in_degrees if d > 0],
        [d for d in out_degrees if d > 0]
    )
