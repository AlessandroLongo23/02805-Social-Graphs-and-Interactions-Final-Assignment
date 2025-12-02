"""
Indirect (about) dialogue analysis for ASoIaF Network Analysis.
Provides utilities for analyzing dialogues about characters.
"""
import networkx as nx
import numpy as np
from typing import List

from lib.utils import _calculate_charisma


def create_graph_from_matrix(
    matrix: np.ndarray,
    row_characters: List[str],
    col_characters: List[str]
) -> nx.DiGraph:
    """
    Create a directed graph from a sentiment difference matrix.
    
    Args:
        matrix: 2D numpy array with edge weights (sentiment differences)
        row_characters: List of row character names (speakers)
        col_characters: List of column character names (addressees)
    
    Returns:
        NetworkX DiGraph with charisma node attributes
    """
    G = nx.DiGraph()
    
    for i, row_char in enumerate(row_characters):
        for j, col_char in enumerate(col_characters):
            if not np.isnan(matrix[i, j]):
                G.add_edge(row_char, col_char, weight=matrix[i, j])
    
    # Calculate charisma for each character
    _calculate_charisma(G)
    
    return G


def compute_sentiment_difference_matrix(
    direct_relationship_data: dict,
    about_relationship_data: dict
) -> tuple:
    """
    Compute the element-wise difference between direct and about sentiment matrices.
    
    Args:
        direct_relationship_data: Relationship data from direct dialogues
        about_relationship_data: Relationship data from "about" dialogues
    
    Returns:
        Tuple of (difference_matrix, all_characters)
    """
    difference_matrix = []
    all_characters = []
    
    for speaker in direct_relationship_data:
        difference_matrix.append([])
        all_characters.append(speaker)
        
        for target in direct_relationship_data[speaker]:
            # Safely get values
            about_item = about_relationship_data.get(speaker, {}).get(target)
            direct_item = direct_relationship_data[speaker][target]
            
            a = about_item['adjusted_sentiment'] if about_item is not None else None
            b = direct_item['adjusted_sentiment'] if direct_item is not None else None
            
            difference_matrix[-1].append(np.nan if a is None or b is None else a - b)
    
    return np.array(difference_matrix), all_characters
