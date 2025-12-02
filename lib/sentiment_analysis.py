"""
Sentiment analysis functions for ASoIaF Network Analysis.
Provides VADER-based sentiment scoring and relationship analysis.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Any, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from lib.config import PLOT_CONFIG, FIGURE_SIZES, PALETTE
from lib.style import Figure, style_axis, get_output_path


# =============================================================================
# SENTIMENT SCORING
# =============================================================================

def _ensure_vader_lexicon() -> None:
    """Ensure VADER lexicon is downloaded."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)


def _get_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """
    Get compound sentiment score for text using VADER.
    
    Args:
        text: Text to analyze
        analyzer: VADER analyzer instance
    
    Returns:
        Compound sentiment score (-1 to 1)
    """
    try:
        if text is None or not str(text).strip():
            return 0.0
        return analyzer.polarity_scores(str(text))['compound']
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0.0


# =============================================================================
# HEATMAP VISUALIZATION
# =============================================================================

def plot_heatmap(
    matrix: np.ndarray,
    all_characters: List[str],
    remove_empty: bool = False,
    title: str = "Sentiment Heatmap",
    file_name: str = "heatmap",
    subdirectory: str = "sentiment",
    show: bool = True
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Plot a styled sentiment heatmap.
    
    Args:
        matrix: 2D numpy array of sentiment values
        all_characters: List of character names
        remove_empty: Whether to remove empty rows/columns
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        Tuple of (processed_matrix, x_characters, y_characters)
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    x_characters = all_characters.copy()
    y_characters = all_characters.copy()
    
    if remove_empty:
        non_empty_rows = ~np.isnan(matrix).all(axis=1)
        non_empty_cols = ~np.isnan(matrix).all(axis=0)
        
        matrix = matrix[non_empty_rows, :][:, non_empty_cols]
        y_characters = [char for i, char in enumerate(all_characters) if non_empty_rows[i]]
        x_characters = [char for i, char in enumerate(all_characters) if non_empty_cols[i]]
    
    masked_matrix = np.ma.masked_invalid(matrix)
    min_val = np.min(masked_matrix) if masked_matrix.count() > 0 else -1
    max_val = np.max(masked_matrix) if masked_matrix.count() > 0 else 1
    
    # Use symmetric limits for diverging colormap
    abs_max = max(abs(min_val), abs(max_val))
    
    with Figure(file_name, FIGURE_SIZES.HEATMAP, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        # Create heatmap
        heatmap = sns.heatmap(
            matrix,
            annot=False,
            cmap='RdYlGn',
            center=0,
            vmin=-abs_max,
            vmax=abs_max,
            cbar_kws={
                'label': 'Adjusted Sentiment Score (Bayesian)',
                'shrink': 0.8
            },
            xticklabels=x_characters,
            yticklabels=y_characters,
            mask=np.isnan(matrix),
            ax=ax,
            linewidths=0.5,
            linecolor=PALETTE.SNOW_WHITE
        )
        
        # Style
        plt.title(title, fontsize=PLOT_CONFIG.title_font_size,
                 fontweight='bold', pad=20, color=PLOT_CONFIG.title_color)
        plt.xlabel('Addressee(s)', fontsize=PLOT_CONFIG.label_font_size,
                  fontweight='bold')
        plt.ylabel('Speaker(s)', fontsize=PLOT_CONFIG.label_font_size,
                  fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        # Style colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=PLOT_CONFIG.tick_font_size)
    
    return matrix, x_characters, y_characters


# =============================================================================
# MAIN SENTIMENT ANALYSIS
# =============================================================================

def sentiment_heatmap(
    G: nx.DiGraph,
    remove_empty: bool = False,
    title: str = "Character Relationship Sentiment Heatmap",
    file_name: str = "sentiment-heatmap",
    subdirectory: str = "sentiment",
    show: bool = True
) -> Tuple[nx.DiGraph, List[Dict], Dict]:
    """
    Analyze sentiment in dialogues and create a heatmap visualization.
    
    Uses Bayesian Average (damped mean) to handle varying dialogue counts.
    
    Args:
        G: NetworkX DiGraph with dialogue data
        remove_empty: Whether to remove empty rows/columns from heatmap
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        Tuple of (updated_graph, dialogue_data, relationship_data)
    """
    _ensure_vader_lexicon()
    
    print("Extracting dialogues from graph edges...")
    
    # Extract dialogues
    dialogues_list = []
    for speaker, addressee, data in G.edges(data=True):
        for dialogue in data.get('dialogues', []):
            dialogues_list.append((speaker, addressee, dialogue))
    
    print(f"Extracted {len(dialogues_list)} dialogues from graph")
    
    # Analyze sentiment
    print("Analyzing sentiment for all dialogues...")
    sid = SentimentIntensityAnalyzer()
    
    dialogue_scores = []
    all_scores = []
    
    for speaker, addressee, dialogue in dialogues_list:
        score = _get_sentiment(dialogue, sid)
        dialogue_scores.append((speaker, addressee, score))
        all_scores.append(score)
    
    # Calculate global statistics for Bayesian Average
    global_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"Global average sentiment score (m): {global_mean:.4f}")
    
    # Group dialogues by character pair
    pair_data = {}
    for speaker, addressee, score in dialogue_scores:
        pair_key = (speaker, addressee)
        if pair_key not in pair_data:
            pair_data[pair_key] = []
        pair_data[pair_key].append(score)
    
    # Calculate confidence parameter
    total_dialogues = len(dialogue_scores)
    num_pairs = len(pair_data)
    C = total_dialogues / num_pairs if num_pairs > 0 else 1.0
    print(f"Confidence parameter (C): {C:.2f} (avg dialogues per pair)")
    
    # Apply Bayesian Average
    adjusted_scores = {}
    raw_scores = {}
    dialogue_counts = {}
    
    for pair_key, scores in pair_data.items():
        n = len(scores)
        sum_s = sum(scores)
        raw_mean = sum_s / n if n > 0 else 0.0
        
        # Bayesian Average: S_adj = (C * m + Σs) / (C + n)
        s_adj = (C * global_mean + sum_s) / (C + n)
        
        adjusted_scores[pair_key] = s_adj
        raw_scores[pair_key] = raw_mean
        dialogue_counts[pair_key] = n
    
    # Create matrix for heatmap
    all_characters = sorted(G.nodes())
    matrix = np.full((len(all_characters), len(all_characters)), np.nan)
    char_idx = {c: i for i, c in enumerate(all_characters)}
    
    for (speaker, addressee), score in adjusted_scores.items():
        i = char_idx[speaker]
        j = char_idx[addressee]
        matrix[i, j] = score
    
    # Plot heatmap
    plot_heatmap(matrix, all_characters, remove_empty, title, file_name, subdirectory, show)
    
    # Prepare return data
    dialogue_data = [
        {'speaker': speaker, 'addressee': addressee, 'score': score}
        for speaker, addressee, score in dialogue_scores
    ]
    
    # Create relationship data structure
    relationship_data = {}
    for speaker in all_characters:
        relationship_data[speaker] = {}
        for addressee in all_characters:
            pair = (speaker, addressee)
            if pair in pair_data:
                relationship_data[speaker][addressee] = {
                    'speaker': speaker,
                    'addressee': addressee,
                    'raw_sentiment': raw_scores[pair],
                    'adjusted_sentiment': adjusted_scores[pair],
                    'dialogue_count': dialogue_counts[pair]
                }
            else:
                relationship_data[speaker][addressee] = None
    
    # Update graph with sentiment data
    for u, v in G.edges():
        pair = (u, v)
        if pair in raw_scores:
            G[u][v]['raw_sentiment'] = raw_scores[pair]
            G[u][v]['adjusted_sentiment'] = adjusted_scores[pair]
            G[u][v]['dialogue_count'] = dialogue_counts[pair]
    
    return G, dialogue_data, relationship_data


# =============================================================================
# RELATIONSHIP ANALYSIS
# =============================================================================

def top_relationships(
    relationship_data: Dict,
    n: int = 10
) -> Dict[str, List]:
    """
    Analyze and display top relationships from sentiment data.
    
    Args:
        relationship_data: Dict of dicts with relationship sentiment data
        n: Number of top relationships to show
    
    Returns:
        Dictionary with top positive/negative/asymmetric relationships
    """
    # Flatten relationships
    all_relationships = []
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                all_relationships.append(item)
    
    # Build lookup for reverse relationships
    pair_lookup = {(rel['speaker'], rel['addressee']): rel for rel in all_relationships}
    
    def _print_section(title: str, relationships: List, is_bidirectional: bool = False):
        """Print a formatted section of relationships."""
        print(f"\n{'='*70}")
        print(f"{title}")
        print('='*70)
        
        for i, rel in enumerate(relationships, 1):
            if is_bidirectional:
                print(f"{i}. {rel['char_a']} ↔ {rel['char_b']}: "
                      f"mean={rel['mean_sentiment']:.4f} "
                      f"({rel['char_a']}→{rel['char_b']}: {rel['sentiment_a_to_b']:.4f}, "
                      f"{rel['char_b']}→{rel['char_a']}: {rel['sentiment_b_to_a']:.4f}) "
                      f"({int(rel['total_dialogues'])} total dialogues)")
            else:
                print(f"{i}. {rel['speaker']} → {rel['addressee']}: "
                      f"{rel['adjusted_sentiment']:.4f} "
                      f"({int(rel['dialogue_count'])} dialogues)")
    
    # Top positive (1-way)
    top_positive_1way = sorted(all_relationships, 
                               key=lambda x: x['adjusted_sentiment'], 
                               reverse=True)[:n]
    _print_section(f"Top {n} Positive Relationships (1-way):", top_positive_1way)
    
    # Top negative (1-way)
    top_negative_1way = sorted(all_relationships, 
                               key=lambda x: x['adjusted_sentiment'])[:n]
    _print_section(f"Top {n} Negative Relationships (1-way):", top_negative_1way)
    
    # Find bidirectional pairs
    bidirectional_pairs = []
    seen_pairs = set()
    
    for rel in all_relationships:
        speaker, addressee = rel['speaker'], rel['addressee']
        reverse_key = (addressee, speaker)
        pair_key = tuple(sorted([speaker, addressee]))
        
        if reverse_key in pair_lookup and pair_key not in seen_pairs:
            reverse_rel = pair_lookup[reverse_key]
            bidirectional_pairs.append({
                'char_a': speaker,
                'char_b': addressee,
                'sentiment_a_to_b': rel['adjusted_sentiment'],
                'sentiment_b_to_a': reverse_rel['adjusted_sentiment'],
                'mean_sentiment': (rel['adjusted_sentiment'] + reverse_rel['adjusted_sentiment']) / 2,
                'asymmetry': abs(rel['adjusted_sentiment'] - reverse_rel['adjusted_sentiment']),
                'total_dialogues': rel['dialogue_count'] + reverse_rel['dialogue_count']
            })
            seen_pairs.add(pair_key)
    
    # Top positive (2-way)
    top_positive_2way = sorted(bidirectional_pairs, 
                               key=lambda x: x['mean_sentiment'], 
                               reverse=True)[:n]
    _print_section(f"Top {n} Positive Relationships (2-way, mean):", 
                  top_positive_2way, is_bidirectional=True)
    
    # Top negative (2-way)
    top_negative_2way = sorted(bidirectional_pairs, 
                               key=lambda x: x['mean_sentiment'])[:n]
    _print_section(f"Top {n} Negative Relationships (2-way, mean):", 
                  top_negative_2way, is_bidirectional=True)
    
    # Top asymmetric
    top_asymmetric = sorted(bidirectional_pairs, 
                            key=lambda x: x['asymmetry'], 
                            reverse=True)[:n]
    _print_section(f"Top {n} Asymmetric Relationships (|A→B - B→A|):", 
                  top_asymmetric, is_bidirectional=True)
    
    return {
        'top_positive_1way': top_positive_1way,
        'top_negative_1way': top_negative_1way,
        'top_positive_2way': top_positive_2way,
        'top_negative_2way': top_negative_2way,
        'top_asymmetric': top_asymmetric
    }


# =============================================================================
# CHARACTER CHARISMA ANALYSIS
# =============================================================================

def most_loved_hated(
    relationship_data: Dict,
    n: int = 10
) -> Tuple[List[Dict], List[Dict]]:
    """
    Find the most loved and most hated characters based on incoming sentiment.
    
    Uses Bayesian average across all incoming edges for each character.
    
    Args:
        relationship_data: Dict of dicts with relationship sentiment data
        n: Number of top/bottom characters to return
    
    Returns:
        Tuple of (most_loved, most_hated) character lists with charisma scores
    """
    # Extract all relationships
    all_relationships = []
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                all_relationships.append(item)
    
    # Group incoming sentiments by character
    character_incoming = {}
    all_sentiments = []
    total_dialogues = 0
    
    for rel in all_relationships:
        character = rel['addressee']
        sentiment = rel['adjusted_sentiment']
        dialogue_count = rel['dialogue_count']
        
        if character not in character_incoming:
            character_incoming[character] = []
        
        character_incoming[character].append((sentiment, dialogue_count))
        all_sentiments.append(sentiment)
        total_dialogues += dialogue_count
    
    # Calculate global statistics
    global_mean = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
    num_characters = len(character_incoming)
    C = total_dialogues / num_characters if num_characters > 0 else 1.0
    
    # Calculate charisma for each character
    character_charisma = {}
    
    for character, incoming_data in character_incoming.items():
        weighted_sum = sum(sentiment * count for sentiment, count in incoming_data)
        total_count = sum(count for _, count in incoming_data)
        
        charisma = (C * global_mean + weighted_sum) / (C + total_count) if (C + total_count) > 0 else 0.0
        
        character_charisma[character] = {
            'character': character,
            'charisma': charisma,
            'incoming_dialogues': total_count,
            'incoming_edges': len(incoming_data)
        }
    
    # Sort and return
    charisma_list = list(character_charisma.values())
    most_loved = sorted(charisma_list, key=lambda x: x['charisma'], reverse=True)[:n]
    most_hated = sorted(charisma_list, key=lambda x: x['charisma'])[:n]
    
    return most_loved, most_hated


def print_charisma_rankings(
    most_loved: List[Dict],
    most_hated: List[Dict]
) -> None:
    """
    Print formatted charisma rankings.
    
    Args:
        most_loved: List of most loved characters
        most_hated: List of most hated characters
    """
    print("=" * 60)
    print("MOST LOVED CHARACTERS (Highest Charisma)")
    print("=" * 60)
    for i, char in enumerate(most_loved, 1):
        print(f"{i:2d}. {char['character']:<25} "
              f"charisma: {char['charisma']:+.4f} "
              f"({char['incoming_dialogues']} incoming dialogues)")
    
    print("\n" + "=" * 60)
    print("MOST HATED CHARACTERS (Lowest Charisma)")
    print("=" * 60)
    for i, char in enumerate(most_hated, 1):
        print(f"{i:2d}. {char['character']:<25} "
              f"charisma: {char['charisma']:+.4f} "
              f"({char['incoming_dialogues']} incoming dialogues)")
