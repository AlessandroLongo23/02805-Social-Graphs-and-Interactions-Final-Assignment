"""
Extended analysis functions for ASoIaF Network Analysis.
Contains additional visualizations and analyses beyond the core functionality.

NOTE: These are optional additional analyses that could enhance the project.
Review and use as needed.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import Counter

from lib.config import PLOT_CONFIG, FIGURE_SIZES, PALETTE
from lib.style import Figure, style_axis, add_bar_labels


# =============================================================================
# CHARACTER IMPORTANCE COMPARISON
# =============================================================================

def plot_character_rankings_comparison(
    G: nx.DiGraph,
    top_n: int = 15,
    file_name: str = "character-rankings-comparison",
    subdirectory: str = "basic",
    show: bool = True
) -> None:
    """
    Compare character rankings across different metrics.
    
    Shows a bump chart/parallel coordinates comparing how characters rank
    across degree, betweenness, eigenvector centrality, and weighted degree.
    
    Args:
        G: NetworkX graph
        top_n: Number of top characters to include
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    # Calculate all metrics
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight='weight'))
    betweenness = nx.betweenness_centrality(G)
    
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector = {n: 0 for n in G.nodes()}
    
    # Get top characters by weighted degree
    top_chars = sorted(weighted_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_char_names = [char for char, _ in top_chars]
    
    # Create rankings for each metric
    metrics = {
        'Degree': degree,
        'Weighted\nDegree': weighted_degree,
        'Betweenness': betweenness,
        'Eigenvector': eigenvector
    }
    
    rankings = {}
    for metric_name, metric_values in metrics.items():
        sorted_chars = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        rankings[metric_name] = {char: rank + 1 for rank, (char, _) in enumerate(sorted_chars)}
    
    with Figure(file_name, (14, 10), subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        metric_names = list(metrics.keys())
        x_positions = range(len(metric_names))
        
        # Get colors for characters
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_char_names)))
        
        for i, char in enumerate(top_char_names):
            char_rankings = [rankings[m].get(char, top_n + 1) for m in metric_names]
            ax.plot(x_positions, char_rankings, 'o-', 
                   color=colors[i], linewidth=2, markersize=8,
                   label=char, alpha=0.8)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylabel('Rank (1 = highest)', fontsize=12)
        ax.set_ylim(top_n + 1, 0)  # Invert y-axis (rank 1 at top)
        ax.set_xlim(-0.3, len(metric_names) - 0.7)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=9, framealpha=0.9)
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.title(f"Character Ranking Comparison (Top {top_n})",
                 fontsize=PLOT_CONFIG.title_font_size, fontweight='bold',
                 pad=20, color=PLOT_CONFIG.title_color)
        plt.tight_layout()


# =============================================================================
# SENTIMENT DISTRIBUTION ANALYSIS
# =============================================================================

def plot_sentiment_distribution(
    relationship_data: Dict,
    file_name: str = "sentiment-distribution",
    subdirectory: str = "sentiment",
    show: bool = True
) -> None:
    """
    Plot the distribution of sentiment scores across all relationships.
    
    Args:
        relationship_data: Dict of dicts with relationship sentiment data
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    # Extract all sentiment scores
    all_sentiments = []
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                all_sentiments.append(item['adjusted_sentiment'])
    
    with Figure(file_name, FIGURE_SIZES.MEDIUM, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        # Create histogram with KDE
        ax.hist(all_sentiments, bins=50, density=True, alpha=0.7,
               color=PALETTE.WINTER_BLUE, edgecolor=PALETTE.NIGHT_BLACK,
               linewidth=0.5, label='Distribution')
        
        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(all_sentiments)
        x_range = np.linspace(min(all_sentiments), max(all_sentiments), 200)
        ax.plot(x_range, kde(x_range), color=PALETTE.DRAGON_RED, 
               linewidth=2.5, label='KDE')
        
        # Add vertical lines for mean and median
        mean_val = np.mean(all_sentiments)
        median_val = np.median(all_sentiments)
        
        ax.axvline(mean_val, color=PALETTE.GOLD, linestyle='--',
                  linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color=PALETTE.LOVE_GREEN, linestyle=':',
                  linewidth=2, label=f'Median: {median_val:.4f}')
        ax.axvline(0, color=PALETTE.NEUTRAL_GRAY, linestyle='-',
                  linewidth=1, alpha=0.5)
        
        style_axis(ax, 
                  title="Distribution of Sentiment Scores",
                  xlabel="Adjusted Sentiment Score (Bayesian)",
                  ylabel="Density",
                  grid_axis='y')
        
        ax.legend(fontsize=10, framealpha=0.9)
        
        # Add statistics annotation
        stats_text = (f"N = {len(all_sentiments)}\n"
                     f"σ = {np.std(all_sentiments):.4f}\n"
                     f"Min = {min(all_sentiments):.4f}\n"
                     f"Max = {max(all_sentiments):.4f}")
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# =============================================================================
# COMMUNITY SENTIMENT ANALYSIS
# =============================================================================

def plot_community_sentiment_comparison(
    G: nx.Graph,
    partition: Dict[str, int],
    relationship_data: Dict,
    file_name: str = "community-sentiment-comparison",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Compare average sentiment within and between communities.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        relationship_data: Dict of dicts with relationship sentiment data
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    communities = sorted(set(partition.values()))
    
    # Calculate internal and external sentiment for each community
    community_stats = {}
    
    for comm_id in communities:
        members = [node for node, c in partition.items() if c == comm_id]
        
        internal_sentiments = []
        external_sentiments = []
        
        for speaker in members:
            if speaker not in relationship_data:
                continue
            for addressee, item in relationship_data[speaker].items():
                if item is None:
                    continue
                
                sentiment = item['adjusted_sentiment']
                
                if addressee in members:
                    internal_sentiments.append(sentiment)
                else:
                    external_sentiments.append(sentiment)
        
        community_stats[comm_id] = {
            'internal_mean': np.mean(internal_sentiments) if internal_sentiments else 0,
            'internal_std': np.std(internal_sentiments) if internal_sentiments else 0,
            'external_mean': np.mean(external_sentiments) if external_sentiments else 0,
            'external_std': np.std(external_sentiments) if external_sentiments else 0,
            'n_internal': len(internal_sentiments),
            'n_external': len(external_sentiments)
        }
    
    with Figure(file_name, FIGURE_SIZES.WIDE, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        x = np.arange(len(communities))
        width = 0.35
        
        internal_means = [community_stats[c]['internal_mean'] for c in communities]
        external_means = [community_stats[c]['external_mean'] for c in communities]
        
        bars1 = ax.bar(x - width/2, internal_means, width, 
                      label='Internal (within community)',
                      color=PALETTE.LOVE_GREEN, edgecolor=PALETTE.NIGHT_BLACK,
                      linewidth=0.5, alpha=0.8)
        bars2 = ax.bar(x + width/2, external_means, width,
                      label='External (to other communities)',
                      color=PALETTE.WINTER_BLUE, edgecolor=PALETTE.NIGHT_BLACK,
                      linewidth=0.5, alpha=0.8)
        
        ax.axhline(y=0, color=PALETTE.NEUTRAL_GRAY, linestyle='-', linewidth=1)
        
        ax.set_xlabel('Community', fontsize=12)
        ax.set_ylabel('Average Sentiment', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in communities])
        ax.legend(fontsize=10)
        
        style_axis(ax, title="Average Sentiment: Internal vs External Communication",
                  grid_axis='y')


# =============================================================================
# DIALOGUE VOLUME HEATMAP
# =============================================================================

def plot_dialogue_volume_heatmap(
    G: nx.DiGraph,
    top_n: int = 20,
    file_name: str = "dialogue-volume-heatmap",
    subdirectory: str = "basic",
    show: bool = True
) -> None:
    """
    Create a heatmap showing dialogue volumes between top characters.
    
    Args:
        G: NetworkX graph with dialogue data
        top_n: Number of top characters to include
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    # Get top characters by degree
    degrees = dict(G.degree())
    top_chars = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_char_names = [char for char, _ in top_chars]
    
    # Create dialogue count matrix
    matrix = np.zeros((len(top_char_names), len(top_char_names)))
    
    for i, speaker in enumerate(top_char_names):
        for j, addressee in enumerate(top_char_names):
            if G.has_edge(speaker, addressee):
                matrix[i, j] = len(G[speaker][addressee].get('dialogues', []))
    
    with Figure(file_name, FIGURE_SIZES.HEATMAP, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        sns.heatmap(matrix, annot=True, fmt='.0f',
                   cmap='YlOrRd', xticklabels=top_char_names,
                   yticklabels=top_char_names,
                   cbar_kws={'label': 'Number of Dialogues'},
                   ax=ax, linewidths=0.5, linecolor=PALETTE.SNOW_WHITE)
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.title(f"Dialogue Volume Between Top {top_n} Characters",
                 fontsize=PLOT_CONFIG.title_font_size, fontweight='bold',
                 pad=20, color=PLOT_CONFIG.title_color)
        plt.xlabel('Addressee', fontsize=PLOT_CONFIG.label_font_size, fontweight='bold')
        plt.ylabel('Speaker', fontsize=PLOT_CONFIG.label_font_size, fontweight='bold')


# =============================================================================
# POV CHARACTER ANALYSIS
# =============================================================================

def analyze_pov_characters(
    G: nx.DiGraph,
    pov_characters: List[str] = None,
    file_name: str = "pov-character-analysis",
    subdirectory: str = "basic",
    show: bool = True
) -> None:
    """
    Analyze the network from POV characters' perspectives.
    
    Shows who POV characters interact with most and sentiment patterns.
    
    Args:
        G: NetworkX graph
        pov_characters: List of POV character names (uses common ones if None)
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    if pov_characters is None:
        # Default POV characters from AGOT
        pov_characters = [
            'Eddard Stark', 'Catelyn Stark', 'Jon Snow', 'Tyrion Lannister',
            'Daenerys Targaryen', 'Sansa Stark', 'Arya Stark', 'Bran Stark'
        ]
    
    # Filter to characters that exist in the graph
    pov_characters = [c for c in pov_characters if c in G.nodes()]
    
    if not pov_characters:
        print("No POV characters found in graph.")
        return
    
    with Figure(file_name, (16, 4 * len(pov_characters)), subdirectory, show=show) as fig:
        for i, pov in enumerate(pov_characters):
            ax = fig.add_subplot(len(pov_characters), 1, i + 1)
            
            # Get all interactions for this POV
            interactions = []
            for neighbor in G.neighbors(pov):
                if G.has_edge(pov, neighbor):
                    dialogues = G[pov][neighbor].get('dialogues', [])
                    interactions.append((neighbor, len(dialogues)))
            
            # Also get incoming interactions
            for pred in G.predecessors(pov):
                if G.has_edge(pred, pov) and pred not in [n for n, _ in interactions]:
                    dialogues = G[pred][pov].get('dialogues', [])
                    interactions.append((pred, len(dialogues)))
            
            # Sort and take top 10
            interactions = sorted(interactions, key=lambda x: x[1], reverse=True)[:10]
            
            if not interactions:
                ax.text(0.5, 0.5, f'{pov}\n(No interactions)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            names = [name for name, _ in interactions][::-1]
            counts = [count for _, count in interactions][::-1]
            
            colors = [PALETTE.DRAGON_RED if c > np.median(counts) else PALETTE.WINTER_BLUE 
                     for c in counts]
            
            bars = ax.barh(names, counts, color=colors, 
                          edgecolor=PALETTE.NIGHT_BLACK, alpha=0.8, linewidth=0.5)
            
            ax.set_title(f'{pov}', fontsize=12, fontweight='bold',
                        color=PLOT_CONFIG.title_color, loc='left')
            ax.set_xlabel('Number of Dialogues', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        plt.suptitle("POV Character Interaction Patterns",
                    fontsize=PLOT_CONFIG.title_font_size, fontweight='bold',
                    y=1.02, color=PLOT_CONFIG.title_color)


# =============================================================================
# NETWORK EVOLUTION (if temporal data available)
# =============================================================================

def plot_degree_vs_sentiment(
    G: nx.DiGraph,
    relationship_data: Dict,
    file_name: str = "degree-vs-sentiment",
    subdirectory: str = "sentiment",
    show: bool = True
) -> None:
    """
    Create a scatter plot showing the relationship between degree and sentiment.
    
    Args:
        G: NetworkX graph
        relationship_data: Dict of dicts with relationship sentiment data
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    # Calculate average incoming sentiment for each character
    char_data = {}
    
    degrees = dict(G.degree())
    
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is None:
                continue
            
            if addressee not in char_data:
                char_data[addressee] = {'sentiments': [], 'degree': degrees.get(addressee, 0)}
            char_data[addressee]['sentiments'].append(item['adjusted_sentiment'])
    
    # Calculate mean sentiment for each character
    plot_data = []
    for char, data in char_data.items():
        if data['sentiments']:
            plot_data.append({
                'character': char,
                'degree': data['degree'],
                'avg_sentiment': np.mean(data['sentiments']),
                'n_incoming': len(data['sentiments'])
            })
    
    with Figure(file_name, FIGURE_SIZES.MEDIUM, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        degrees_plot = [d['degree'] for d in plot_data]
        sentiments_plot = [d['avg_sentiment'] for d in plot_data]
        sizes = [d['n_incoming'] * 10 + 50 for d in plot_data]
        
        # Color by sentiment
        colors = [PALETTE.LOVE_GREEN if s > 0 else PALETTE.HATE_RED for s in sentiments_plot]
        
        scatter = ax.scatter(degrees_plot, sentiments_plot, s=sizes, c=colors,
                            alpha=0.7, edgecolors=PALETTE.NIGHT_BLACK, linewidth=0.5)
        
        # Add labels for notable characters
        threshold_degree = np.percentile(degrees_plot, 90)
        for d in plot_data:
            if d['degree'] > threshold_degree:
                ax.annotate(d['character'], 
                           (d['degree'], d['avg_sentiment']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.axhline(y=0, color=PALETTE.NEUTRAL_GRAY, linestyle='--', linewidth=1, alpha=0.5)
        
        style_axis(ax,
                  title="Character Degree vs. Incoming Sentiment",
                  xlabel="Degree (number of connections)",
                  ylabel="Average Incoming Sentiment")
        
        # Add note about size
        ax.text(0.98, 0.02, "Point size = number of incoming edges",
               transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
               style='italic', alpha=0.7)

