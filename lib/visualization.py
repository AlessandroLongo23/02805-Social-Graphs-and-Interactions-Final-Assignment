"""
Visualization functions for ASoIaF Network Analysis.
Provides consistent, beautiful network and statistical visualizations.
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Optional, Any

from lib.config import (
    PLOT_CONFIG, FIGURE_SIZES, NETWORK_CONFIG, PALETTE,
    IceAndFirePalette
)
from lib.style import (
    Figure, style_axis, add_title, get_output_path,
    calculate_node_sizes, normalize_weights, add_colorbar,
    style_histogram, add_bar_labels, get_bidirectional_sentiment_color,
    create_sentiment_legend
)
from lib.utils import (
    fit_powerlaw, create_sentiment_network, get_normalized_weights,
    get_degree_sequences, get_nonzero_degrees
)


# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================

def draw_network(
    G: nx.DiGraph,
    node_size_criterion: str = 'degree',
    color_parameter: str = 'degree',
    title: str = "ASoIaF: Character Dialogue Network",
    file_name: str = "dialogue-network",
    subdirectory: str = "basic",
    show: bool = True
) -> None:
    """
    Draw a network visualization with consistent styling.
    
    Args:
        G: NetworkX graph
        node_size_criterion: Criterion for node sizing
            ('degree', 'in_degree', 'out_degree', 'weighted_in_degree', 'weighted_out_degree')
        color_parameter: Parameter for node coloring ('degree', 'sentiment')
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    if G is None or len(G.nodes) == 0:
        print("Graph is empty or file not found.")
        return
    
    with Figure(file_name, FIGURE_SIZES.NETWORK, subdirectory, show=show) as fig:
        # Calculate node sizes
        node_size_dict = calculate_node_sizes(G, node_size_criterion)
        node_sizes = [node_size_dict[n] for n in G.nodes()]
        
        # Calculate node colors
        if color_parameter == 'degree':
            if 'in' in node_size_criterion:
                values = dict(G.in_degree(weight='normalized_weight' if 'weighted' in node_size_criterion else None))
            elif 'out' in node_size_criterion:
                values = dict(G.out_degree(weight='normalized_weight' if 'weighted' in node_size_criterion else None))
            else:
                values = dict(G.degree())
            node_colors = [values.get(n, 0) for n in G.nodes()]
            cmap = plt.cm.YlOrRd
        elif color_parameter == 'sentiment':
            node_colors = [G.nodes[n].get('charisma', 0) for n in G.nodes()]
            cmap = plt.cm.RdYlGn
        else:
            node_colors = [PALETTE.STEEL_GRAY for _ in G.nodes()]
            cmap = None
        
        # Get edge widths
        edge_widths = [data.get('normalized_weight', 0.5) * 8 + 0.5 
                       for _, _, data in G.edges(data=True)]
        
        # Layout
        pos = nx.forceatlas2_layout(
            G,
            max_iter=NETWORK_CONFIG.layout_iterations,
            scaling_ratio=NETWORK_CONFIG.layout_scaling,
            node_mass=node_size_dict,
            node_size=node_size_dict,
            dissuade_hubs=True,
            weight='weight'
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            arrows=True,
            arrowstyle=NETWORK_CONFIG.arrow_style,
            arrowsize=NETWORK_CONFIG.arrow_size,
            edge_color=PALETTE.IRON_GRAY,
            alpha=PLOT_CONFIG.edge_alpha,
            node_size=node_sizes,
            connectionstyle=NETWORK_CONFIG.connection_style
        )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=PLOT_CONFIG.node_alpha,
            edgecolors=PLOT_CONFIG.node_edge_color,
            linewidths=PLOT_CONFIG.node_edge_width
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=NETWORK_CONFIG.label_font_size,
            font_family='serif',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=PLOT_CONFIG.label_alpha,
                pad=NETWORK_CONFIG.label_pad
            )
        )
        
        # Add title and colorbar
        plt.title(title, fontsize=PLOT_CONFIG.title_font_size, 
                 fontweight='bold', pad=20, color=PLOT_CONFIG.title_color)
        
        if cmap is not None and nodes is not None:
            label = 'Degree' if color_parameter == 'degree' else 'Charisma'
            add_colorbar(nodes, label=label)
        
        plt.axis('off')


def draw_sentiment_network(
    DG: nx.DiGraph,
    relationship_data: Dict,
    network_type: str = "undirected",
    title: str = "ASoIaF: Sentiment Network",
    file_name: str = "sentiment-network",
    subdirectory: str = "sentiment",
    show: bool = True
) -> nx.Graph:
    """
    Create a sentiment network visualization with color-coded edges.
    
    Args:
        DG: Directed graph with sentiment data
        relationship_data: Dict of dicts with relationship sentiment data
        network_type: 'undirected' or 'directed'
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        The created sentiment network graph
    """
    G = create_sentiment_network(DG, relationship_data, network_type)
    
    with Figure(file_name, FIGURE_SIZES.NETWORK, subdirectory, show=show) as fig:
        # Calculate edge colors and widths based on sentiment
        edge_colors = []
        edge_widths = []
        
        normalized_weights = get_normalized_weights(G)
        
        for u, v, data in G.edges(data=True):
            sentiment_ab = data.get('sentiment_ab', 0.0)
            sentiment_ba = data.get('sentiment_ba', 0.0)
            color, width = get_bidirectional_sentiment_color(sentiment_ab, sentiment_ba)
            edge_colors.append(color)
            edge_widths.append(width)
        
        # Node properties
        degrees = dict(G.degree())
        node_sizes = [int((degree + 1) * NETWORK_CONFIG.node_size_multiplier) 
                     for degree in degrees.values()]
        
        # Layout
        pos = nx.forceatlas2_layout(
            G,
            max_iter=NETWORK_CONFIG.layout_iterations,
            scaling_ratio=NETWORK_CONFIG.layout_scaling,
            node_mass={n: int((d + 1) * 20) for n, d in degrees.items()},
            node_size={n: int((d + 1) * 30) for n, d in degrees.items()},
            dissuade_hubs=True,
            weight='weight'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=PALETTE.STEEL_GRAY,
            alpha=PLOT_CONFIG.node_alpha,
            edgecolors=PLOT_CONFIG.node_edge_color,
            linewidths=PLOT_CONFIG.node_edge_width
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 2 + 0.5 for w in edge_widths],
            edge_color=edge_colors,
            alpha=0.8,
            arrows=False,
            node_size=node_sizes
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=NETWORK_CONFIG.label_font_size,
            font_family='serif',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=PLOT_CONFIG.label_alpha,
                pad=NETWORK_CONFIG.label_pad
            )
        )
        
        # Add legend
        legend_elements = create_sentiment_legend()
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10,
                  framealpha=0.9, title="Relationship Type", title_fontsize=11)
        
        plt.title(title, fontsize=PLOT_CONFIG.title_font_size,
                 fontweight='bold', pad=20, color=PLOT_CONFIG.title_color)
        plt.axis('off')
    
    return G


# =============================================================================
# DEGREE DISTRIBUTION PLOTS
# =============================================================================

def plot_degree_distributions(
    G: nx.DiGraph,
    file_name: str = "degree-distribution",
    subdirectory: str = "basic",
    show: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Plot in-degree, out-degree, and total degree distributions.
    
    Args:
        G: NetworkX DiGraph
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        Tuple of (degrees, in_degrees, out_degrees)
    """
    degrees, in_degrees, out_degrees = get_degree_sequences(G)
    
    with Figure(file_name, FIGURE_SIZES.EXTRA_WIDE, subdirectory, show=show) as fig:
        fig.suptitle("Degree Distributions", fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=1.02, color=PLOT_CONFIG.title_color)
        
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        
        # Define colors for each distribution
        colors = [PALETTE.WINTER_BLUE, PALETTE.EMBER, PALETTE.LOVE_GREEN]
        data_sets = [
            (in_degrees, "In-Degree Distribution", "In-Degree", ax1),
            (out_degrees, "Out-Degree Distribution", "Out-Degree", ax2),
            (degrees, "Total Degree Distribution", "Degree", ax3)
        ]
        
        for (data, title, xlabel, ax), color in zip(data_sets, colors):
            style_histogram(ax, data, bins=40, color=color, show_mean=True)
            style_axis(ax, title=title, xlabel=xlabel, ylabel="Number of Nodes",
                      grid_axis='y')
            ax.legend(fontsize=9, framealpha=0.9)
    
    return degrees, in_degrees, out_degrees


def plot_loglog_degree_distributions(
    G: nx.DiGraph,
    file_name: str = "loglog-degree-distribution",
    subdirectory: str = "basic",
    show: bool = True
) -> Tuple[float, float, float]:
    """
    Plot degree distributions in log-log scale with power-law fits.
    
    Args:
        G: NetworkX DiGraph
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        Tuple of (in_alpha, out_alpha, total_alpha) power-law exponents
    """
    degrees, in_degrees, out_degrees = get_nonzero_degrees(G)
    
    # Fit power laws
    in_alpha, in_xmin = fit_powerlaw(in_degrees)
    out_alpha, out_xmin = fit_powerlaw(out_degrees)
    total_alpha, total_xmin = fit_powerlaw(degrees)
    
    with Figure(file_name, FIGURE_SIZES.EXTRA_WIDE, subdirectory, show=show) as fig:
        fig.suptitle("Degree Distributions (Log-Log Scale)", 
                    fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=1.02, color=PLOT_CONFIG.title_color)
        
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        
        distributions = [
            (in_degrees, in_alpha, in_xmin, "In-Degree", ax1),
            (out_degrees, out_alpha, out_xmin, "Out-Degree", ax2),
            (degrees, total_alpha, total_xmin, "Total Degree", ax3)
        ]
        
        for data, alpha, xmin, label, ax in distributions:
            _plot_loglog_distribution(ax, data, alpha, xmin, label)
            style_axis(ax, title=f"{label} Distribution", 
                      xlabel=f"{label} (k)", ylabel="P(k)")
    
    return in_alpha, out_alpha, total_alpha


def _plot_loglog_distribution(
    ax: plt.Axes,
    data: List[int],
    alpha: float,
    xmin: float,
    label: str
) -> None:
    """
    Plot a single log-log distribution with power-law fit.
    
    Args:
        ax: Matplotlib axes
        data: Degree data
        alpha: Power-law exponent
        xmin: Minimum x value for fit
        label: Distribution label
    """
    # Calculate frequency distribution
    unique, counts = np.unique(data, return_counts=True)
    freq = counts / len(data)
    
    # Scatter plot
    ax.scatter(unique, freq, s=60, c=PALETTE.WINTER_BLUE, alpha=0.7,
              edgecolors='white', linewidth=0.5, label='Empirical', zorder=3)
    
    # Power-law fit line
    k_fit = np.linspace(xmin, max(unique), 100)
    mask = unique >= xmin
    if np.any(mask):
        C = freq[mask][0] * (xmin ** alpha)
        p_fit = C * k_fit ** (-alpha)
        ax.plot(k_fit, p_fit, color=PALETTE.DRAGON_RED, linewidth=2.5,
               label=f'Power-law (α = {alpha:.2f})', zorder=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')


# =============================================================================
# DIALOGUE DISTRIBUTION
# =============================================================================

def plot_dialogues_distribution(
    G: nx.DiGraph,
    file_name: str = "dialogue-distribution",
    subdirectory: str = "basic",
    show: bool = True
) -> List[int]:
    """
    Plot distribution of dialogues per directed edge.
    
    Args:
        G: NetworkX DiGraph with dialogue data
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        List of dialogue counts per edge
    """
    dialogue_counts = [len(G[u][v]['dialogues']) for u, v in G.edges()]
    
    if not dialogue_counts:
        print("No edges found in graph.")
        return []
    
    # Statistics
    stats = {
        'min': min(dialogue_counts),
        'max': max(dialogue_counts),
        'mean': np.mean(dialogue_counts),
        'median': np.median(dialogue_counts),
        'total_edges': len(dialogue_counts)
    }
    
    with Figure(file_name, FIGURE_SIZES.MEDIUM, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        num_bins = min(stats['max'] - stats['min'] + 1, 50) if stats['max'] > stats['min'] else 1
        
        style_histogram(ax, dialogue_counts, bins=num_bins, 
                       color=PALETTE.GOLD, show_mean=True, show_median=True)
        
        style_axis(ax, 
                  title="Distribution of Dialogues per Directed Edge",
                  xlabel="Number of Dialogues (n)",
                  ylabel="Number of Directed Edges",
                  grid_axis='y')
        
        ax.legend(fontsize=10, framealpha=0.9)
    
    # Print statistics
    print(f"\nDialogue Distribution Statistics:")
    print(f"Min dialogues per directed edge: {stats['min']}")
    print(f"Max dialogues per directed edge: {stats['max']}")
    print(f"Mean dialogues per directed edge: {stats['mean']:.2f}")
    print(f"Median dialogues per directed edge: {stats['median']:.1f}")
    print(f"Total directed edges: {stats['total_edges']}")
    
    return dialogue_counts


# =============================================================================
# CENTRALITY ANALYSIS
# =============================================================================

def plot_centrality_analysis(
    G: nx.DiGraph,
    limit: int = 10,
    title: str = "Character Centrality Analysis",
    file_name: str = "centrality-analysis",
    subdirectory: str = "basic",
    show: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Visualize centrality metrics for top characters.
    
    Args:
        G: NetworkX graph
        limit: Number of top characters to display
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    
    Returns:
        Tuple of (degree_centrality, betweenness_centrality, eigenvector_centrality)
    """
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("[!] Eigenvector centrality failed to converge. Using zeros.")
        eigenvector_centrality = {n: 0 for n in G.nodes()}
    
    metrics = [
        (degree_centrality, "Degree Centrality", "Activity", PALETTE.DRAGON_RED),
        (betweenness_centrality, "Betweenness Centrality", "Bridge", PALETTE.WINTER_BLUE),
        (eigenvector_centrality, "Eigenvector Centrality", "Influence", PALETTE.LOVE_GREEN)
    ]
    
    with Figure(file_name, FIGURE_SIZES.EXTRA_WIDE, subdirectory, show=show) as fig:
        fig.suptitle(title, fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=1.02, color=PLOT_CONFIG.title_color)
        
        for i, (centrality_dict, metric_title, subtitle, color) in enumerate(metrics):
            ax = fig.add_subplot(1, 3, i + 1)
            
            # Get top nodes
            sorted_nodes = sorted(centrality_dict.items(), 
                                 key=lambda x: x[1], reverse=True)[:limit]
            names = [node for node, _ in sorted_nodes][::-1]
            scores = [score for _, score in sorted_nodes][::-1]
            
            # Create horizontal bar chart
            bars = ax.barh(names, scores, color=color, 
                          edgecolor=PALETTE.NIGHT_BLACK, alpha=0.85, linewidth=0.5)
            
            # Add value labels
            add_bar_labels(ax, bars, scores, horizontal=True)
            
            # Style
            style_axis(ax, title=f"{metric_title}\n({subtitle})",
                      xlabel="Centrality Score", grid_axis='x')
            ax.set_xlim(0, max(scores) * 1.2)
            ax.tick_params(axis='y', labelsize=9)
    
    return degree_centrality, betweenness_centrality, eigenvector_centrality
