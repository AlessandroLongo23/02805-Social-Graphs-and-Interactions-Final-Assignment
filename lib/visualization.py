import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np
from lib.utils import fit_powerlaw, create_sentiment_network, get_continuous_edge_style, get_normalized_weights


def draw_network(
    G, 
    node_size_criterion='degree',
    color_parameter='degree', 
    title="Character Interaction Network: A Game of Thrones", 
    file_name="dialogue-network"
):
    # Check if graph is empty
    if G is None or len(G.nodes) == 0:
        print("Graph is empty or file not found.")
        return

    fig = plt.figure(figsize=(15, 12))

    if color_parameter == 'degree':
        if node_size_criterion == 'degree':
            degrees = dict(G.degree())
            node_colors = [degrees[n] for n in G.nodes()]
        elif node_size_criterion == 'in_degree':
            in_degrees = dict(G.in_degree())
            node_colors = [in_degrees[n] for n in G.nodes()]
        elif node_size_criterion == 'weighted_in_degree':
            weighted_in_degrees = dict(G.in_degree(weight='normalized_weight'))
            node_colors = [weighted_in_degrees[n] for n in G.nodes()]
        elif node_size_criterion == 'out_degree':
            out_degrees = dict(G.out_degree())
            node_colors = [out_degrees[n] for n in G.nodes()]
        elif node_size_criterion == 'weighted_out_degree':
            weighted_out_degrees = dict(G.out_degree(weight='normalized_weight'))
            node_colors = [weighted_out_degrees[n] for n in G.nodes()]
        else:
            node_colors = [0 for _ in G.nodes()]
    elif color_parameter == 'sentiment':
        node_colors = [G.nodes[n]['charisma'] for n in G.nodes()]
    else:
        node_colors = ['gray' for _ in G.nodes()]
    
    if node_size_criterion == 'degree':
        node_size_dict = {node: int((degree + 1) * 80) for node, degree in degrees.items()}
    elif node_size_criterion == 'in_degree':
        node_size_dict = {node: int((in_degree + 1) * 80) for node, in_degree in G.in_degree()}
    elif node_size_criterion == 'weighted_in_degree':
        node_size_dict = {node: int((weighted_in_degree + 1) * 80) for node, weighted_in_degree in G.in_degree(weight='normalized_weight')}
    elif node_size_criterion == 'out_degree':
        node_size_dict = {node: int((out_degree + 1) * 80) for node, out_degree in G.out_degree()}
    elif node_size_criterion == 'weighted_out_degree':
        node_size_dict = {node: int((weighted_out_degree + 1) * 80) for node, weighted_out_degree in G.out_degree(weight='normalized_weight')}
    else:
        node_size_dict = {node: 80 for node in G.nodes()}

    node_sizes = [node_size_dict[n] for n in G.nodes()]
    edge_colors = ['gray' for _ in G.edges()]

    pos = nx.forceatlas2_layout(G,
        max_iter=2000,
        scaling_ratio=7,
        node_mass={node: node_size_dict[node] for node in G.nodes()},
        node_size={node: node_size_dict[node] for node in G.nodes()},
        dissuade_hubs=True,
        weight='weight',
    )
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.RdYlGn,
        alpha=0.9
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=[data['normalized_weight'] * 10 + 0.5 for _, _, data in G.edges(data=True)],
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        edge_color=edge_colors,
        alpha=0.3,
        node_size=node_sizes,
        connectionstyle='arc3, rad=0.1' 
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_family='sans-serif',
        bbox=dict(facecolor="white", edgecolor='none', alpha=0.7, pad=0.5)
    )

    # Aesthetics and legend
    plt.title(title, fontsize=20)
    
    cbar = plt.colorbar(nodes)
    if color_parameter == 'degree':
        cbar.set_label('Degree', rotation=270, labelpad=15)
    elif color_parameter == 'sentiment':
        cbar.set_label('Charisma', rotation=270, labelpad=15)

    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def draw_sentiment_network(DG, relationship_data, type="undirected", title="Character Interaction Network: A Game of Thrones", file_name="sentiment-network"):
    """
    Create a sentiment network visualization.
    Edge color and width are based on bidirectional sentiment between characters.
    
    Args:
        DG: Directed graph with sentiment data
        relationship_data: dict of dicts where relationship_data[speaker][addressee] = {...} or None
    """

    if type == "undirected":
        G = create_sentiment_network(DG, relationship_data, type="undirected")
    elif type == "directed":
        G = create_sentiment_network(DG, relationship_data, type="directed")

    fig = plt.figure(figsize=(15, 12))

    normalized_weights = get_normalized_weights(G)
    
    edge_colors = []
    edge_widths = []
    
    i = 0
    for u, v, data in G.edges(data=True):
        sentiment_ab = data.get('sentiment_ab', 0.0)
        sentiment_ba = data.get('sentiment_ba', 0.0)
        color, width = get_continuous_edge_style(sentiment_ab, sentiment_ba)
        edge_colors.append(color)
        edge_widths.append(width)
        G.edges[u, v]['weight'] = normalized_weights[i]
        i += 1

    degrees = dict(G.degree())
    node_colors = ['gray' for n in G.nodes()]
    
    node_size_dict = {node: int((degree + 1) * 80) for node, degree in degrees.items()}
    node_sizes = [node_size_dict[n] for n in G.nodes()]

    pos = nx.forceatlas2_layout(G, 
        max_iter=2000,
        scaling_ratio=7,
        node_mass={node: int((degree + 1) * 20) for node, degree in degrees.items()},
        node_size={node: int((degree + 1) * 30) for node, degree in degrees.items()},
        dissuade_hubs=True,
        weight='weight',
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.RdYlGn,
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G, pos,
        width=[weight * 10 + 0.5 for weight in normalized_weights],
        edge_color=edge_colors, 
        alpha=1.0,
        arrows=True,
        arrowstyle='-',
        arrowsize=20,
        node_size=node_sizes,
        connectionstyle='arc3, rad=0.1' 
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_family='sans-serif',
        bbox=dict(facecolor="white", edgecolor='none', alpha=0.7, pad=0.5)
    )

    # 6. Add a legend explaining the colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Mutual positive (love)'),
        Patch(facecolor='red', label='Mutual negative (hate)'),
        Patch(facecolor='purple', label='Asymmetric (conflict)'),
        Patch(facecolor='grey', label='Neutral')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.title(title, 
              fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return G


def plot_loglog_degree_distributions(G, file_name="loglog-degree-distributions"):
    """
    Plot in-degree and out-degree distributions side by side in log-log scale
    with power-law fit lines.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph to analyze
        
    Returns:
    --------
    tuple : (in_alpha, out_alpha) - power-law exponents for both distributions
    """
    # Calculate degree sequences (without weights for cleaner distribution)
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    degrees = [d for n, d in G.degree()]
    
    # Filter out zero degrees for log-log plotting
    in_degrees = [d for d in in_degrees if d > 0]
    out_degrees = [d for d in out_degrees if d > 0]
    degrees = [d for d in degrees if d > 0]

    # Fit power laws
    in_alpha, in_xmin = fit_powerlaw(in_degrees)
    out_alpha, out_xmin = fit_powerlaw(out_degrees)
    alpha, xmin = fit_powerlaw(degrees)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Degree Distributions (Log-Log Scale)", fontsize=18, fontweight='bold', y=1.02)
    
    # Color palette
    scatter_color = "#3498DB"
    fit_color = "#E74C3C"
    
    # Helper function to compute and plot distribution
    def plot_distribution(ax, degrees, alpha, xmin, title, xlabel):
        # Calculate frequency distribution
        unique, counts = np.unique(degrees, return_counts=True)
        freq = counts / len(degrees)  # Normalize to probability
        
        # Scatter plot of empirical distribution
        ax.scatter(unique, freq, s=60, c=scatter_color, alpha=0.7, 
                   edgecolors='white', linewidth=0.5, label='Empirical', zorder=3)
        
        # Power-law fit line: P(k) ∝ k^(-alpha)
        # Normalize to match the empirical distribution
        k_fit = np.linspace(xmin, max(unique), 100)
        # Calculate normalization constant using xmin
        C = freq[unique >= xmin][0] * (xmin ** alpha) if np.any(unique >= xmin) else 1
        p_fit = C * k_fit ** (-alpha)
        
        ax.plot(k_fit, p_fit, color=fit_color, linewidth=2.5, 
                label=f'Power-law fit (α = {alpha:.2f})', zorder=2)
        
        # Log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Styling
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("P(k)", fontsize=11)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot both distributions
    plot_distribution(ax1, in_degrees, in_alpha, in_xmin, 
                      "In-Degree Distribution", "In-Degree (k)")
    plot_distribution(ax2, out_degrees, out_alpha, out_xmin, 
                      "Out-Degree Distribution", "Out-Degree (k)")
    plot_distribution(ax3, degrees, alpha, xmin, 
                      "Degree Distribution", "Degree (k)")
    
    plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return in_alpha, out_alpha, alpha


def plot_degree_distributions(G, file_name="degree-distributions"):
    """
    Plot in-degree, out-degree, and total degree distributions side by side.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph containing character interactions
        
    Returns:
    --------
    tuple : (in_degree_sequence, out_degree_sequence, degree_sequence)
    """
    # Calculate degree sequences
    in_degree_sequence = [d for n, d in G.in_degree()]
    out_degree_sequence = [d for n, d in G.out_degree()]
    degree_sequence = [d for n, d in G.degree()]
    
    # Calculate means
    mean_in_degree = np.mean(in_degree_sequence)
    mean_out_degree = np.mean(out_degree_sequence)
    mean_degree = np.mean(degree_sequence)
    
    # Create figure with 3 subplots matching log-log version
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Degree Distributions", fontsize=18, fontweight='bold', y=1.02)
    
    # Helper function to plot histogram with consistent styling
    def plot_histogram(ax, degree_sequence, mean_degree, title, xlabel):
        ax.hist(degree_sequence, bins=40, align='left', rwidth=0.8, 
                edgecolor='black', alpha=0.7)
        ax.axvline(mean_degree, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_degree:.2f}')
        
        # Styling to match log-log version
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Number of Nodes", fontsize=11)
        ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Plot all three distributions
    plot_histogram(ax1, in_degree_sequence, mean_in_degree,
                   "In-Degree Distribution", "In-Degree")
    plot_histogram(ax2, out_degree_sequence, mean_out_degree,
                   "Out-Degree Distribution", "Out-Degree")
    plot_histogram(ax3, degree_sequence, mean_degree,
                   "Degree Distribution", "Degree")
    
    plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return in_degree_sequence, out_degree_sequence, degree_sequence


def plot_dialogues_distribution(G, file_name="dialogues-distribution"):
    """
    Plot a histogram showing the distribution of number of dialogues per directed edge.
    
    For each n (number of dialogues), the histogram shows how many directed edges [x, y]
    have exactly n dialogues, where n counts only dialogues from x to y (one direction).
    Note: [x, y] and [y, x] are counted separately.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph containing character interactions with dialogues stored in edge attributes
        
    Returns:
    --------
    list : List of dialogue counts per directed edge
    """
    # Extract number of dialogues for each directed edge [x, y]
    # This counts only dialogues from x to y, not bidirectional
    dialogue_counts = [len(G[u][v]['dialogues']) for u, v in G.edges()]
    
    if not dialogue_counts:
        print("No edges found in graph.")
        return []
    
    # Calculate statistics
    min_dialogues = min(dialogue_counts)
    max_dialogues = max(dialogue_counts)
    mean_dialogues = np.mean(dialogue_counts)
    median_dialogues = np.median(dialogue_counts)
    
    # Determine appropriate number of bins
    # Use integer bins since we're counting discrete dialogues
    num_bins = min(max_dialogues - min_dialogues + 1, 50) if max_dialogues > min_dialogues else 1
    
    # Plot histogram
    fig = plt.figure(figsize=(12, 6))
    plt.hist(dialogue_counts, bins=num_bins, align='left', rwidth=0.8, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Number of Dialogues per Directed Edge", fontsize=14, pad=15)
    plt.xlabel("Number of Dialogues (n)", fontsize=12)
    plt.ylabel("Number of Directed Edges [x, y]", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics as vertical lines
    plt.axvline(mean_dialogues, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_dialogues:.2f}')
    plt.axvline(median_dialogues, color='blue', linestyle='--', linewidth=2, 
                label=f'Median: {median_dialogues:.1f}')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nDialogue Distribution Statistics:")
    print(f"Min dialogues per directed edge: {min_dialogues}")
    print(f"Max dialogues per directed edge: {max_dialogues}")
    print(f"Mean dialogues per directed edge: {mean_dialogues:.2f}")
    print(f"Median dialogues per directed edge: {median_dialogues:.1f}")
    print(f"Total directed edges: {len(dialogue_counts)}")
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return dialogue_counts


def plot_centrality_analysis(G, limit=10, title="Character Centrality Analysis", file_name="centrality-analysis"):
    """
    Visualize centrality metrics for top characters using horizontal bar charts.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The graph to analyze
    limit : int
        Number of top characters to display (default: 10)
    """
    # Calculate different centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("[!] Eigenvector centrality failed to converge. Defaulting to 0.")
        eigenvector_centrality = {n: 0 for n in G.nodes()}

    # Prepare data for each metric
    metrics = [
        (degree_centrality, "Degree Centrality (Activity)", "#E74C3C"),
        (betweenness_centrality, "Betweenness Centrality (Bridges)", "#3498DB"),
        (eigenvector_centrality, "Eigenvector Centrality (Influence)", "#2ECC71")
    ]
    
    # Create figure with 3 horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    for ax, (centrality_dict, title, color) in zip(axes, metrics):
        # Get top nodes
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
        names = [node for node, score in sorted_nodes]
        scores = [score for node, score in sorted_nodes]
        
        # Reverse for horizontal bar chart (highest at top)
        names = names[::-1]
        scores = scores[::-1]
        
        # Create horizontal bar chart
        bars = ax.barh(names, scores, color=color, edgecolor='white', alpha=0.85)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + max(scores) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Style the subplot
        ax.set_xlabel("Centrality Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(0, max(scores) * 1.15)  # Add space for labels
        ax.tick_params(axis='y', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Return the centrality dictionaries for further analysis
    return degree_centrality, betweenness_centrality, eigenvector_centrality