import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import powerlaw


def create_graph(path_name="data/dialogues.csv"):
    # Load dataset
    all_dialogues = pd.read_csv(path_name)

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
        
    return all_dialogues, G


def add_edges(G, path_name="data/dialogues.csv", edge_type="direct"):
    # Load dataset
    all_dialogues = pd.read_csv(path_name)

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
            data['weight'] /= max_weight

    return all_dialogues, G


def draw_graph(G, spring_layout_args={'k': 0.66, 'iterations': 200, 'seed': 42}, color_parameter='degree'):
    # Check if graph is empty
    if G is None or len(G.nodes) == 0:
        print("Graph is empty or file not found.")
        return

    # Setup plot
    plt.figure(figsize=(15, 12))

    # 1. Calculate edge size based on amount of dialogues
    max_weight = 0
    for _, _, data in G.edges(data=True):
        if data['weight'] > max_weight:
            max_weight = data['weight']

    if max_weight > 0:
        for _, _, data in G.edges(data=True):
            data['weight'] = data['weight'] / max_weight
    normalized_weights = [data['weight'] for _, _, data in G.edges(data=True)]


    pos = nx.spring_layout(G, **spring_layout_args)

    # Calculate node colors based on degree
    degrees = dict(G.degree())
    if color_parameter == 'degree':
        node_colors = [degrees[n] for n in G.nodes()]
    elif color_parameter == 'sentiment':
        node_colors = [G.nodes[n]['charisma'] for n in G.nodes()]
    
    node_size_dict = {node: int((degree + 1) * 80) for node, degree in degrees.items()}
    node_sizes = [node_size_dict[n] for n in G.nodes()]
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.RdYlGn,
        alpha=0.9
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=[weight * 10 + 0.5 for weight in normalized_weights],
        arrowstyle='-|>',
        arrowsize=20,
        edge_color='gray',
        alpha=0.3,
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

    # Aesthetics and legend
    plt.title("Character Interaction Network: A Game of Thrones", fontsize=20)
    
    cbar = plt.colorbar(nodes)
    cbar.set_label('Degree', rotation=270, labelpad=15)

    plt.axis('off')
    plt.tight_layout()

    # Show plot
    plt.show()


def graph_data(G):
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


def dialogue_distribution(G):
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
    plt.figure(figsize=(12, 6))
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
    plt.show()
    
    # Print summary statistics
    print(f"\nDialogue Distribution Statistics:")
    print(f"Min dialogues per directed edge: {min_dialogues}")
    print(f"Max dialogues per directed edge: {max_dialogues}")
    print(f"Mean dialogues per directed edge: {mean_dialogues:.2f}")
    print(f"Median dialogues per directed edge: {median_dialogues:.1f}")
    print(f"Total directed edges: {len(dialogue_counts)}")
    
    return dialogue_counts


def degree_distribution(G):
    degree_sequence = [d for n, d in G.degree(weight="weight")]

    # Calculate mean and variance
    mean_degree = np.mean(degree_sequence)

    # Plot histogram
    plt.hist(degree_sequence, bins=40, align='left', rwidth=0.8)
    plt.title("Degree Distribution")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")

    # Add mean and variance as vertical dashed lines
    plt.axvline(mean_degree, color='red', linestyle='--', label=f'Mean: {mean_degree:.2f}')
    # plt.axvline(variance_degree, color='blue', linestyle='--', label=f'Variance: {variance_degree:.2f}')
    plt.legend()

    plt.show()

    return degree_sequence


def in_out_degree_distributions(G):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    in_degree_sequence = [d for n, d in G.in_degree(weight="weight")]
    mean_in_degree = np.mean(in_degree_sequence)
    
    ax1.hist(in_degree_sequence, bins=40, align='left', rwidth=0.8)
    ax1.set_title("In-Degree Distribution")
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Number of Nodes")
    ax1.axvline(mean_in_degree, color='red', linestyle='--', label=f'Mean: {mean_in_degree:.2f}')
    ax1.legend()

    out_degree_sequence = [d for n, d in G.out_degree(weight="weight")]
    mean_out_degree = np.mean(out_degree_sequence)
    
    ax2.hist(out_degree_sequence, bins=40, align='left', rwidth=0.8)
    ax2.set_title("Out-Degree Distribution")
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Number of Nodes")
    ax2.axvline(mean_out_degree, color='red', linestyle='--', label=f'Mean: {mean_out_degree:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return in_degree_sequence, out_degree_sequence


def fit_powerlaw(degree_sequence):
    degree_fit = powerlaw.Fit(degree_sequence, discrete=True, xmin=1, verbose=False)
    alpha = degree_fit.alpha
    xmin = degree_fit.xmin
    return alpha, xmin


def loglog_degree_distributions(G):
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
    plt.show()
    
    return in_alpha, out_alpha, alpha


def centrality_analysis(G, limit=10):
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
    fig.suptitle("Character Centrality Analysis", fontsize=18, fontweight='bold', y=1.02)
    
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
    plt.show()
    
    # Return the centrality dictionaries for further analysis
    return degree_centrality, betweenness_centrality, eigenvector_centrality


def assortativity_analysis(G):
    r = nx.degree_assortativity_coefficient(G, weight="weight")
    print(f"Degree Assortativity Coefficient (r): {r:.4f}")
