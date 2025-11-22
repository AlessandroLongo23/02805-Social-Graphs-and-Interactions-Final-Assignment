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
    for index, row in all_dialogues.iterrows():
        speaker_raw = row['Speaker(s)']
        addressee_raw = row['Addressee(s)']
        dialogue = row['Dialogue']

        if pd.notna(addressee_raw) and pd.notna(speaker_raw):
            speakers = [speaker.strip() for speaker in str(speaker_raw).split(';')]
            addressees = [addressee.strip() for addressee in str(addressee_raw).split(';')]

            for speaker in speakers:
                for addressee in addressees:
                    if G.has_edge(speaker, addressee):
                        G[speaker][addressee]['weight'] += len(dialogue)
                        G[speaker][addressee]['dialogues'].append(dialogue)
                    else:
                        G.add_edge(speaker, addressee, weight=len(dialogue), dialogues=[dialogue])
    
    # Normalize weights
    max_weight = 0
    for u, v, data in G.edges(data=True):
        if data['weight'] > max_weight:
            max_weight = data['weight']

    if max_weight > 0:
        for u, v, data in G.edges(data=True):
            data['weight'] /= max_weight

    return all_dialogues, G


def draw_graph(G):
    # Check if graph is empty
    if G is None or len(G.nodes) == 0:
        print("Graph is empty or file not found.")
        return

    # Setup plot
    plt.figure(figsize=(15, 12))

    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    # Calculate node colors based on degree
    degrees = dict(G.degree())
    node_colors = [degrees[n] for n in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=[degree*80 for degree in degrees.values()],
        node_color=node_colors,
        cmap=plt.cm.plasma,
        alpha=0.9
    )

    # Calculate edge size based on amount of dialogues
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    normalized_weights = [(w / max_weight) * 10 + 1 for w in weights]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=normalized_weights,
        arrowstyle='-|>',
        arrowsize=20,
        edge_color='gray',
        alpha=0.3,
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
    print(f"Average number of dialogues: {np.mean([len(G[u][v]['dialogues']) for u, v in G.edges()])}")


def degree_distribution(G):
    degree_sequence = [d for n, d in G.degree(weight="weight")]

    # Calculate mean and variance
    mean_degree = np.mean(degree_sequence)
    variance_degree = np.var(degree_sequence)

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


def in_degree_distribution(G):
    in_degree_sequence = [d for n, d in G.in_degree(weight="weight")]

    # Calculate mean and variance
    mean_in_degree = np.mean(in_degree_sequence)
    variance_in_degree = np.var(in_degree_sequence)

    # Plot histogram
    plt.hist(in_degree_sequence, bins=40, align='left', rwidth=0.8)
    plt.title("In-Degree Distribution")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("In-Degree")
    plt.ylabel("Number of Nodes")

    # Add mean and variance as vertical dashed lines
    plt.axvline(mean_in_degree, color='red', linestyle='--', label=f'Mean: {mean_in_degree:.2f}')
    # plt.axvline(variance_in_degree, color='blue', linestyle='--', label=f'Variance: {variance_in_degree:.2f}')
    plt.legend()

    plt.show()

    return in_degree_sequence


def out_degree_distribution(G):
    out_degree_sequence = [d for n, d in G.out_degree(weight="weight")]

    # Calculate mean and variance
    mean_out_degree = np.mean(out_degree_sequence)
    variance_out_degree = np.var(out_degree_sequence)

    # Plot histogram
    plt.hist(out_degree_sequence, bins=40, align='left', rwidth=0.8)
    plt.title("Out-Degree Distribution")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Out-Degree")
    plt.ylabel("Number of Nodes")

    # Add mean and variance as vertical dashed lines
    plt.axvline(mean_out_degree, color='red', linestyle='--', label=f'Mean: {mean_out_degree:.2f}')
    # plt.axvline(variance_out_degree, color='blue', linestyle='--', label=f'Variance: {variance_out_degree:.2f}')
    plt.legend()

    plt.show()

    return out_degree_sequence


def in_out_degree_distributions(G):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    in_degree_sequence = [d for n, d in G.in_degree(weight="weight")]
    mean_in_degree = np.mean(in_degree_sequence)
    
    ax1.hist(in_degree_sequence, bins=40, align='left', rwidth=0.8)
    ax1.set_title("In-Degree Distribution")
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Number of Nodes")
    ax1.axvline(mean_in_degree, color='red', linestyle='--', label=f'Mean: {mean_in_degree:.2f}')
    ax1.legend()

    out_degree_sequence = [d for n, d in G.out_degree(weight="weight")]
    mean_out_degree = np.mean(out_degree_sequence)
    
    ax2.hist(out_degree_sequence, bins=40, align='left', rwidth=0.8)
    ax2.set_title("Out-Degree Distribution")
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Number of Nodes")
    ax2.axvline(mean_out_degree, color='red', linestyle='--', label=f'Mean: {mean_out_degree:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return in_degree_sequence, out_degree_sequence


def fit_powerlaw(degree_sequence):
    degree_fit = powerlaw.Fit(degree_sequence, discrete=True, xmin=1)
    
    alpha = degree_fit.alpha

    xmin = degree_fit.xmin

    print(f"Degree: α = {alpha:.3f}  (xmin = {xmin})")

    return alpha


def centrality_analysis(G):
    def print_top_nodes(centrality_dict, title, limit=10):
        print(f"\n{title}")
    
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:limit]
    
        for rank, (node, score) in enumerate(sorted_nodes, 1):
            print(f"{rank:>2}. {node:<20} {score:.4f}")
    
    # Calculate different centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        print("\n[!] Eigenvector centrality failed to converge. Defaulting to 0.")
        eigenvector_centrality = {n: 0 for n in G.nodes()}

    # Print top 10 nodes for each centrality metric
    print_top_nodes(degree_centrality, "Top 10 by Degree (Activity)")
    print_top_nodes(betweenness_centrality, "Top 10 by Betweenness (Bridges)")
    print_top_nodes(eigenvector_centrality, "Top 10 by Eigenvector (Influence)")


def assortativity_analysis(G):
    r = nx.degree_assortativity_coefficient(G, weight="weight")
    print(f"Degree Assortativity Coefficient (r): {r:.4f}")
