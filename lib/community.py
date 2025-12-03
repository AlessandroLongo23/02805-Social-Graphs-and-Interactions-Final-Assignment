import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import seaborn as sns

def detect_communities(G):
    """
    Detect communities using the Louvain method.
    Returns partition dictionary and modularity score.
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # remove nodes with no edges
    G_undirected.remove_nodes_from(list(nx.isolates(G_undirected)))

    # Apply Louvain method
    partition = community_louvain.best_partition(G_undirected, weight='weight')
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G_undirected, weight='weight')
    
    print(f"Number of communities detected: {len(set(partition.values()))}")
    print(f"Modularity Q-score: {modularity:.4f}")
    print(f"(Higher modularity indicates stronger community structure, max = 1.0)\n")
    
    return G_undirected, partition, modularity


def analyze_communities(partition, G):
    """
    Analyze and display information about each community.
    """
    # Count members per community
    community_counts = Counter(partition.values())
    num_communities = len(community_counts)
    
    print("="*70)
    print("COMMUNITY COMPOSITION ANALYSIS")
    print("="*70)
    
    # Sort communities by size
    sorted_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)
    
    for comm_id, size in sorted_communities:
        members = [node for node, comm in partition.items() if comm == comm_id]
        
        print(f"\n{'='*70}")
        print(f"COMMUNITY {comm_id} ({size} members):")
        print(f"{'='*70}")
        
        # List all members
        print("Members:")
        for i, member in enumerate(sorted(members, key=lambda m: G.degree(m, weight='weight'), reverse=True), 1):
            degree = G.degree(member, weight='weight')
            print(f"  {i:2d}. {member:<25} (weighted degree: {degree:.2f})")
        
        # Calculate internal vs external edges
        internal_edges = 0
        external_edges = 0
        
        for node in members:
            for neighbor in G.neighbors(node):
                if partition[neighbor] == comm_id:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        print(f"\nCommunity Statistics:")
        print(f"  Internal connections: {internal_edges}")
        print(f"  External connections: {external_edges}")
        if internal_edges + external_edges > 0:
            cohesion = internal_edges / (internal_edges + external_edges)
            print(f"  Cohesion ratio: {cohesion:.3f}")
    
    return sorted_communities


def visualize_communities(G, partition, title="Character Communities in Game of Thrones", file_name="communities-network", figsize=(18, 14), modularity=None):
    """
    Visualize the network with nodes colored by community.
    """
    fig = plt.figure(figsize=figsize)
    
    # Get unique communities and assign colors
    communities = set(partition.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    community_colors = {comm: colors[i] for i, comm in enumerate(sorted(communities))}
    
    # Node colors based on community
    node_colors = [community_colors[partition[node]] for node in G.nodes()]
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * 80 for n in G.nodes()]
    
    # Draw edges
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    normalized_weights = [(w / max_weight) * 5 + 0.5 for w in weights]

    pos = nx.forceatlas2_layout(
        G,
        max_iter=200,
        scaling_ratio=5,
        node_mass={node: node_sizes[i] for i, node in enumerate(G.nodes())},
        node_size={node: node_sizes[i] for i, node in enumerate(G.nodes())},
        dissuade_hubs=True,
        weight='weight',
    )
    
    nx.draw_networkx_edges(
        G, pos,
        width=normalized_weights,
        alpha=0.2,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=7,
        font_family='sans-serif',
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
    )
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=community_colors[comm], 
                                  markersize=12, label=f'Community {comm}',
                                  markeredgecolor='black', markeredgewidth=1)
                      for comm in sorted(communities)]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              framealpha=0.9, title=title)
    
    plt.title(f"{title}\n"
             f"(Louvain Method, Modularity Q = {modularity:.3f})", 
             fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def visualize_communities_separately_grid(G, partition, n_cols=2, figsize=(16, 6), title="Community Details - All Communities", file_name="communities-networks"):
    """
    Create separate visualization for each community in a grid layout.
    """
    communities = sorted(set(partition.values()))
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    # Adjust figure size based on number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    
    if n_communities == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    
    for i, comm_id in enumerate(communities):
        ax = axes[i]
        
        # Get nodes in this community
        comm_nodes = [node for node, comm in partition.items() if comm == comm_id]
        
        # Create subgraph for this community
        subgraph = G.subgraph(comm_nodes)
        
        if len(subgraph.nodes()) == 0:
            ax.text(0.5, 0.5, f'Community {comm_id}\n(Empty)', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        # Node sizes based on degree
        degrees = dict(subgraph.degree())
        node_sizes = [degrees[n] * 80 for n in subgraph.nodes()]
        
        # Edge weights
        if len(subgraph.edges()) > 0:
            weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
            max_weight = max(weights) if weights else 1
            normalized_weights = [(w / max_weight) * 8 + 1 for w in weights]
        else:
            normalized_weights = []
        
        pos = nx.forceatlas2_layout(
            subgraph,
            max_iter=200,
            scaling_ratio=5,
            node_mass={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
            node_size={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
            dissuade_hubs=True,
            weight='weight',
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph, pos,
            width=normalized_weights,
            alpha=0.3,
            edge_color='gray',
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_size=node_sizes,
            node_color=[colors[i]],
            alpha=0.9,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph, pos,
            font_size=9,
            font_family='sans-serif',
            font_weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5),
            ax=ax
        )
        
        ax.set_title(f'Community {comm_id} ({len(comm_nodes)} members)', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(n_communities, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def visualize_major_characters(G, partition, top_n=30, figsize=(16, 12), title="Top N Most Connected Characters", file_name="top-n-characters-network"):
    """
    Visualize only the top N most connected characters for clarity.
    """
    # Get top nodes by weighted degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [node for node, degree in top_nodes]
    
    # Create subgraph with only top nodes
    subgraph = G.subgraph(top_node_names)
    
    fig = plt.figure(figsize=figsize)
    
    # Get unique communities in this subgraph
    communities_in_subgraph = set(partition[node] for node in subgraph.nodes())
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(partition.values()))))
    community_colors = {comm: colors[i] for i, comm in enumerate(sorted(set(partition.values())))}
    
    # Node colors based on community
    node_colors = [community_colors[partition[node]] for node in subgraph.nodes()]
    
    # Node sizes based on degree
    node_sizes = [degrees[n] * 80 for n in subgraph.nodes()]
    
    # Draw edges
    if len(subgraph.edges()) > 0:
        weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
        max_weight = max(weights) if weights else 1
        normalized_weights = [(w / max_weight) * 6 + 1 for w in weights]
    else:
        normalized_weights = []
    
    pos = nx.forceatlas2_layout(
        subgraph,
        max_iter=200,
        scaling_ratio=5,
        node_mass={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
        node_size={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
        dissuade_hubs=True,
        weight='weight',
    )
    
    nx.draw_networkx_edges(
        subgraph, pos,
        width=normalized_weights,
        alpha=0.3,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw labels with larger font
    nx.draw_networkx_labels(
        subgraph, pos,
        font_size=11,
        font_family='sans-serif',
        font_weight='bold',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2)
    )
    
    # Create legend for communities that appear in this view
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=community_colors[comm], 
                                  markersize=12, label=f'Community {comm}',
                                  markeredgecolor='black', markeredgewidth=1.5)
                      for comm in sorted(communities_in_subgraph)]
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11, 
              framealpha=0.9, title='Communities', title_fontsize=12)
    
    plt.title(f"{title}\n"
             f"(Node size = interaction strength)", 
             fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()


def visualize_communities_grid(G, partition, n_cols=3, figsize=(18, 12), title="All Communities - Overview Grid", file_name="all-communities-overview-grid"):
    """
    Show all communities in a grid layout, one subplot per community.
    """
    communities = sorted(set(partition.values()))
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_communities == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(communities)))
    
    for i, comm_id in enumerate(communities):
        ax = axes[i]
        
        # Get nodes in this community
        comm_nodes = [node for node, comm in partition.items() if comm == comm_id]
        subgraph = G.subgraph(comm_nodes)
        
        if len(subgraph.nodes()) == 0:
            ax.text(0.5, 0.5, f'Community {comm_id}\n(Empty)', 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # Node sizes
        degrees = dict(subgraph.degree())
        if degrees:
            node_sizes = [degrees[n] * 80 for n in subgraph.nodes()]
        else:
            node_sizes = [100] * len(subgraph.nodes())
        
        # Edge weights
        if len(subgraph.edges()) > 0:
            weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]
            max_weight = max(weights) if weights else 1
            normalized_weights = [(w / max_weight) * 3 + 0.5 for w in weights]
        else:
            normalized_weights = []
        
        pos = nx.forceatlas2_layout(
            subgraph,
            max_iter=200,
            scaling_ratio=5,
            node_mass={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
            node_size={node: node_sizes[i] for i, node in enumerate(subgraph.nodes())},
            dissuade_hubs=True,
            weight='weight',
        )
        
        # Draw
        nx.draw_networkx_edges(subgraph, pos, width=normalized_weights, 
                              alpha=0.3, edge_color='gray', arrows=False, ax=ax)
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color=[colors[i]], alpha=0.8, 
                              edgecolors='black', linewidths=1, ax=ax)
        
        # Labels (only for communities with few members)
        if len(comm_nodes) <= 10:
            nx.draw_networkx_labels(subgraph, pos, font_size=7, 
                                   font_weight='bold', ax=ax)
        
        ax.set_title(f'Community {comm_id}\n({len(comm_nodes)} members)', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(n_communities, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title, 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def get_community_names(partition, G, top_n=2):
    """
    Generate meaningful names for communities based on their most central members.
    """
    communities = sorted(set(partition.values()))
    community_names = {}
    
    for comm_id in communities:
        # Get members of this community
        members = [node for node, comm in partition.items() if comm == comm_id]
        
        # Get their degrees (weighted)
        member_degrees = [(node, G.degree(node)) for node in members]
        
        # Sort by degree and get top N
        top_members = sorted(member_degrees, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Create name from top members
        if len(top_members) == 1:
            name = top_members[0][0]
        elif len(top_members) == 2:
            name = f"{top_members[0][0]} & {top_members[1][0]}"
        else:
            name = f"{top_members[0][0]}, {top_members[1][0]}"
            if len(top_members) > 2:
                name += f" +{len(members)-2}"
        
        community_names[comm_id] = name
    
    return community_names

def plot_community_interaction_matrix_named(G, partition, title="Inter-Community Interaction Strength", file_name="inter-community-confusion-matrix"):
    """
    Create a heatmap showing interactions between communities with meaningful names.
    """
    communities = sorted(set(partition.values()))
    n_comm = len(communities)
    
    # Get community names
    community_names = get_community_names(partition, G, top_n=2)
    
    # Initialize interaction matrix
    interaction_matrix = np.zeros((n_comm, n_comm))
    
    # Calculate interactions
    for u, v, data in G.edges(data=True):
        comm_u = partition[u]
        comm_v = partition[v]
        weight = data['weight']
        
        u_idx = communities.index(comm_u)
        v_idx = communities.index(comm_v)
        
        interaction_matrix[u_idx][v_idx] += weight
    
    # Create labels
    labels = [community_names[c] for c in communities]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(interaction_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Interaction Strength (normalized)'},
                ax=ax,
                square=True)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.title(f"{title}\n(Row = Speaker, Column = Addressee)", 
             fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Target Community', fontsize=12, fontweight='bold')
    plt.ylabel('Source Community', fontsize=12, fontweight='bold')
    plt.tight_layout()
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    # Print interpretation
    print("\n" + "="*70)
    print("COMMUNITY INTERACTION INTERPRETATION")
    print("="*70)
    print("\nDiagonal values (high) = Strong internal communication")
    print("Off-diagonal values (high) = Strong cross-community interaction")
    print("\nKey interactions:")
    
    # Find top cross-community interactions
    interactions = []
    for i, comm_i in enumerate(communities):
        for j, comm_j in enumerate(communities):
            if i != j and interaction_matrix[i][j] > 0:
                interactions.append((
                    community_names[comm_i],
                    community_names[comm_j],
                    interaction_matrix[i][j]
                ))
    
    # Sort and display top interactions
    top_interactions = sorted(interactions, key=lambda x: x[2], reverse=True)[:5]
    for rank, (source, target, strength) in enumerate(top_interactions, 1):
        print(f"  {rank}. {source:<30} → {target:<30} ({strength:.2f})")


def print_summary_statistics(partition, modularity, G):
    """
    Print comprehensive summary statistics.
    """
    print("="*70)
    print("COMMUNITY DETECTION SUMMARY")
    print("="*70)
    
    num_communities = len(set(partition.values()))
    num_nodes = len(partition)
    
    print(f"\nAlgorithm: Louvain Method")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of communities: {num_communities}")
    print(f"Modularity (Q-score): {modularity:.4f}")
    
    # Community size statistics
    community_sizes = Counter(partition.values())
    sizes = list(community_sizes.values())
    
    print(f"\nCommunity Size Statistics:")
    print(f"  Largest community: {max(sizes)} members")
    print(f"  Smallest community: {min(sizes)} members")
    print(f"  Average community size: {np.mean(sizes):.2f} members")
    print(f"  Median community size: {np.median(sizes):.2f} members")
    
    # Interpretation guide
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("Modularity Score Interpretation:")
    print("  Q > 0.3  : Strong community structure")
    print("  Q = 0.2-0.3 : Moderate community structure")
    print("  Q < 0.2  : Weak community structure")
    print(f"\nModularity: {modularity:.4f} indicates ", end="")
    if modularity > 0.3:
        print("STRONG community structure")
    elif modularity > 0.2:
        print("MODERATE community structure")
    else:
        print("WEAK community structure")