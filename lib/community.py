"""
Community detection and analysis for ASoIaF Network Analysis.
Provides Louvain-based community detection and visualization.
"""
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
from typing import Tuple, Dict, List, Optional, Any

from lib.config import PLOT_CONFIG, FIGURE_SIZES, NETWORK_CONFIG, PALETTE
from lib.style import Figure, style_axis, add_title


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def detect_communities(G: nx.DiGraph) -> Tuple[nx.Graph, Dict[str, int], float]:
    """
    Detect communities using the Louvain method.
    
    Args:
        G: NetworkX directed graph
    
    Returns:
        Tuple of (undirected_graph, partition_dict, modularity_score)
    """
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    # Remove isolated nodes
    G_undirected.remove_nodes_from(list(nx.isolates(G_undirected)))
    
    # Apply Louvain method
    partition = community_louvain.best_partition(G_undirected, weight='weight')
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G_undirected, weight='weight')
    
    print(f"Number of communities detected: {len(set(partition.values()))}")
    print(f"Modularity Q-score: {modularity:.4f}")
    print("(Higher modularity indicates stronger community structure, max = 1.0)\n")
    
    return G_undirected, partition, modularity


# =============================================================================
# COMMUNITY ANALYSIS
# =============================================================================

def analyze_communities(
    partition: Dict[str, int],
    G: nx.Graph
) -> List[Tuple[int, int]]:
    """
    Analyze and display detailed information about each community.
    
    Args:
        partition: Dict mapping nodes to community IDs
        G: NetworkX graph
    
    Returns:
        List of (community_id, size) tuples sorted by size
    """
    community_counts = Counter(partition.values())
    
    print("=" * 70)
    print("COMMUNITY COMPOSITION ANALYSIS")
    print("=" * 70)
    
    sorted_communities = sorted(community_counts.items(), key=lambda x: x[1], reverse=True)
    
    for comm_id, size in sorted_communities:
        members = [node for node, comm in partition.items() if comm == comm_id]
        
        print(f"\n{'='*70}")
        print(f"COMMUNITY {comm_id} ({size} members)")
        print("=" * 70)
        
        # List members sorted by weighted degree
        print("Members:")
        sorted_members = sorted(members, 
                               key=lambda m: G.degree(m, weight='weight'), 
                               reverse=True)
        for i, member in enumerate(sorted_members, 1):
            degree = G.degree(member, weight='weight')
            print(f"  {i:2d}. {member:<25} (weighted degree: {degree:.1f})")
        
        # Calculate internal vs external connections
        internal_edges = 0
        external_edges = 0
        
        for node in members:
            for neighbor in G.neighbors(node):
                if partition.get(neighbor, -1) == comm_id:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        # Print statistics
        print(f"\nCommunity Statistics:")
        print(f"  Internal connections: {internal_edges}")
        print(f"  External connections: {external_edges}")
        
        if internal_edges + external_edges > 0:
            cohesion = internal_edges / (internal_edges + external_edges)
            print(f"  Cohesion ratio: {cohesion:.3f}")
    
    return sorted_communities


def get_community_names(
    partition: Dict[str, int],
    G: nx.Graph,
    top_n: int = 2
) -> Dict[int, str]:
    """
    Generate meaningful names for communities based on their most central members.
    
    Args:
        partition: Dict mapping nodes to community IDs
        G: NetworkX graph
        top_n: Number of top members to include in name
    
    Returns:
        Dict mapping community IDs to descriptive names
    """
    communities = sorted(set(partition.values()))
    community_names = {}
    
    for comm_id in communities:
        members = [node for node, comm in partition.items() if comm == comm_id]
        member_degrees = [(node, G.degree(node, weight='weight')) for node in members]
        top_members = sorted(member_degrees, key=lambda x: x[1], reverse=True)[:top_n]
        
        if len(top_members) == 1:
            name = top_members[0][0]
        elif len(top_members) == 2:
            name = f"{top_members[0][0]} & {top_members[1][0]}"
        else:
            name = f"{top_members[0][0]}, {top_members[1][0]}"
            if len(members) > 2:
                name += f" (+{len(members)-2})"
        
        community_names[comm_id] = name
    
    return community_names


# =============================================================================
# COMMUNITY VISUALIZATION
# =============================================================================

def _get_community_colors(partition: Dict[str, int]) -> Dict[int, np.ndarray]:
    """Get color mapping for communities."""
    communities = sorted(set(partition.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(communities), 10)))
    return {comm: colors[i % len(colors)] for i, comm in enumerate(communities)}


def visualize_communities(
    G: nx.Graph,
    partition: Dict[str, int],
    modularity: float = None,
    title: str = "Character Communities in ASoIaF",
    file_name: str = "communities-network",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Visualize the network with nodes colored by community.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        modularity: Modularity score (optional, for title)
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    with Figure(file_name, FIGURE_SIZES.NETWORK, subdirectory, show=show) as fig:
        # Layout
        pos = nx.spring_layout(G, k=NETWORK_CONFIG.layout_k, iterations=200, seed=42)
        
        # Community colors
        community_colors = _get_community_colors(partition)
        node_colors = [community_colors[partition[node]] for node in G.nodes()]
        
        # Node sizes based on degree
        degrees = dict(G.degree(weight='weight'))
        node_sizes = [degrees[n] * 50 + NETWORK_CONFIG.min_node_size for n in G.nodes()]
        
        # Edge weights
        weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        edge_widths = [(w / max_weight) * 5 + 0.5 for w in weights]
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.2,
            edge_color=PALETTE.IRON_GRAY,
            arrows=True,
            arrowsize=15,
            connectionstyle=NETWORK_CONFIG.connection_style
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=PLOT_CONFIG.node_alpha,
            edgecolors=PALETTE.NIGHT_BLACK,
            linewidths=PLOT_CONFIG.node_edge_width
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=7,
            font_family='serif',
            font_weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', 
                     alpha=PLOT_CONFIG.label_alpha, pad=1)
        )
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=community_colors[comm],
                      markersize=12, label=f'Community {comm}',
                      markeredgecolor=PALETTE.NIGHT_BLACK, markeredgewidth=1)
            for comm in sorted(community_colors.keys())
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=9,
                  framealpha=0.9, title="Communities", title_fontsize=10)
        
        # Title
        title_text = title
        if modularity is not None:
            title_text += f"\n(Louvain Method, Modularity Q = {modularity:.3f})"
        
        plt.title(title_text, fontsize=PLOT_CONFIG.title_font_size,
                 fontweight='bold', pad=20, color=PLOT_CONFIG.title_color)
        plt.axis('off')


def visualize_communities_separately_grid(
    G: nx.Graph,
    partition: Dict[str, int],
    n_cols: int = 2,
    title: str = "Community Details - All Communities",
    file_name: str = "communities-networks",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Create separate visualization for each community in a grid layout.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        n_cols: Number of columns in grid
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    communities = sorted(set(partition.values()))
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    figsize = (n_cols * 8, n_rows * 6)
    
    with Figure(file_name, figsize, subdirectory, show=show) as fig:
        axes = []
        for i in range(n_communities):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            axes.append(ax)
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_communities, 10)))
        
        for i, comm_id in enumerate(communities):
            ax = axes[i]
            
            # Get community members
            comm_nodes = [node for node, comm in partition.items() if comm == comm_id]
            subgraph = G.subgraph(comm_nodes)
            
            if len(subgraph.nodes()) == 0:
                ax.text(0.5, 0.5, f'Community {comm_id}\n(Empty)',
                       ha='center', va='center', fontsize=12,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Layout
            pos = nx.spring_layout(subgraph, k=2, iterations=100, seed=42)
            
            # Node sizes
            degrees = dict(subgraph.degree(weight='weight'))
            node_sizes = [degrees[n] * 100 + 200 for n in subgraph.nodes()]
            
            # Edge weights
            if len(subgraph.edges()) > 0:
                weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
                max_weight = max(weights) if weights else 1
                edge_widths = [(w / max_weight) * 8 + 1 for w in weights]
            else:
                edge_widths = []
            
            # Draw
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.3,
                                  edge_color=PALETTE.IRON_GRAY, arrows=True,
                                  arrowsize=15, connectionstyle=NETWORK_CONFIG.connection_style,
                                  ax=ax)
            
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                  node_color=[colors[i % len(colors)]],
                                  alpha=PLOT_CONFIG.node_alpha,
                                  edgecolors=PALETTE.NIGHT_BLACK,
                                  linewidths=2, ax=ax)
            
            nx.draw_networkx_labels(subgraph, pos, font_size=9,
                                   font_family='serif', font_weight='bold',
                                   bbox=dict(facecolor='white', edgecolor='none',
                                            alpha=0.8, pad=1.5), ax=ax)
            
            ax.set_title(f'Community {comm_id} ({len(comm_nodes)} members)',
                        fontsize=12, fontweight='bold', pad=10,
                        color=PLOT_CONFIG.title_color)
            ax.axis('off')
        
        # Hide empty axes
        for j in range(n_communities, n_rows * n_cols):
            if j < len(fig.get_axes()):
                fig.get_axes()[j].axis('off')
        
        plt.suptitle(title, fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=0.995, color=PLOT_CONFIG.title_color)


def visualize_major_characters(
    G: nx.Graph,
    partition: Dict[str, int],
    top_n: int = 30,
    title: str = "Top N Most Connected Characters",
    file_name: str = "top-n-characters-network",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Visualize only the top N most connected characters for clarity.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        top_n: Number of top characters to include
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    # Get top nodes by weighted degree
    degrees = dict(G.degree(weight='weight'))
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_names = [node for node, _ in top_nodes]
    
    subgraph = G.subgraph(top_node_names)
    
    with Figure(file_name, FIGURE_SIZES.NETWORK, subdirectory, show=show) as fig:
        # Layout
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        
        # Community colors
        community_colors = _get_community_colors(partition)
        node_colors = [community_colors[partition[node]] for node in subgraph.nodes()]
        
        # Node sizes
        node_sizes = [degrees[n] * 80 + 300 for n in subgraph.nodes()]
        
        # Edge weights
        if len(subgraph.edges()) > 0:
            weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
            max_weight = max(weights) if weights else 1
            edge_widths = [(w / max_weight) * 6 + 1 for w in weights]
        else:
            edge_widths = []
        
        # Draw
        nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.3,
                              edge_color=PALETTE.IRON_GRAY, arrows=True,
                              arrowsize=20, connectionstyle=NETWORK_CONFIG.connection_style)
        
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                              node_color=node_colors, alpha=PLOT_CONFIG.node_alpha,
                              edgecolors=PALETTE.NIGHT_BLACK, linewidths=2)
        
        nx.draw_networkx_labels(subgraph, pos, font_size=11,
                               font_family='serif', font_weight='bold',
                               bbox=dict(facecolor='white', edgecolor='none',
                                        alpha=0.8, pad=2))
        
        # Legend
        communities_in_subgraph = set(partition[node] for node in subgraph.nodes())
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=community_colors[comm],
                      markersize=12, label=f'Community {comm}',
                      markeredgecolor=PALETTE.NIGHT_BLACK, markeredgewidth=1.5)
            for comm in sorted(communities_in_subgraph)
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=11,
                  framealpha=0.9, title='Communities', title_fontsize=12)
        
        plt.title(f"{title}\n(Node size = interaction strength)",
                 fontsize=PLOT_CONFIG.title_font_size, fontweight='bold', pad=20,
                 color=PLOT_CONFIG.title_color)
        plt.axis('off')


def visualize_communities_grid(
    G: nx.Graph,
    partition: Dict[str, int],
    n_cols: int = 3,
    title: str = "All Communities - Overview Grid",
    file_name: str = "all-communities-overview-grid",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Show all communities in a compact grid layout.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        n_cols: Number of columns in grid
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    communities = sorted(set(partition.values()))
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    figsize = (n_cols * 6, n_rows * 4)
    
    with Figure(file_name, figsize, subdirectory, show=show) as fig:
        axes = []
        for i in range(n_communities):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            axes.append(ax)
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_communities, 10)))
        
        for i, comm_id in enumerate(communities):
            ax = axes[i]
            
            comm_nodes = [node for node, comm in partition.items() if comm == comm_id]
            subgraph = G.subgraph(comm_nodes)
            
            if len(subgraph.nodes()) == 0:
                ax.text(0.5, 0.5, f'Community {comm_id}\n(Empty)',
                       ha='center', va='center', fontsize=10,
                       transform=ax.transAxes)
                ax.axis('off')
                continue
            
            pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)
            
            degrees = dict(subgraph.degree(weight='weight'))
            node_sizes = [degrees[n] * 30 + 50 for n in subgraph.nodes()] if degrees else [100] * len(subgraph.nodes())
            
            if len(subgraph.edges()) > 0:
                weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
                max_weight = max(weights) if weights else 1
                edge_widths = [(w / max_weight) * 3 + 0.5 for w in weights]
            else:
                edge_widths = []
            
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.3,
                                  edge_color=PALETTE.IRON_GRAY, arrows=False, ax=ax)
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                                  node_color=[colors[i % len(colors)]], alpha=0.8,
                                  edgecolors=PALETTE.NIGHT_BLACK, linewidths=1, ax=ax)
            
            if len(comm_nodes) <= 10:
                nx.draw_networkx_labels(subgraph, pos, font_size=7,
                                       font_weight='bold', ax=ax)
            
            ax.set_title(f'Community {comm_id}\n({len(comm_nodes)} members)',
                        fontsize=10, fontweight='bold', color=PLOT_CONFIG.title_color)
            ax.axis('off')
        
        for j in range(n_communities, n_rows * n_cols):
            if j < len(fig.get_axes()):
                fig.get_axes()[j].axis('off')
        
        plt.suptitle(title, fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=0.995, color=PLOT_CONFIG.title_color)


# =============================================================================
# INTERACTION MATRIX
# =============================================================================

def plot_community_interaction_matrix_named(
    G: nx.Graph,
    partition: Dict[str, int],
    title: str = "Inter-Community Interaction Strength",
    file_name: str = "inter-community-confusion-matrix",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Create a heatmap showing interactions between communities.
    
    Args:
        G: NetworkX graph
        partition: Dict mapping nodes to community IDs
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    communities = sorted(set(partition.values()))
    n_comm = len(communities)
    
    community_names = get_community_names(partition, G, top_n=2)
    
    # Calculate interaction matrix
    interaction_matrix = np.zeros((n_comm, n_comm))
    
    for u, v, data in G.edges(data=True):
        if u in partition and v in partition:
            comm_u = partition[u]
            comm_v = partition[v]
            weight = data.get('weight', 1)
            
            u_idx = communities.index(comm_u)
            v_idx = communities.index(comm_v)
            interaction_matrix[u_idx][v_idx] += weight
    
    labels = [community_names[c] for c in communities]
    
    with Figure(file_name, FIGURE_SIZES.HEATMAP, subdirectory, show=show) as fig:
        ax = fig.add_subplot(1, 1, 1)
        
        sns.heatmap(interaction_matrix, annot=True, fmt='.0f',
                   cmap='YlOrRd', xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Interaction Strength'}, ax=ax, square=True,
                   linewidths=0.5, linecolor=PALETTE.SNOW_WHITE)
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.title(f"{title}\n(Row = Source, Column = Target)",
                 fontsize=PLOT_CONFIG.title_font_size, fontweight='bold', pad=15,
                 color=PLOT_CONFIG.title_color)
        plt.xlabel('Target Community', fontsize=PLOT_CONFIG.label_font_size, fontweight='bold')
        plt.ylabel('Source Community', fontsize=PLOT_CONFIG.label_font_size, fontweight='bold')
    
    # Print interpretation
    print("\n" + "=" * 70)
    print("COMMUNITY INTERACTION INTERPRETATION")
    print("=" * 70)
    print("\nDiagonal values (high) = Strong internal communication")
    print("Off-diagonal values (high) = Strong cross-community interaction")
    print("\nTop cross-community interactions:")
    
    interactions = []
    for i, comm_i in enumerate(communities):
        for j, comm_j in enumerate(communities):
            if i != j and interaction_matrix[i][j] > 0:
                interactions.append((community_names[comm_i], community_names[comm_j],
                                   interaction_matrix[i][j]))
    
    top_interactions = sorted(interactions, key=lambda x: x[2], reverse=True)[:5]
    for rank, (source, target, strength) in enumerate(top_interactions, 1):
        print(f"  {rank}. {source:<30} → {target:<30} ({strength:.0f})")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(
    partition: Dict[str, int],
    modularity: float,
    G: nx.Graph
) -> None:
    """
    Print comprehensive summary statistics for community detection.
    
    Args:
        partition: Dict mapping nodes to community IDs
        modularity: Modularity score
        G: NetworkX graph
    """
    print("=" * 70)
    print("COMMUNITY DETECTION SUMMARY")
    print("=" * 70)
    
    num_communities = len(set(partition.values()))
    num_nodes = len(partition)
    
    print(f"\nAlgorithm: Louvain Method")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of communities: {num_communities}")
    print(f"Modularity (Q-score): {modularity:.4f}")
    
    community_sizes = Counter(partition.values())
    sizes = list(community_sizes.values())
    
    print(f"\nCommunity Size Statistics:")
    print(f"  Largest community: {max(sizes)} members")
    print(f"  Smallest community: {min(sizes)} members")
    print(f"  Average community size: {np.mean(sizes):.2f} members")
    print(f"  Median community size: {np.median(sizes):.2f} members")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("Modularity Score Interpretation:")
    print("  Q > 0.3   : Strong community structure")
    print("  Q = 0.2-0.3 : Moderate community structure")
    print("  Q < 0.2   : Weak community structure")
    
    print(f"\nYour modularity ({modularity:.4f}) indicates: ", end="")
    if modularity > 0.3:
        print("STRONG community structure ✓")
    elif modularity > 0.2:
        print("MODERATE community structure")
    else:
        print("WEAK community structure")
