import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np
from lib.utils import fit_powerlaw, create_sentiment_network, get_continuous_edge_style


def _compute_node_metric(G, criterion, weight_attr='normalized_weight'):
    """
    Compute node metric values based on the given criterion.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    criterion : str
        One of: 'degree', 'in_degree', 'out_degree', 'weighted_degree',
                'weighted_in_degree', 'weighted_out_degree'
    weight_attr : str
        Edge weight attribute name for weighted metrics
        
    Returns:
    --------
    dict : {node: metric_value}
    """
    is_directed = G.is_directed()
    
    if criterion == 'degree':
        return dict(G.degree())
    elif criterion == 'in_degree':
        if not is_directed:
            return dict(G.degree())
        return dict(G.in_degree())
    elif criterion == 'out_degree':
        if not is_directed:
            return dict(G.degree())
        return dict(G.out_degree())
    elif criterion == 'weighted_degree':
        return dict(G.degree(weight=weight_attr))
    elif criterion == 'weighted_in_degree':
        if not is_directed:
            return dict(G.degree(weight=weight_attr))
        return dict(G.in_degree(weight=weight_attr))
    elif criterion == 'weighted_out_degree':
        if not is_directed:
            return dict(G.degree(weight=weight_attr))
        return dict(G.out_degree(weight=weight_attr))
    else:
        # Default: uniform value
        return {node: 1 for node in G.nodes()}


def _compute_node_colors(G, criterion, fixed_color='gray', weight_attr='normalized_weight'):
    """
    Compute node colors based on the given criterion.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    criterion : str
        One of: 'degree', 'in_degree', 'out_degree', 'weighted_degree',
                'weighted_in_degree', 'weighted_out_degree', 'charisma', 'fixed'
    fixed_color : str
        Color to use when criterion is 'fixed'
    weight_attr : str
        Edge weight attribute name for weighted metrics
        
    Returns:
    --------
    tuple : (colors_list, is_numeric) - colors and whether they need a colormap
    """
    if criterion == 'charisma':
        colors = [G.nodes[n].get('charisma', 0) for n in G.nodes()]
        return colors, True
    elif criterion == 'fixed':
        return [fixed_color for _ in G.nodes()], False
    else:
        metric = _compute_node_metric(G, criterion, weight_attr)
        return [metric[n] for n in G.nodes()], True


def _compute_node_sizes(G, criterion, scale=80, weight_attr='normalized_weight'):
    """
    Compute node sizes based on the given criterion.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    criterion : str
        Same options as _compute_node_metric, or 'fixed'
    scale : float
        Scaling factor for sizes
    weight_attr : str
        Edge weight attribute name for weighted metrics
        
    Returns:
    --------
    tuple : (sizes_list, size_dict)
    """
    if criterion == 'fixed':
        size_dict = {node: scale for node in G.nodes()}
    else:
        metric = _compute_node_metric(G, criterion, weight_attr)
        size_dict = {node: int((val + 1) * scale) for node, val in metric.items()}
    
    sizes = [size_dict[n] for n in G.nodes()]
    return sizes, size_dict


def _compute_edge_colors(G, criterion, fixed_color='gray'):
    """
    Compute edge colors based on the given criterion.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    criterion : str
        One of: 'sentiment', 'fixed'
    fixed_color : str
        Color to use when criterion is 'fixed' or 'weight'
        
    Returns:
    --------
    list : Edge colors
    """
    if criterion == 'sentiment':
        colors = []
        for u, v, data in G.edges(data=True):
            sentiment_ab = data.get('sentiment_ab', 0.0)
            sentiment_ba = data.get('sentiment_ba', 0.0)
            color, _ = get_continuous_edge_style(sentiment_ab, sentiment_ba)
            colors.append(color)
        return colors
    else:
        return [fixed_color for _ in G.edges()]


def _compute_edge_widths(G, scale=10, weight_attr='weight', min_width=0.5):
    """
    Compute edge widths based on weight attribute.
    
    Parameters:
    -----------
    G : networkx.Graph or networkx.DiGraph
        The graph to analyze
    scale : float
        Scaling factor for widths
    weight_attr : str
        Edge weight attribute name
    min_width : float
        Minimum edge width
        
    Returns:
    --------
    list : Edge widths
    """
    widths = []
    for _, _, data in G.edges(data=True):
        w = data.get(weight_attr, data.get('normalized_weight', 0.1))
        widths.append(w * scale + min_width)
    return widths


def draw_network(
    G,
    # Optional: create sentiment network from relationship data
    relationship_data=None,
    graph_type=None,  # "directed", "undirected", or None (auto-detect)
    
    # Node filtering
    remove_isolates=False,  # Remove nodes with degree 0
    
    # Node appearance
    node_color_criterion='degree',  # 'degree', 'in_degree', 'out_degree', 'weighted_*', 'charisma', 'fixed'
    node_size_criterion='degree',   # same options (except 'charisma')
    node_fixed_color='gray',
    node_size_scale=80,
    node_cmap=plt.cm.RdYlGn,
    node_alpha=1,
    edge_color='black',
    edge_linewidth=1,
    
    # Edge appearance
    edge_color_criterion='fixed',  # 'sentiment', 'fixed'
    edge_fixed_color='gray',
    edge_width_scale=10,
    edge_weight_attr='normalized_weight',
    edge_alpha=0.25,
    show_arrows=None,  # None = auto (True for directed), or explicit True/False
    arrow_style='-|>',
    arrow_size=20,
    connection_style='arc3, rad=0.1',
    
    # Layout
    layout_iterations=100,
    scaling_ratio=10,
    dissuade_hubs=True,
    
    # Labels
    show_labels=True,
    font_size=8,
    
    # Legend/Colorbar
    show_colorbar=None,  # None = auto (True if node colors are numeric)
    colorbar_label=None,  # Auto-generated if None
    show_sentiment_legend=None,  # None = auto (True if edge_color_criterion='sentiment')
    
    # Output
    title="Network Visualization",
    file_name="network",
    figsize=(15, 12),
    dpi=300,
    save=True
):
    # Handle empty graph
    if G is None or len(G.nodes) == 0:
        print("Graph is empty or not provided.")
        return None
    
    # Create sentiment network if relationship_data is provided
    if relationship_data is not None:
        target_type = graph_type if graph_type else "undirected"
        G = create_sentiment_network(G, relationship_data, type=target_type)
        # Default to sentiment-based edge coloring
        if edge_color_criterion == 'fixed':
            edge_color_criterion = 'sentiment'
    
    # Remove isolated nodes if requested
    if remove_isolates:
        isolates = list(nx.isolates(G))
        if isolates:
            G = G.copy()  # Don't modify the original graph
            G.remove_nodes_from(isolates)
            if len(G.nodes) == 0:
                print("All nodes were isolated. Nothing to visualize.")
                return None
    
    # Auto-detect graph type
    is_directed = G.is_directed() if graph_type is None else (graph_type == "directed")
    
    # Auto-detect arrow display
    if show_arrows is None:
        show_arrows = is_directed
    
    # Compute node sizes (needed for layout)
    node_sizes, node_size_dict = _compute_node_sizes(
        G, node_size_criterion, node_size_scale, edge_weight_attr
    )
    
    # Compute node colors
    node_colors, colors_are_numeric = _compute_node_colors(
        G, node_color_criterion, node_fixed_color, edge_weight_attr
    )
    
    # Compute edge colors and widths
    edge_colors = _compute_edge_colors(G, edge_color_criterion, edge_fixed_color)
    edge_widths = _compute_edge_widths(G, edge_width_scale, edge_weight_attr)
    
    # Compute layout
    pos = nx.forceatlas2_layout(
        G,
        max_iter=layout_iterations,
        scaling_ratio=scaling_ratio,
        node_mass={node: node_size_dict[node] for node in G.nodes()},
        node_size={node: node_size_dict[node] for node in G.nodes()},
        dissuade_hubs=dissuade_hubs,
        weight='weight',
    )
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Draw nodes
    nodes_artist = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=node_cmap if colors_are_numeric else None,
        alpha=node_alpha,
        edgecolors=edge_color, 
        linewidths=edge_linewidth
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=edge_alpha,
        arrows=show_arrows,
        arrowstyle=arrow_style if show_arrows else '-',
        arrowsize=arrow_size,
        node_size=node_sizes,
        connectionstyle=connection_style
    )
    
    # Draw labels
    if show_labels:
        nx.draw_networkx_labels(
            G, pos,
            font_size=font_size,
            font_family='sans-serif',
            bbox=dict(facecolor="white", edgecolor='none', alpha=0.7, pad=0.5)
        )
    
    # Title
    plt.title(title, fontsize=20 if figsize[0] >= 12 else 14, pad=20)
    
    # Colorbar (auto or explicit)
    if show_colorbar is None:
        show_colorbar = colors_are_numeric
    
    if show_colorbar and colors_are_numeric and nodes_artist is not None:
        cbar = plt.colorbar(nodes_artist)
        if colorbar_label is None:
            # Auto-generate label
            label_map = {
                'degree': 'Degree',
                'in_degree': 'In-Degree',
                'out_degree': 'Out-Degree',
                'weighted_degree': 'Weighted Degree',
                'weighted_in_degree': 'Weighted In-Degree',
                'weighted_out_degree': 'Weighted Out-Degree',
                'charisma': 'Charisma'
            }
            colorbar_label = label_map.get(node_color_criterion, node_color_criterion.replace('_', ' ').title())
        cbar.set_label(colorbar_label, rotation=270, labelpad=15)
    
    # Sentiment legend (auto or explicit)
    if show_sentiment_legend is None:
        show_sentiment_legend = (edge_color_criterion == 'sentiment')
    
    if show_sentiment_legend:
        legend_elements = [
            Patch(facecolor='green', label='Mutual positive (love)'),
            Patch(facecolor='red', label='Mutual negative (hate)'),
            Patch(facecolor='purple', label='Asymmetric (conflict)'),
            Patch(facecolor='grey', label='Neutral')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    if save:
        if not os.path.exists('images'):
            os.makedirs('images')
        output_path = f'images/{file_name}.png'
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)
    
    return G


def _compute_degree_data(G):
    """
    Extract degree sequences and compute statistics for all degree types.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph to analyze
        
    Returns:
    --------
    dict : Dictionary with 'in', 'out', 'total' keys, each containing:
           - 'raw': full degree sequence
           - 'nonzero': degrees > 0 (for log-log plotting)
           - 'mean': mean degree
           - 'alpha': power-law exponent (fitted)
           - 'xmin': power-law xmin (fitted)
    """
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    total_degrees = [d for n, d in G.degree()]
    
    # Nonzero for log-log
    in_nonzero = [d for d in in_degrees if d > 0]
    out_nonzero = [d for d in out_degrees if d > 0]
    total_nonzero = [d for d in total_degrees if d > 0]
    
    # Fit power laws
    in_alpha, in_xmin = fit_powerlaw(in_nonzero) if in_nonzero else (0, 1)
    out_alpha, out_xmin = fit_powerlaw(out_nonzero) if out_nonzero else (0, 1)
    total_alpha, total_xmin = fit_powerlaw(total_nonzero) if total_nonzero else (0, 1)
    
    return {
        'in': {
            'raw': in_degrees,
            'nonzero': in_nonzero,
            'mean': np.mean(in_degrees),
            'alpha': in_alpha,
            'xmin': in_xmin,
            'label': 'In-Degree',
            'xlabel': 'In-Degree (k)'
        },
        'out': {
            'raw': out_degrees,
            'nonzero': out_nonzero,
            'mean': np.mean(out_degrees),
            'alpha': out_alpha,
            'xmin': out_xmin,
            'label': 'Out-Degree',
            'xlabel': 'Out-Degree (k)'
        },
        'total': {
            'raw': total_degrees,
            'nonzero': total_nonzero,
            'mean': np.mean(total_degrees),
            'alpha': total_alpha,
            'xmin': total_xmin,
            'label': 'Degree',
            'xlabel': 'Degree (k)'
        }
    }


def _plot_linear_histogram(ax, degrees, mean_degree, title=None, xlabel=None, show_ylabel=True, color='#3498DB'):
    """
    Plot linear scale histogram of degree distribution with accurate binning.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    degrees : list
        Degree sequence
    mean_degree : float
        Mean degree for vertical line
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    show_ylabel : bool
        Whether to show y-axis label
    color : str
        Bar color
    """
    # Fix: Use proper discrete binning for integer degree values
    min_deg, max_deg = min(degrees), max(degrees)
    bins = np.arange(min_deg, max_deg + 2) - 0.5  # Center bars on integers
    
    ax.hist(degrees, bins=bins, align='mid', rwidth=0.85, 
            color=color, edgecolor='white', alpha=0.8)
    ax.axvline(mean_degree, color='#E74C3C', linestyle='--', linewidth=2,
               label=f'Mean: {mean_degree:.2f}')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Number of Nodes", fontsize=11)
    
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)


def _plot_loglog_distribution(ax, degrees, alpha, xmin, title=None, xlabel=None, show_ylabel=True):
    """
    Plot log-log scale distribution with power-law fit.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    degrees : list
        Degree sequence (should be nonzero values)
    alpha : float
        Power-law exponent
    xmin : float
        Power-law xmin
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    show_ylabel : bool
        Whether to show y-axis label
    """
    scatter_color = "#3498DB"
    fit_color = "#E74C3C"
    
    # Calculate frequency distribution
    unique, counts = np.unique(degrees, return_counts=True)
    freq = counts / len(degrees)
    
    # Scatter plot of empirical distribution
    ax.scatter(unique, freq, s=60, c=scatter_color, alpha=0.7, 
               edgecolors='white', linewidth=0.5, label='Empirical', zorder=3)
    
    # Power-law fit line
    k_fit = np.linspace(xmin, max(unique), 100)
    C = freq[unique >= xmin][0] * (xmin ** alpha) if np.any(unique >= xmin) else 1
    p_fit = C * k_fit ** (-alpha)
    
    ax.plot(k_fit, p_fit, color=fit_color, linewidth=2.5, 
            label=f'Power-law fit (α = {alpha:.2f})', zorder=2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if show_ylabel:
        ax.set_ylabel("P(k)", fontsize=11)
    
    ax.legend(fontsize=10, framealpha=0.9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=10)


def plot_combined_degree_distributions(
    G, 
    layout="stacked", 
    show=None, 
    file_name="combined-degree-distributions"
):
    """
    Plot degree distributions combining linear and log-log scales.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        The directed graph to analyze
    layout : str
        "overlay" - overlays linear (left/bottom axes) and loglog (top/right axes) on same plot
        "stacked" - two rows: first for linear, second for log-log
    show : dict
        Dictionary specifying which degree types to show:
        {'in': bool, 'out': bool, 'total': bool}
        Defaults to {'in': True, 'out': True, 'total': True}
    file_name : str
        Name for saving the figure
        
    Returns:
    --------
    dict : Computed degree data including power-law exponents
    """
    if show is None:
        show = {'in': True, 'out': True, 'total': True}
    
    # Compute all degree data
    degree_data = _compute_degree_data(G)
    
    # Determine which plots to create
    active_types = [key for key in ['in', 'out', 'total'] if show.get(key, False)]
    n_plots = len(active_types)
    
    if n_plots == 0:
        print("No degree types selected. Please set at least one of 'in', 'out', or 'total' to True.")
        return degree_data
    
    if layout == "overlay":
        # Create n_plots x 1 grid, each with dual axes
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle("Degree Distributions (Linear & Log-Log)", fontsize=18, fontweight='bold', y=1.02)
        
        # Store overlay axes for later positioning (after tight_layout)
        overlay_axes_data = []
        
        for i, deg_type in enumerate(active_types):
            data = degree_data[deg_type]
            ax_linear = axes[i]
            
            # Plot linear histogram on primary axes (left/bottom)
            _plot_linear_histogram(
                ax_linear, 
                data['raw'], 
                data['mean'],
                title=f"{data['label']} Distribution",
                xlabel=data['xlabel'],
                show_ylabel=(i == 0),
                color='#3498DB'
            )
            
            # Style linear axes - keep left and bottom visible
            ax_linear.spines['top'].set_visible(False)
            ax_linear.spines['right'].set_visible(False)
            
            # Store data for overlay axes (created after tight_layout)
            overlay_axes_data.append((i, ax_linear, data, deg_type))
        
        # Apply tight_layout first so positions are finalized
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)  # Make room for suptitle
        
        # Now create overlay axes for log-log at the exact same positions
        for i, ax_linear, data, deg_type in overlay_axes_data:
            degrees = data['nonzero']
            if len(degrees) == 0:
                continue
                
            # Create overlay axes with independent x and y scales
            pos = ax_linear.get_position()
            ax_log = fig.add_axes(pos, frame_on=False)
            
            # Set log-log scales
            ax_log.set_xscale('log')
            ax_log.set_yscale('log')
            
            # Make background transparent
            ax_log.patch.set_visible(False)
            
            # Move ticks and labels to top and right
            ax_log.xaxis.tick_top()
            ax_log.xaxis.set_label_position('top')
            ax_log.yaxis.tick_right()
            ax_log.yaxis.set_label_position('right')
            
            # Calculate frequency distribution
            unique, counts = np.unique(degrees, return_counts=True)
            freq = counts / len(degrees)
            
            # Scatter plot on log-log axes
            ax_log.scatter(unique, freq, s=40, c='#E74C3C', alpha=0.7,
                          marker='o', edgecolors='white', linewidth=0.3,
                          label='Log-log empirical', zorder=4)
            
            # Power-law fit line
            k_fit = np.linspace(data['xmin'], max(unique), 100)
            C = freq[unique >= data['xmin']][0] * (data['xmin'] ** data['alpha']) if np.any(unique >= data['xmin']) else 1
            p_fit = C * k_fit ** (-data['alpha'])
            ax_log.plot(k_fit, p_fit, color='#9B59B6', linewidth=2, linestyle='-',
                       label=f'Power-law (α={data["alpha"]:.2f})', zorder=3)
            
            # Style the log-log axes
            ax_log.spines['bottom'].set_visible(False)
            ax_log.spines['left'].set_visible(False)
            ax_log.spines['top'].set_color('#E74C3C')
            ax_log.spines['right'].set_color('#E74C3C')
            
            # Set tick colors
            ax_log.tick_params(axis='x', colors='#E74C3C', labelsize=9)
            ax_log.tick_params(axis='y', colors='#E74C3C', labelsize=9)
            
            # Labels
            ax_log.set_xlabel(f"{data['xlabel']} [log]", fontsize=10, color='#E74C3C')
            if i == n_plots - 1:
                ax_log.set_ylabel("P(k) [log]", fontsize=10, color='#E74C3C')
                ax_log.legend(fontsize=8, loc='upper right', framealpha=0.9)
            
            # Set reasonable axis limits for log scale
            ax_log.set_xlim(0.8, max(unique) * 1.5)
            ax_log.set_ylim(min(freq) * 0.5, max(freq) * 2)
    
    elif layout == "stacked":
        # Create 2 x n_plots grid
        fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 10))
        if n_plots == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle("Degree Distributions", fontsize=18, fontweight='bold', y=1.02)
        
        for i, deg_type in enumerate(active_types):
            data = degree_data[deg_type]
            
            # Row 0: Linear histogram
            ax_linear = axes[0, i]
            _plot_linear_histogram(
                ax_linear, 
                data['raw'], 
                data['mean'],
                title=f"{data['label']} Distribution (Linear)",
                xlabel=data['xlabel'] if n_plots <= 2 else None,
                show_ylabel=(i == 0)
            )
            
            # Row 1: Log-log distribution
            ax_log = axes[1, i]
            if len(data['nonzero']) > 0:
                _plot_loglog_distribution(
                    ax_log,
                    data['nonzero'],
                    data['alpha'],
                    data['xmin'],
                    title=f"{data['label']} Distribution (Log-Log)",
                    xlabel=data['xlabel'],
                    show_ylabel=(i == 0)
                )
            else:
                ax_log.text(0.5, 0.5, 'No nonzero degrees', ha='center', va='center', 
                           transform=ax_log.transAxes, fontsize=12)
                ax_log.set_title(f"{data['label']} Distribution (Log-Log)", fontsize=13, fontweight='bold', pad=12)
    
    else:
        raise ValueError(f"Unknown layout '{layout}'. Use 'overlay' or 'stacked'.")
    
    # Only call tight_layout for stacked (overlay handles it earlier to get correct positions)
    if layout == "stacked":
        plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return degree_data


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
    tuple : (in_alpha, out_alpha, alpha) - power-law exponents for all distributions
    """
    degree_data = _compute_degree_data(G)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Degree Distributions (Log-Log Scale)", fontsize=18, fontweight='bold', y=1.02)
    
    # Plot all three distributions
    for ax, deg_type in zip([ax1, ax2, ax3], ['in', 'out', 'total']):
        data = degree_data[deg_type]
        if len(data['nonzero']) > 0:
            _plot_loglog_distribution(
                ax,
                data['nonzero'],
                data['alpha'],
                data['xmin'],
                title=f"{data['label']} Distribution",
                xlabel=data['xlabel'],
                show_ylabel=True
            )
    
    plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return degree_data['in']['alpha'], degree_data['out']['alpha'], degree_data['total']['alpha']


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
    degree_data = _compute_degree_data(G)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Degree Distributions", fontsize=18, fontweight='bold', y=1.02)
    
    # Plot all three distributions
    for ax, deg_type in zip([ax1, ax2, ax3], ['in', 'out', 'total']):
        data = degree_data[deg_type]
        _plot_linear_histogram(
            ax,
            data['raw'],
            data['mean'],
            title=f"{data['label']} Distribution",
            xlabel=data['xlabel'],
            show_ylabel=True
        )
    
    plt.tight_layout()
    
    if not os.path.exists('images'):
        os.makedirs('images')
    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return degree_data['in']['raw'], degree_data['out']['raw'], degree_data['total']['raw']


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
    num_bins = max_dialogues - min_dialogues + 1 if max_dialogues > min_dialogues else 1
    
    # Plot histogram
    fig = plt.figure(figsize=(12, 6))
    plt.hist(dialogue_counts, bins=num_bins, align='left', rwidth=1, edgecolor='black')
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