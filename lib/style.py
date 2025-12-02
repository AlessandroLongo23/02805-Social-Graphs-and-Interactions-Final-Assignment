"""
Styling utilities for ASoIaF Network Analysis.
Provides consistent styling functions and decorators for all visualizations.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable
from functools import wraps

from lib.config import (
    PLOT_CONFIG, FIGURE_SIZES, NETWORK_CONFIG, PALETTE,
    IceAndFirePalette, Paths
)


# =============================================================================
# DIRECTORY UTILITIES
# =============================================================================

def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path or file path (will extract directory)
    
    Returns:
        The directory path
    """
    directory = os.path.dirname(path) if '.' in os.path.basename(path) else path
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_output_path(filename: str, subdirectory: str = "") -> str:
    """
    Get full output path for a file, ensuring directory exists.
    
    Args:
        filename: Base filename (without extension or with .png)
        subdirectory: Optional subdirectory within images/
    
    Returns:
        Full path to the output file
    """
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    if subdirectory:
        path = os.path.join(Paths.IMAGES_DIR, subdirectory, filename)
    else:
        path = os.path.join(Paths.IMAGES_DIR, filename)
    
    ensure_directory(path)
    return path


# =============================================================================
# FIGURE MANAGEMENT
# =============================================================================

class Figure:
    """Context manager for creating and saving figures with consistent styling."""
    
    def __init__(
        self,
        filename: str,
        figsize: Tuple[int, int] = FIGURE_SIZES.MEDIUM,
        subdirectory: str = "",
        show: bool = True,
        save: bool = True,
        tight_layout: bool = True
    ):
        """
        Initialize figure context manager.
        
        Args:
            filename: Output filename (without path)
            figsize: Figure size tuple (width, height)
            subdirectory: Subdirectory within images/
            show: Whether to display the figure
            save: Whether to save the figure
            tight_layout: Whether to apply tight_layout
        """
        self.filename = filename
        self.figsize = figsize
        self.subdirectory = subdirectory
        self.show = show
        self.save = save
        self.tight_layout = tight_layout
        self.fig = None
        self.ax = None
    
    def __enter__(self):
        """Create and return figure and axes."""
        self.fig = plt.figure(figsize=self.figsize)
        return self.fig
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle saving and cleanup."""
        if exc_type is not None:
            plt.close(self.fig)
            return False
        
        if self.tight_layout:
            plt.tight_layout()
        
        if self.save:
            output_path = get_output_path(self.filename, self.subdirectory)
            self.fig.savefig(output_path, dpi=PLOT_CONFIG.figure_dpi, 
                           bbox_inches='tight', facecolor=self.fig.get_facecolor())
        
        if self.show:
            plt.show()
        
        plt.close(self.fig)
        return False


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[int, int] = None,
    sharex: bool = False,
    sharey: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create subplots with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (auto-calculated if None)
        sharex: Share x axis
        sharey: Share y axis
    
    Returns:
        Tuple of (figure, axes array)
    """
    if figsize is None:
        width = 6 * ncols
        height = 5 * nrows
        figsize = (width, height)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axes


# =============================================================================
# AXIS STYLING
# =============================================================================

def style_axis(
    ax: plt.Axes,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    remove_spines: bool = True,
    grid: bool = True,
    grid_axis: str = 'both'
) -> plt.Axes:
    """
    Apply consistent styling to an axis.
    
    Args:
        ax: Matplotlib axes object
        title: Optional title
        xlabel: Optional x-axis label
        ylabel: Optional y-axis label
        remove_spines: Whether to remove top/right spines
        grid: Whether to show grid
        grid_axis: Which axis to show grid on ('x', 'y', or 'both')
    
    Returns:
        The styled axes object
    """
    if title:
        ax.set_title(title, fontsize=PLOT_CONFIG.subtitle_font_size, 
                    fontweight='bold', pad=12, color=PLOT_CONFIG.title_color)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG.label_font_size)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG.label_font_size)
    
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    if grid:
        ax.grid(True, alpha=PLOT_CONFIG.grid_alpha, 
               linestyle=PLOT_CONFIG.grid_style, axis=grid_axis)
    
    ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG.tick_font_size)
    
    return ax


def add_title(
    fig_or_ax,
    title: str,
    subtitle: str = None,
    pad: int = 20
):
    """
    Add a styled title (and optional subtitle) to a figure or axes.
    
    Args:
        fig_or_ax: Figure or Axes object
        title: Main title text
        subtitle: Optional subtitle text
        pad: Padding below title
    """
    if isinstance(fig_or_ax, plt.Figure):
        if subtitle:
            full_title = f"{title}\n{subtitle}"
        else:
            full_title = title
        fig_or_ax.suptitle(
            full_title,
            fontsize=PLOT_CONFIG.title_font_size,
            fontweight=PLOT_CONFIG.title_weight,
            color=PLOT_CONFIG.title_color,
            y=1.02
        )
    else:
        fig_or_ax.set_title(
            title,
            fontsize=PLOT_CONFIG.subtitle_font_size,
            fontweight='bold',
            pad=pad,
            color=PLOT_CONFIG.title_color
        )


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def get_color_gradient(
    values: List[float],
    cmap_name: str = None,
    vmin: float = None,
    vmax: float = None
) -> List[Tuple[float, float, float, float]]:
    """
    Map values to colors using a colormap.
    
    Args:
        values: List of numeric values
        cmap_name: Colormap name (default from config)
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
    
    Returns:
        List of RGBA color tuples
    """
    if cmap_name is None:
        cmap_name = PLOT_CONFIG.sequential_cmap
    
    cmap = plt.cm.get_cmap(cmap_name)
    
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    
    if vmax == vmin:
        return [cmap(0.5) for _ in values]
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    return [cmap(norm(v)) for v in values]


def get_sentiment_color(sentiment: float, intensity: float = 1.0) -> str:
    """
    Get color for a sentiment value.
    
    Args:
        sentiment: Sentiment value (-1 to 1)
        intensity: Color intensity multiplier
    
    Returns:
        Hex color string
    """
    if sentiment > 0.05:
        base_color = np.array(mcolors.to_rgb(PALETTE.LOVE_GREEN))
    elif sentiment < -0.05:
        base_color = np.array(mcolors.to_rgb(PALETTE.HATE_RED))
    else:
        base_color = np.array(mcolors.to_rgb(PALETTE.NEUTRAL_GRAY))
    
    # Adjust intensity
    white = np.array([1.0, 1.0, 1.0])
    adjusted = base_color * intensity + white * (1 - intensity)
    adjusted = np.clip(adjusted, 0, 1)
    
    return mcolors.to_hex(adjusted)


def get_bidirectional_sentiment_color(
    sentiment_ab: float,
    sentiment_ba: float
) -> Tuple[Tuple[float, float, float], float]:
    """
    Get color and width for bidirectional sentiment edge.
    
    Args:
        sentiment_ab: Sentiment from A to B
        sentiment_ba: Sentiment from B to A
    
    Returns:
        Tuple of (RGB color, edge width)
    """
    c_love = np.array(mcolors.to_rgb(PALETTE.LOVE_GREEN))
    c_hate = np.array(mcolors.to_rgb(PALETTE.HATE_RED))
    c_conflict = np.array(mcolors.to_rgb(PALETTE.CONFLICT_PURPLE))
    
    # Calculate forces
    force_love = max(0, sentiment_ab + sentiment_ba)
    force_hate = max(0, -(sentiment_ab + sentiment_ba))
    force_conflict = abs(sentiment_ab - sentiment_ba)
    
    total_force = force_love + force_hate + force_conflict
    
    if total_force == 0:
        final_color = mcolors.to_rgb(PALETTE.NEUTRAL_GRAY)
    else:
        mixed_rgb = (force_love * c_love + 
                    force_hate * c_hate + 
                    force_conflict * c_conflict) / total_force
        final_color = tuple(mixed_rgb)
    
    # Calculate width
    raw_magnitude = np.sqrt(sentiment_ab ** 2 + sentiment_ba ** 2)
    final_width = 1 + (raw_magnitude * 5)
    
    return final_color, final_width


def create_sentiment_legend() -> List[Patch]:
    """Create legend elements for sentiment visualization."""
    return [
        Patch(facecolor=PALETTE.LOVE_GREEN, label='Mutual positive', 
              edgecolor=PALETTE.NIGHT_BLACK, linewidth=0.5),
        Patch(facecolor=PALETTE.HATE_RED, label='Mutual negative',
              edgecolor=PALETTE.NIGHT_BLACK, linewidth=0.5),
        Patch(facecolor=PALETTE.CONFLICT_PURPLE, label='Asymmetric (conflict)',
              edgecolor=PALETTE.NIGHT_BLACK, linewidth=0.5),
        Patch(facecolor=PALETTE.NEUTRAL_GRAY, label='Neutral',
              edgecolor=PALETTE.NIGHT_BLACK, linewidth=0.5)
    ]


# =============================================================================
# NODE SIZE UTILITIES
# =============================================================================

def calculate_node_sizes(
    G,
    criterion: str = 'degree',
    min_size: int = None,
    multiplier: int = None
) -> Dict[str, int]:
    """
    Calculate node sizes based on various criteria.
    
    Args:
        G: NetworkX graph
        criterion: One of 'degree', 'in_degree', 'out_degree', 
                   'weighted_in_degree', 'weighted_out_degree'
        min_size: Minimum node size
        multiplier: Size multiplier
    
    Returns:
        Dictionary mapping node to size
    """
    if min_size is None:
        min_size = NETWORK_CONFIG.min_node_size
    if multiplier is None:
        multiplier = NETWORK_CONFIG.node_size_multiplier
    
    if criterion == 'degree':
        values = dict(G.degree())
    elif criterion == 'in_degree':
        values = dict(G.in_degree())
    elif criterion == 'out_degree':
        values = dict(G.out_degree())
    elif criterion == 'weighted_in_degree':
        values = dict(G.in_degree(weight='normalized_weight'))
    elif criterion == 'weighted_out_degree':
        values = dict(G.out_degree(weight='normalized_weight'))
    else:
        values = {node: 1 for node in G.nodes()}
    
    return {node: int((value + 1) * multiplier) for node, value in values.items()}


def normalize_weights(G) -> List[float]:
    """
    Normalize edge weights to [0, 1] range.
    
    Args:
        G: NetworkX graph
    
    Returns:
        List of normalized weights
    """
    weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
    
    if not weights:
        return []
    
    max_weight = max(weights)
    if max_weight == 0:
        return [0.5 for _ in weights]
    
    normalized = [w / max_weight for w in weights]
    
    # Also update the graph
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        data['normalized_weight'] = normalized[i]
    
    return normalized


# =============================================================================
# COLORBAR UTILITIES
# =============================================================================

def add_colorbar(
    mappable,
    ax: plt.Axes = None,
    label: str = "",
    orientation: str = "vertical"
) -> plt.colorbar:
    """
    Add a styled colorbar to a plot.
    
    Args:
        mappable: The mappable object (e.g., ScalarMappable, image)
        ax: Axes to attach colorbar to
        label: Colorbar label
        orientation: 'vertical' or 'horizontal'
    
    Returns:
        Colorbar object
    """
    cbar = plt.colorbar(mappable, ax=ax, orientation=orientation)
    cbar.set_label(label, rotation=270 if orientation == 'vertical' else 0,
                  labelpad=15, fontsize=PLOT_CONFIG.label_font_size)
    cbar.ax.tick_params(labelsize=PLOT_CONFIG.tick_font_size)
    return cbar


# =============================================================================
# HISTOGRAM STYLING
# =============================================================================

def style_histogram(
    ax: plt.Axes,
    data: List[float],
    bins: int = 40,
    color: str = None,
    alpha: float = 0.7,
    edgecolor: str = None,
    show_mean: bool = True,
    show_median: bool = False,
    label: str = None
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Create a styled histogram.
    
    Args:
        ax: Axes to plot on
        data: Data to histogram
        bins: Number of bins
        color: Bar color
        alpha: Bar transparency
        edgecolor: Edge color
        show_mean: Whether to show mean line
        show_median: Whether to show median line
        label: Optional label for legend
    
    Returns:
        Tuple of (counts, bins, patches)
    """
    if color is None:
        color = PLOT_CONFIG.primary_color
    if edgecolor is None:
        edgecolor = PALETTE.NIGHT_BLACK
    
    n, bins_out, patches = ax.hist(
        data, bins=bins, align='left', rwidth=0.85,
        color=color, alpha=alpha, edgecolor=edgecolor,
        linewidth=0.5, label=label
    )
    
    if show_mean:
        mean_val = np.mean(data)
        ax.axvline(mean_val, color=PALETTE.DRAGON_RED, linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_val:.2f}')
    
    if show_median:
        median_val = np.median(data)
        ax.axvline(median_val, color=PALETTE.WINTER_BLUE, linestyle='--',
                  linewidth=2, label=f'Median: {median_val:.2f}')
    
    return n, bins_out, patches


# =============================================================================
# BAR CHART STYLING
# =============================================================================

def add_bar_labels(
    ax: plt.Axes,
    bars,
    scores: List[float],
    horizontal: bool = True,
    format_str: str = "{:.3f}",
    offset_ratio: float = 0.02
):
    """
    Add value labels to bar chart.
    
    Args:
        ax: Axes object
        bars: Bar container
        scores: Score values to display
        horizontal: Whether bars are horizontal
        format_str: Format string for values
        offset_ratio: Label offset as ratio of max score
    """
    max_score = max(scores) if scores else 1
    
    for bar, score in zip(bars, scores):
        if horizontal:
            width = bar.get_width()
            ax.text(
                width + max_score * offset_ratio,
                bar.get_y() + bar.get_height() / 2,
                format_str.format(score),
                va='center', ha='left',
                fontsize=PLOT_CONFIG.annotation_font_size,
                fontweight='bold',
                color=PLOT_CONFIG.title_color
            )
        else:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max_score * offset_ratio,
                format_str.format(score),
                ha='center', va='bottom',
                fontsize=PLOT_CONFIG.annotation_font_size,
                fontweight='bold',
                color=PLOT_CONFIG.title_color
            )

