"""
Configuration module for ASoIaF Network Analysis.
Centralizes all styling, colors, and configuration options.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
# COLOR PALETTES
# =============================================================================

# Primary palette - inspired by the harsh winters and fire of ASoIaF
# Using a dark, atmospheric theme with warm accent colors

class IceAndFirePalette:
    """Color palette inspired by A Song of Ice and Fire."""
    
    # Base colors
    SNOW_WHITE = "#F8F9FA"
    NIGHT_BLACK = "#1A1A2E"
    IRON_GRAY = "#4A4E69"
    STEEL_GRAY = "#9A8C98"
    
    # Ice colors (cool tones)
    ICE_BLUE = "#A0CED9"
    FROST = "#CAF0F8"
    WINTER_BLUE = "#457B9D"
    STARK_GRAY = "#6C757D"
    
    # Fire colors (warm tones)
    DRAGON_RED = "#C1121F"
    EMBER = "#E5383B"
    GOLD = "#FFB703"
    AMBER = "#FB8500"
    TARGARYEN_BLACK = "#370617"
    
    # Sentiment colors
    LOVE_GREEN = "#06D6A0"
    HATE_RED = "#EF476F"
    CONFLICT_PURPLE = "#8338EC"
    NEUTRAL_GRAY = "#ADB5BD"
    
    # House colors for communities
    HOUSE_COLORS = [
        "#C1121F",  # Targaryen Red
        "#1D3557",  # Stark Blue-Gray
        "#FFB703",  # Lannister Gold
        "#2D6A4F",  # Tyrell Green
        "#7209B7",  # Baratheon Purple/Black
        "#F77F00",  # Martell Orange
        "#3A86FF",  # Arryn Blue
        "#8D99AE",  # Iron Islands Gray
        "#D62828",  # Bolton Pink-Red
        "#06D6A0",  # Reed Green
    ]
    
    # Gradient for degree/centrality
    GRADIENT_LOW = "#CAF0F8"
    GRADIENT_HIGH = "#C1121F"
    
    @classmethod
    def get_community_color(cls, index: int) -> str:
        """Get color for community by index (cycles through palette)."""
        return cls.HOUSE_COLORS[index % len(cls.HOUSE_COLORS)]
    
    @classmethod
    def get_community_colors(cls, n: int) -> List[str]:
        """Get n distinct community colors."""
        if n <= len(cls.HOUSE_COLORS):
            return cls.HOUSE_COLORS[:n]
        # If we need more colors, interpolate
        colors = cls.HOUSE_COLORS.copy()
        while len(colors) < n:
            idx = len(colors) % len(cls.HOUSE_COLORS)
            colors.append(cls.HOUSE_COLORS[idx])
        return colors[:n]


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    
    # Figure settings
    figure_dpi: int = 300
    background_color: str = IceAndFirePalette.SNOW_WHITE
    
    # Font settings
    font_family: str = "Georgia"
    title_font_size: int = 18
    subtitle_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    annotation_font_size: int = 9
    
    # Title styling
    title_weight: str = "bold"
    title_color: str = IceAndFirePalette.NIGHT_BLACK
    
    # Grid and axes
    grid_alpha: float = 0.3
    grid_style: str = "--"
    spine_color: str = IceAndFirePalette.IRON_GRAY
    spine_width: float = 1.0
    remove_top_spine: bool = True
    remove_right_spine: bool = True
    
    # Colors
    primary_color: str = IceAndFirePalette.DRAGON_RED
    secondary_color: str = IceAndFirePalette.WINTER_BLUE
    accent_color: str = IceAndFirePalette.GOLD
    
    # Network-specific
    node_alpha: float = 0.9
    edge_alpha: float = 0.4
    label_alpha: float = 0.85
    node_edge_color: str = IceAndFirePalette.NIGHT_BLACK
    node_edge_width: float = 1.5
    
    # Colormap
    diverging_cmap: str = "RdYlBu_r"  # Red-Yellow-Blue reversed (red = high)
    sequential_cmap: str = "YlOrRd"
    sentiment_cmap: str = "RdYlGn"


@dataclass
class FigureSizes:
    """Standard figure sizes for different plot types."""
    
    SMALL: Tuple[int, int] = (10, 8)
    MEDIUM: Tuple[int, int] = (14, 10)
    LARGE: Tuple[int, int] = (16, 12)
    WIDE: Tuple[int, int] = (18, 8)
    EXTRA_WIDE: Tuple[int, int] = (20, 8)
    SQUARE: Tuple[int, int] = (12, 12)
    HEATMAP: Tuple[int, int] = (14, 12)
    NETWORK: Tuple[int, int] = (16, 14)


@dataclass 
class NetworkConfig:
    """Configuration for network visualizations."""
    
    # Layout settings
    layout_iterations: int = 2000
    layout_scaling: float = 7.0
    layout_k: float = 0.66  # For spring layout
    
    # Node settings
    min_node_size: int = 100
    node_size_multiplier: int = 80
    
    # Edge settings
    min_edge_width: float = 0.5
    max_edge_width: float = 8.0
    arrow_size: int = 15
    arrow_style: str = "-|>"
    connection_style: str = "arc3,rad=0.1"
    
    # Label settings
    label_font_size: int = 8
    label_bbox_alpha: float = 0.8
    label_pad: float = 1.5


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Default configuration instances
PLOT_CONFIG = PlotConfig()
FIGURE_SIZES = FigureSizes()
NETWORK_CONFIG = NetworkConfig()
PALETTE = IceAndFirePalette()


# =============================================================================
# PATHS
# =============================================================================

class Paths:
    """Standard paths for the project."""
    DATA_DIR = "data"
    IMAGES_DIR = "images"
    IMAGES_BASIC = "images/basic"
    IMAGES_COMMUNITY = "images/community"
    IMAGES_SENTIMENT = "images/sentiment"


# =============================================================================
# MATPLOTLIB STYLE SETUP
# =============================================================================

def apply_global_style():
    """Apply global matplotlib style settings."""
    plt.rcParams.update({
        # Figure
        'figure.facecolor': PLOT_CONFIG.background_color,
        'figure.dpi': 100,
        'savefig.dpi': PLOT_CONFIG.figure_dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Fonts
        'font.family': 'serif',
        'font.serif': [PLOT_CONFIG.font_family, 'DejaVu Serif', 'Times New Roman'],
        'font.size': PLOT_CONFIG.tick_font_size,
        
        # Axes
        'axes.facecolor': PLOT_CONFIG.background_color,
        'axes.edgecolor': PLOT_CONFIG.spine_color,
        'axes.linewidth': PLOT_CONFIG.spine_width,
        'axes.titlesize': PLOT_CONFIG.title_font_size,
        'axes.titleweight': PLOT_CONFIG.title_weight,
        'axes.labelsize': PLOT_CONFIG.label_font_size,
        'axes.labelcolor': PLOT_CONFIG.title_color,
        'axes.titlecolor': PLOT_CONFIG.title_color,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': PLOT_CONFIG.grid_alpha,
        'grid.linestyle': PLOT_CONFIG.grid_style,
        
        # Ticks
        'xtick.labelsize': PLOT_CONFIG.tick_font_size,
        'ytick.labelsize': PLOT_CONFIG.tick_font_size,
        'xtick.color': PLOT_CONFIG.spine_color,
        'ytick.color': PLOT_CONFIG.spine_color,
        
        # Legend
        'legend.fontsize': PLOT_CONFIG.tick_font_size,
        'legend.framealpha': 0.9,
        'legend.edgecolor': PLOT_CONFIG.spine_color,
        
        # Lines
        'lines.linewidth': 2.0,
        
        # Patches (for bars, etc.)
        'patch.edgecolor': PLOT_CONFIG.spine_color,
    })


# Apply style on import
apply_global_style()

