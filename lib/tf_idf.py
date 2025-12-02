"""
TF-IDF analysis for ASoIaF Network Analysis.
Provides term frequency analysis and visualization for community dialogues.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from typing import Dict, List, Tuple, Optional

from lib.config import PLOT_CONFIG, PALETTE
from lib.style import Figure


# =============================================================================
# DIALOGUE EXTRACTION
# =============================================================================

def extract_community_dialogues(
    all_dialogues: pd.DataFrame,
    partition: Dict[str, int]
) -> Dict[int, str]:
    """
    Extract all dialogues for each community.
    
    Args:
        all_dialogues: DataFrame with dialogue data
        partition: Dict mapping nodes to community IDs
    
    Returns:
        Dict mapping community IDs to concatenated dialogue text
    """
    community_dialogues = defaultdict(list)
    
    for _, row in all_dialogues.iterrows():
        speaker_raw = row['Speaker(s)']
        dialogue = row['Dialogue']
        
        if pd.notna(speaker_raw) and pd.notna(dialogue):
            speakers = [s.strip() for s in str(speaker_raw).split(';')]
            
            for speaker in speakers:
                if speaker in partition:
                    comm_id = partition[speaker]
                    community_dialogues[comm_id].append(str(dialogue))
    
    # Combine dialogues per community
    community_texts = {
        comm_id: ' '.join(dialogues)
        for comm_id, dialogues in community_dialogues.items()
    }
    
    print(f"Extracted dialogues for {len(community_texts)} communities")
    for comm_id, text in sorted(community_texts.items()):
        print(f"  Community {comm_id}: {len(text.split())} words")
    
    return community_texts


# =============================================================================
# TF-IDF COMPUTATION
# =============================================================================

def compute_community_tfidf(
    community_texts: Dict[int, str],
    max_features: int = 20,
    min_df: int = 1,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute TF-IDF scores for each community's dialogue.
    
    Args:
        community_texts: Dict mapping community IDs to text
        max_features: Maximum number of features per community
        min_df: Minimum document frequency
        ngram_range: Range of n-grams to extract
    
    Returns:
        Dict mapping community IDs to list of (term, score) tuples
    """
    comm_ids = sorted(community_texts.keys())
    documents = [community_texts[comm_id] for comm_id in comm_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=min_df,
        ngram_range=ngram_range
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top terms for each community
    community_top_terms = {}
    
    for i, comm_id in enumerate(comm_ids):
        scores = tfidf_matrix[i].toarray()[0]
        top_indices = scores.argsort()[-max_features:][::-1]
        top_terms = [
            (feature_names[idx], scores[idx])
            for idx in top_indices if scores[idx] > 0
        ]
        community_top_terms[comm_id] = top_terms
    
    return community_top_terms


# =============================================================================
# WORD CLOUD VISUALIZATION
# =============================================================================

# Custom colormaps for word clouds - themed for ASoIaF
COMMUNITY_COLORMAPS = [
    'Reds',      # Fire (Targaryen)
    'Blues',     # Ice (Stark)
    'YlOrBr',    # Gold (Lannister)
    'Greens',    # Growth (Tyrell)
    'Purples',   # Royal (Baratheon)
    'Oranges',   # Sun (Martell)
    'GnBu',      # Sky (Arryn)
    'Greys',     # Iron (Iron Islands)
    'RdPu',      # Blood (Bolton)
    'BuGn',      # Swamp (Reed)
]


def create_community_wordclouds(
    community_tfidf: Dict[int, List[Tuple[str, float]]],
    partition: Dict[str, int],
    n_cols: int = 3,
    title: str = "Community Characteristic Terms (TF-IDF)",
    file_name: str = "community-characteristic-terms-tfidf",
    subdirectory: str = "community",
    show: bool = True
) -> None:
    """
    Create word clouds for each community based on TF-IDF scores.
    
    Args:
        community_tfidf: Dict mapping community IDs to term-score lists
        partition: Dict mapping nodes to community IDs
        n_cols: Number of columns in grid
        title: Plot title
        file_name: Output filename
        subdirectory: Subdirectory in images/
        show: Whether to display the plot
    """
    communities = sorted(community_tfidf.keys())
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    figsize = (n_cols * 6, n_rows * 5)
    
    with Figure(file_name, figsize, subdirectory, show=show) as fig:
        axes = []
        for i in range(n_communities):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            axes.append(ax)
        
        for i, comm_id in enumerate(communities):
            ax = axes[i]
            
            # Get member names for subtitle
            members = [node for node, comm in partition.items() if comm == comm_id]
            member_str = ', '.join(sorted(members, 
                                         key=lambda m: sum(1 for n, c in partition.items() 
                                                          if c == comm_id))[:3])
            if len(members) > 3:
                member_str += f" (+{len(members)-3} more)"
            
            # Prepare word frequencies
            tfidf_dict = {term: score for term, score in community_tfidf[comm_id]}
            
            if not tfidf_dict:
                ax.text(0.5, 0.5, f'Community {comm_id}\n(No data)',
                       ha='center', va='center', fontsize=14,
                       transform=ax.transAxes, color=PLOT_CONFIG.title_color)
                ax.axis('off')
                continue
            
            # Create word cloud with themed colormap
            colormap = COMMUNITY_COLORMAPS[i % len(COMMUNITY_COLORMAPS)]
            
            wordcloud = WordCloud(
                width=800,
                height=500,
                background_color='white',
                colormap=colormap,
                max_words=50,
                relative_scaling=0.5,
                collocations=False,
                prefer_horizontal=0.8,
                min_font_size=10
            ).generate_from_frequencies(tfidf_dict)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'Community {comm_id}\n{member_str}',
                        fontsize=11, fontweight='bold',
                        color=PLOT_CONFIG.title_color)
            ax.axis('off')
        
        # Hide empty axes
        for j in range(n_communities, n_rows * n_cols):
            if j < len(fig.get_axes()):
                fig.get_axes()[j].axis('off')
        
        plt.suptitle(title, fontsize=PLOT_CONFIG.title_font_size,
                    fontweight='bold', y=0.995, color=PLOT_CONFIG.title_color)


def print_tfidf_results(
    community_tfidf: Dict[int, List[Tuple[str, float]]],
    top_n: int = 15
) -> None:
    """
    Print TF-IDF results in a formatted manner.
    
    Args:
        community_tfidf: Dict mapping community IDs to term-score lists
        top_n: Number of top terms to display per community
    """
    print("=" * 70)
    print("TOP TF-IDF TERMS BY COMMUNITY")
    print("=" * 70)
    
    for comm_id in sorted(community_tfidf.keys()):
        print(f"\n{'='*70}")
        print(f"COMMUNITY {comm_id}")
        print("=" * 70)
        
        for rank, (term, score) in enumerate(community_tfidf[comm_id][:top_n], 1):
            print(f"  {rank:2d}. {term:<25} {score:.4f}")
