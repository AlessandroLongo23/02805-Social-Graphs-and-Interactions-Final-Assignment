import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
import os
from wordcloud import WordCloud

def extract_community_dialogues(all_dialogues, partition):
    """
    Extract all dialogues for each community.
    """
    community_dialogues = defaultdict(list)
    
    for index, row in all_dialogues.iterrows():
        speaker_raw = row['Speaker(s)']
        dialogue = row['Dialogue']
        
        if pd.notna(speaker_raw) and pd.notna(dialogue):
            speakers = [s.strip() for s in str(speaker_raw).split(';')]
            
            for speaker in speakers:
                if speaker in partition:
                    comm_id = partition[speaker]
                    community_dialogues[comm_id].append(str(dialogue))
    
    # Combine all dialogues per community
    community_texts = {}
    for comm_id, dialogues in community_dialogues.items():
        community_texts[comm_id] = ' '.join(dialogues)
    
    print(f"Extracted dialogues for {len(community_texts)} communities")
    for comm_id, text in community_texts.items():
        print(f"  Community {comm_id}: {len(text.split())} words")
    
    return community_texts


def compute_community_tfidf(community_texts, max_features=20):
    """
    Compute TF-IDF scores for each community's dialogue.
    """
    # Prepare documents
    comm_ids = sorted(community_texts.keys())
    documents = [community_texts[comm_id] for comm_id in comm_ids]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        min_df=1,
        ngram_range=(1, 2)  # Include bigrams
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top terms for each community
    community_top_terms = {}
    
    for i, comm_id in enumerate(comm_ids):
        # Get TF-IDF scores for this community
        scores = tfidf_matrix[i].toarray()[0]
        
        # Get top terms
        top_indices = scores.argsort()[-max_features:][::-1]
        top_terms = [(feature_names[idx], scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        community_top_terms[comm_id] = top_terms
    
    return community_top_terms


def create_community_wordclouds(community_tfidf, partition, n_cols=3, title="Community Characteristic Terms (TF-IDF)", file_name="community-characteristic-terms-tfidf"):
    """
    Create word clouds for each community based on TF-IDF scores.
    """
    communities = sorted(community_tfidf.keys())
    n_communities = len(communities)
    n_rows = (n_communities + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    
    if n_communities == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    colormaps = ['plasma', 'viridis', 'inferno', 'magma', 'cividis', 
                 'twilight_shifted', 'RdPu', 'YlOrRd']
    
    for i, comm_id in enumerate(communities):
        # Get member names for subtitle
        members = [node for node, comm in partition.items() if comm == comm_id]
        member_str = ', '.join(members[:3])
        if len(members) > 3:
            member_str += f" (+{len(members)-3} more)"
        
        # Prepare word frequencies
        tfidf_dict = {term: score for term, score in community_tfidf[comm_id]}
        
        if not tfidf_dict:
            axes[i].text(0.5, 0.5, f'Community {comm_id}\n(No data)', 
                        ha='center', va='center', fontsize=14)
            axes[i].axis('off')
            continue
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color='white',
            colormap=colormaps[i % len(colormaps)],
            max_words=50,
            relative_scaling=0.5,
            collocations=False
        ).generate_from_frequencies(tfidf_dict)
        
        # Plot
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Community {comm_id}\n{member_str}', 
                         fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
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