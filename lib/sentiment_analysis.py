import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
import networkx as nx

def plot_heatmap(matrix, all_characters, remove_empty=False, title="Heatmap", file_name="heatmap"):
    fig = plt.figure(figsize=(14, 12))
    
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    x_characters = all_characters.copy()
    y_characters = all_characters.copy()
    
    if remove_empty:
        non_empty_rows = ~np.isnan(matrix).all(axis=1)
        non_empty_cols = ~np.isnan(matrix).all(axis=0)
        
        matrix = matrix[non_empty_rows, :][:, non_empty_cols]
        
        y_characters = [char for i, char in enumerate(all_characters) if non_empty_rows[i]]
        x_characters = [char for i, char in enumerate(all_characters) if non_empty_cols[i]]
    
    masked_matrix = np.ma.masked_invalid(matrix)
    min_matrix_value = np.min(masked_matrix)
    max_matrix_value = np.max(masked_matrix)
    
    ax = sns.heatmap(matrix, annot=False, cmap='coolwarm', center=0, vmin=min_matrix_value, vmax=max_matrix_value,
                     cbar_kws={'label': 'Adjusted Sentiment Score (Bayesian)'}, 
                     xticklabels=x_characters, yticklabels=y_characters,
                     mask=np.isnan(matrix))
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Addressee(s)', fontsize=12)
    plt.ylabel('Speaker(s)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = f'images/{file_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return matrix, x_characters, y_characters

def sentiment_heatmap(G, remove_empty=False, title="Character Relationship Sentiment Heatmap", file_name="sentiment-heatmap"):
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

    # 1. Extract dialogues from graph edges
    print("Extracting dialogues from graph edges...")
    dialogues_list = []  # List of tuples: (speaker, addressee, dialogue_text)
    
    for speaker, addressee, data in G.edges(data=True):
        dialogues = data.get('dialogues', [])
        
        for dialogue in dialogues:
            dialogues_list.append((speaker, addressee, dialogue))

    print(f"Extracted {len(dialogues_list)} dialogues from graph")

    # 2. Initialize VADER
    sid = SentimentIntensityAnalyzer()

    # 3. Define sentiment scoring function
    def get_sentiment(text):
        """Extract compound sentiment score from text using VADER."""
        try:
            if text is None or not str(text).strip():
                return 0.0
            return sid.polarity_scores(str(text))['compound']
        except Exception as e:
            print(f"Error processing text: {e}")
            return 0.0

    # 4. Compute sentiment scores for all dialogues
    print("Analyzing sentiment for all dialogues...")
    dialogue_scores = []  # List of tuples: (speaker, addressee, sentiment_score)
    all_scores = []
    
    for speaker, addressee, dialogue in dialogues_list:
        score = get_sentiment(dialogue)
        dialogue_scores.append((speaker, addressee, score))
        all_scores.append(score)

    # 5. Calculate global statistics for Bayesian Average
    global_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"Global average sentiment score (m): {global_mean:.4f}")

    # Group dialogues by character pair and compute statistics
    pair_data = {}  # {(speaker, addressee): [list of scores]}
    for speaker, addressee, score in dialogue_scores:
        pair_key = (speaker, addressee)
        if pair_key not in pair_data:
            pair_data[pair_key] = []
        pair_data[pair_key].append(score)

    # Calculate C: average number of dialogues per character pair
    total_dialogues = len(dialogue_scores)
    num_pairs = len(pair_data)
    C = total_dialogues / num_pairs if num_pairs > 0 else 1.0
    print(f"Confidence parameter (C): {C:.2f} (avg dialogues per pair)")

    # 6. Apply Bayesian Average (Damped Mean) to each pair
    # S_adj = (C * m + Σs) / (C + n)
    # where:
    #   C = confidence parameter (avg dialogues per pair)
    #   m = global average sentiment
    #   Σs = sum of sentiment scores for the pair
    #   n = number of dialogues for the pair
    
    adjusted_scores = {}  # {(speaker, addressee): adjusted_score}
    raw_scores = {}  # {(speaker, addressee): raw_mean_score}
    dialogue_counts = {}  # {(speaker, addressee): count}
    
    for pair_key, scores in pair_data.items():
        n = len(scores)
        sum_s = sum(scores)
        raw_mean = sum_s / n if n > 0 else 0.0
        
        # Bayesian Average
        s_adj = (C * global_mean + sum_s) / (C + n)
        
        adjusted_scores[pair_key] = s_adj
        raw_scores[pair_key] = raw_mean
        dialogue_counts[pair_key] = n

    # 7. Get ALL characters from the graph for consistent N x N structure
    all_characters = sorted(G.nodes())
    
    # Create matrix for heatmap (N x N where N = number of nodes)
    matrix = np.full((len(all_characters), len(all_characters)), np.nan)
    char_idx = {c: i for i, c in enumerate(all_characters)}
    
    for (speaker, addressee), score in adjusted_scores.items():
        i = char_idx[speaker]
        j = char_idx[addressee]
        matrix[i, j] = score

    plot_heatmap(matrix, all_characters, remove_empty, title=title, file_name=file_name)

    # 9. Prepare return data (as dictionaries instead of DataFrames)
    # dialogue_data: list of dicts with all dialogues and their scores
    dialogue_data = [
        {'speaker': speaker, 'addressee': addressee, 'score': score}
        for speaker, addressee, score in dialogue_scores
    ]
    
    # relationship_data: dict of dicts for consistent N x N access
    # relationship_data[speaker][addressee] = {...} or None
    relationship_data = {}
    for speaker in all_characters:
        relationship_data[speaker] = {}
        for addressee in all_characters:
            pair = (speaker, addressee)
            if pair in pair_data:
                relationship_data[speaker][addressee] = {
                    'speaker': speaker,
                    'addressee': addressee,
                    'raw_sentiment': raw_scores[pair],
                    'adjusted_sentiment': adjusted_scores[pair],
                    'dialogue_count': dialogue_counts[pair]
                }
            else:
                relationship_data[speaker][addressee] = None

    # before returning, add the raw sentiment, adjusted sentiment and dialogue count to the graph
    for u, v, data in G.edges(data=True):
        data['raw_sentiment'] = raw_scores[(u, v)]
        data['adjusted_sentiment'] = adjusted_scores[(u, v)]
        data['dialogue_count'] = dialogue_counts[(u, v)]

    return G, dialogue_data, relationship_data


def most_loved_hated(relationship_data, n=10):
    """
    Find the most loved and most hated characters based on incoming sentiment.
    Uses Bayesian average (charisma) across all incoming edges for each character.
    
    Args:
        relationship_data: dict of dicts where relationship_data[speaker][addressee] = {...} or None
        n: number of top/bottom characters to return (default 10)
    
    Returns:
        most_loved: list of top n characters with highest charisma (Bayesian average of incoming sentiments)
        most_hated: list of top n characters with lowest charisma (Bayesian average of incoming sentiments)
    """
    # 1. Extract all relationships from the dict of dicts
    all_relationships = []
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                all_relationships.append(item)
    
    # 2. Group incoming sentiments by character (addressee)
    # For each character, collect all sentiments directed towards them
    character_incoming = {}  # {character: [(sentiment, dialogue_count), ...]}
    all_sentiments = []
    total_dialogues = 0
    
    for rel in all_relationships:
        character = rel['addressee']
        sentiment = rel['adjusted_sentiment']
        dialogue_count = rel['dialogue_count']
        
        if character not in character_incoming:
            character_incoming[character] = []
        
        character_incoming[character].append((sentiment, dialogue_count))
        all_sentiments.append(sentiment)
        total_dialogues += dialogue_count
    
    # 3. Calculate global statistics for Bayesian Average
    global_mean = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
    
    # Calculate C: average number of dialogues per character
    num_characters = len(character_incoming)
    C = total_dialogues / num_characters if num_characters > 0 else 1.0
    
    # 4. Calculate charisma (Bayesian average) for each character
    # S_adj = (C * m + Σs) / (C + n)
    # where:
    #   C = confidence parameter (avg dialogues per character)
    #   m = global average sentiment
    #   Σs = weighted sum of incoming sentiments (weighted by dialogue_count)
    #   n = total number of incoming dialogues for the character
    character_charisma = {}
    
    for character, incoming_data in character_incoming.items():
        # Calculate weighted sum of sentiments
        weighted_sum = sum(sentiment * count for sentiment, count in incoming_data)
        total_count = sum(count for _, count in incoming_data)
        
        # Bayesian Average
        charisma = (C * global_mean + weighted_sum) / (C + total_count) if (C + total_count) > 0 else 0.0
        
        character_charisma[character] = {
            'character': character,
            'charisma': charisma,
            'incoming_dialogues': total_count,
            'incoming_edges': len(incoming_data)
        }
    
    # 5. Sort and return top n and bottom n
    charisma_list = list(character_charisma.values())
    most_loved = sorted(charisma_list, key=lambda x: x['charisma'], reverse=True)[:n]
    most_hated = sorted(charisma_list, key=lambda x: x['charisma'])[:n]
    
    return most_loved, most_hated


def top_relationships(relationship_data, n=5):
    """
    Analyze top relationships from the relationship_data dict of dicts.
    
    Args:
        relationship_data: dict of dicts where relationship_data[speaker][addressee] = {...} or None
        n: number of top relationships to show (default 5)
    """
    # 1. Flatten the dict and filter out None entries for 1-way analysis
    all_relationships = []
    for speaker, addressees in relationship_data.items():
        for addressee, item in addressees.items():
            if item is not None:
                all_relationships.append(item)
    
    # 2. Build lookup dict by (speaker, addressee) for reverse lookups
    pair_lookup = {}
    for rel in all_relationships:
        pair_lookup[(rel['speaker'], rel['addressee'])] = rel
    
    # === TOP N POSITIVE (1-WAY) ===
    print("\n" + "="*60)
    print(f"Top {n} Positive Relationships (1-way):")
    print("="*60)
    top_positive_1way = sorted(all_relationships, key=lambda x: x['adjusted_sentiment'], reverse=True)[:n]
    for i, rel in enumerate(top_positive_1way):
        print(f"{i+1}. {rel['speaker']} → {rel['addressee']}: {rel['adjusted_sentiment']:.4f} ({int(rel['dialogue_count'])} dialogues)")
    
    # === TOP N NEGATIVE (1-WAY) ===
    print("\n" + "="*60)
    print(f"Top {n} Negative Relationships (1-way):")
    print("="*60)
    top_negative_1way = sorted(all_relationships, key=lambda x: x['adjusted_sentiment'])[:n]
    for i, rel in enumerate(top_negative_1way):
        print(f"{i+1}. {rel['speaker']} → {rel['addressee']}: {rel['adjusted_sentiment']:.4f} ({int(rel['dialogue_count'])} dialogues)")
    
    # 3. Find bidirectional pairs (where both A→B and B→A exist)
    bidirectional_pairs = []
    seen_pairs = set()
    
    for rel in all_relationships:
        speaker, addressee = rel['speaker'], rel['addressee']
        reverse_key = (addressee, speaker)
        pair_key = tuple(sorted([speaker, addressee]))  # canonical key for the pair
        
        if reverse_key in pair_lookup and pair_key not in seen_pairs:
            reverse_rel = pair_lookup[reverse_key]
            mean_sentiment = (rel['adjusted_sentiment'] + reverse_rel['adjusted_sentiment']) / 2
            abs_diff = abs(rel['adjusted_sentiment'] - reverse_rel['adjusted_sentiment'])
            total_dialogues = rel['dialogue_count'] + reverse_rel['dialogue_count']
            
            bidirectional_pairs.append({
                'char_a': speaker,
                'char_b': addressee,
                'sentiment_a_to_b': rel['adjusted_sentiment'],
                'sentiment_b_to_a': reverse_rel['adjusted_sentiment'],
                'mean_sentiment': mean_sentiment,
                'asymmetry': abs_diff,
                'total_dialogues': total_dialogues
            })
            seen_pairs.add(pair_key)
    
    # === TOP N POSITIVE (2-WAY, MEAN) ===
    print("\n" + "="*60)
    print(f"Top {n} Positive Relationships (2-way, mean):")
    print("="*60)
    top_positive_2way = sorted(bidirectional_pairs, key=lambda x: x['mean_sentiment'], reverse=True)[:n]
    for i, pair in enumerate(top_positive_2way):
        print(f"{i+1}. {pair['char_a']} ↔ {pair['char_b']}: mean={pair['mean_sentiment']:.4f} "
              f"({pair['char_a']}→{pair['char_b']}: {pair['sentiment_a_to_b']:.4f}, "
              f"{pair['char_b']}→{pair['char_a']}: {pair['sentiment_b_to_a']:.4f}) "
              f"({int(pair['total_dialogues'])} total dialogues)")
    
    # === TOP N NEGATIVE (2-WAY, MEAN) ===
    print("\n" + "="*60)
    print(f"Top {n} Negative Relationships (2-way, mean):")
    print("="*60)
    top_negative_2way = sorted(bidirectional_pairs, key=lambda x: x['mean_sentiment'])[:n]
    for i, pair in enumerate(top_negative_2way):
        print(f"{i+1}. {pair['char_a']} ↔ {pair['char_b']}: mean={pair['mean_sentiment']:.4f} "
              f"({pair['char_a']}→{pair['char_b']}: {pair['sentiment_a_to_b']:.4f}, "
              f"{pair['char_b']}→{pair['char_a']}: {pair['sentiment_b_to_a']:.4f}) "
              f"({int(pair['total_dialogues'])} total dialogues)")
    
    # === TOP N ASYMMETRIC (2-WAY, ABS DIFFERENCE) ===
    print("\n" + "="*60)
    print(f"Top {n} Asymmetric Relationships (2-way, |A→B - B→A|):")
    print("="*60)
    top_asymmetric = sorted(bidirectional_pairs, key=lambda x: x['asymmetry'], reverse=True)[:n]
    for i, pair in enumerate(top_asymmetric):
        print(f"{i+1}. {pair['char_a']} ↔ {pair['char_b']}: asymmetry={pair['asymmetry']:.4f} "
              f"({pair['char_a']}→{pair['char_b']}: {pair['sentiment_a_to_b']:.4f}, "
              f"{pair['char_b']}→{pair['char_a']}: {pair['sentiment_b_to_a']:.4f}) "
              f"({int(pair['total_dialogues'])} total dialogues)")
    
    return {
        'top_positive_1way': top_positive_1way,
        'top_negative_1way': top_negative_1way,
        'top_positive_2way': top_positive_2way,
        'top_negative_2way': top_negative_2way,
        'top_asymmetric': top_asymmetric
    }