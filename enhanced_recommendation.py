import numpy as np
import pandas as pd
import pickle
from collections import Counter

def load_models():
    """Load all trained models and data"""
    train_df = pickle.load(open('model/musicrec_train_df.pkl', 'rb'))
    tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
    tfidf_vectors = pickle.load(open('model/tfidf_vectors.pkl', 'rb'))
    tfidf_similarity = pickle.load(open('model/tfidf_similarity.pkl', 'rb'))
    return train_df, tfidf, tfidf_vectors, tfidf_similarity

def recommend_with_analysis(song_title, top_n=5):
    """
    Enhanced recommendation function that returns:
    - Similarity percentages
    - Core components that contributed to the recommendation
    - Common tags between songs
    """
    try:
        # Load models
        train_df, tfidf, tfidf_vectors, tfidf_similarity = load_models()
        
        # Find the index of the input song
        idx = train_df[train_df['title'] == song_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(tfidf_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        
        for i, (rec_idx, similarity_score) in enumerate(sim_scores):
            # Convert similarity to percentage
            similarity_percentage = round(similarity_score * 100, 2)
            
            # Get the recommended song title
            recommended_song = train_df.iloc[rec_idx].title
            
            # Get tags for both songs
            input_tags = train_df.iloc[idx]['tags'].split()
            rec_tags = train_df.iloc[rec_idx]['tags'].split()
            
            # Find common components
            common_components = set(input_tags) & set(rec_tags)
            
            # Get TF-IDF feature names
            feature_names = tfidf.get_feature_names_out()
            
            # Get TF-IDF vectors for both songs
            input_vector = tfidf_vectors[idx]
            rec_vector = tfidf_vectors[rec_idx]
            
            # Find top contributing features
            contributing_features = []
            for j, (input_val, rec_val) in enumerate(zip(input_vector, rec_vector)):
                if input_val > 0 and rec_val > 0:
                    feature_name = feature_names[j]
                    # Calculate contribution score
                    contribution = min(input_val, rec_val)
                    contributing_features.append((feature_name, contribution))
            
            # Sort by contribution and get top 5
            contributing_features.sort(key=lambda x: x[1], reverse=True)
            top_contributors = contributing_features[:5]
            
            # Format core components
            core_components = []
            for feature, contribution in top_contributors:
                core_components.append(f"{feature} ({contribution:.3f})")
            
            # Create recommendation object
            recommendation = {
                'song': recommended_song,
                'similarity_percentage': similarity_percentage,
                'core_components': core_components,
                'common_tags': list(common_components)[:5]  # Top 5 common tags
            }
            
            recommendations.append(recommendation)
        
        return recommendations
        
    except IndexError:
        return []
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

def get_song_details(song_title):
    """Get detailed information about a song"""
    try:
        train_df, _, _, _ = load_models()
        song_data = train_df[train_df['title'] == song_title]
        if not song_data.empty:
            return song_data.iloc[0]['tags']
        return "Song not found"
    except:
        return "Error retrieving song details"

# Test function
if __name__ == "__main__":
    test_song = "Humko Humise Chura Lo"
    print(f"Testing enhanced recommendations for: {test_song}")
    print("=" * 60)
    
    recommendations = recommend_with_analysis(test_song)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['song']} - {rec['similarity_percentage']}% similar")
        print(f"   Core Components: {', '.join(rec['core_components'])}")
        print(f"   Common Tags: {', '.join(rec['common_tags'])}")
        print("-" * 40) 