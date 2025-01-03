import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
# Load the cleaned dataset
data = pd.read_csv('dataset/cleaned_dataset.csv')
# Baseline Recommendation System
def create_baseline_recommendation_system(data):
    """
    Creates a content-based recommendation system using product reviews (Text) and cosine similarity.

    Args:
    data (DataFrame): The dataset containing reviews and metadata.

    Returns:
    callable: A function to recommend products based on the baseline system.
    """
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['Text'])

    # Cosine Similarity Matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation Function
    def recommend(product_id, top_n=5):
        idx = data.index[data['ProductId'] == product_id].tolist()
        if not idx:
            return "Product ID not found."
        idx = idx[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N recommendations
        top_similar = sim_scores[1:top_n + 1]
        recommended_idx = [i[0] for i in top_similar]
        recommendations = data.iloc[recommended_idx][['ProductId', 'Summary', 'Score']]
        recommendations['SimilarityScore'] = [i[1] for i in top_similar]
        return recommendations

    return recommend
# Improving the Recommendation System
def improve_recommendation_system(data):
    """
    Improves the recommendation system by incorporating sentiment analysis and helpfulness ratio.

    Args:
    data (DataFrame): The dataset containing reviews and metadata.

    Returns:
    callable: A function to recommend products based on the improved system.
    """
    # Sentiment Analysis
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # Sentiment polarity: -1 (negative) to 1 (positive)

    data['SentimentScore'] = data['Text'].apply(analyze_sentiment)

    # Weighted Combined Score
    data['CombinedScore'] = (
        data['Score'] * data['HelpfulnessRatio'] * (data['SentimentScore'] + 1)
    )

    # TF-IDF Vectorization with Combined Score Weighting
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['Text'])

    # Adjust TF-IDF values by multiplying with CombinedScore
    weighted_matrix = tfidf_matrix.multiply(data['CombinedScore'].values[:, None])

    # Cosine Similarity Matrix
    similarity_matrix = cosine_similarity(weighted_matrix, weighted_matrix)

    # Recommendation Function
    def recommend(product_id, top_n=5):
        idx = data.index[data['ProductId'] == product_id].tolist()
        if not idx:
            return "Product ID not found."
        idx = idx[0]

        # Get similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N recommendations
        top_similar = sim_scores[1:top_n + 1]
        recommended_idx = [i[0] for i in top_similar]
        recommendations = data.iloc[recommended_idx][['ProductId', 'Summary', 'Score', 'SentimentScore']]
        recommendations['SimilarityScore'] = [i[1] for i in top_similar]
        return recommendations

    return recommend
# Create Baseline and Improved Systems
baseline_recommend = create_baseline_recommendation_system(data)
improved_recommend = improve_recommendation_system(data)
# Test Recommendations
product_id_to_test = 'B008FHUFAU'
print("Baseline Recommendations:")
print(baseline_recommend(product_id_to_test))
print("\nImproved Recommendations:")
print(improved_recommend(product_id_to_test))