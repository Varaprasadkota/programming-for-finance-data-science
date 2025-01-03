import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
# Load the cleaned dataset
data = pd.read_csv('dataset/cleaned_dataset.csv')
# Sentiment Analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Sentiment polarity: -1 (negative) to 1 (positive)

data['SentimentScore'] = data['Text'].apply(analyze_sentiment)
# Normalize Sentiment and Helpfulness Ratios
scaler = MinMaxScaler()
data[['HelpfulnessRatio', 'SentimentScore']] = scaler.fit_transform(data[['HelpfulnessRatio', 'SentimentScore']])
# Weighted Combined Score
data['CombinedScore'] = (
    0.5 * data['Score'] +  # Weight for Score
    0.3 * data['HelpfulnessRatio'] +  # Weight for Helpfulness
    0.2 * data['SentimentScore']  # Weight for Sentiment
)
# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['Text'])
# Adjust TF-IDF values using Combined Score
weighted_matrix = tfidf_matrix.multiply(data['CombinedScore'].values[:, None])
# Cosine Similarity Matrix
similarity_matrix = cosine_similarity(weighted_matrix, weighted_matrix)
# Improved Recommendation Function
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
    recommendations = data.iloc[recommended_idx][['ProductId', 'Summary', 'Score', 'SentimentScore', 'HelpfulnessRatio']]
    recommendations['SimilarityScore'] = [i[1] for i in top_similar]
    return recommendations
# Example: Improved Recommendations
product_id_to_test = 'B008FHUFAU'
recommendations = recommend(product_id_to_test, top_n=5)
print("Improved Recommendations:")
print(recommendations)