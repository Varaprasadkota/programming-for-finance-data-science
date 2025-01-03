import pandas as pd
import plotly.express as px# Load the dataset
data = pd.read_csv('dataset/dataset.csv')
data.head()
# Data Cleaning
# Drop duplicate rows based on 'Id'
data = data.drop_duplicates(subset='Id')
data.shape
# Remove rows with missing values in critical columns
data = data.dropna(subset=['ProductId', 'UserId', 'ProfileName', 'Score', 'Text'])
# Convert 'Time' column to datetime format
data['Time'] = pd.to_datetime(data['Time'], unit='s')
# Remove rows where 'HelpfulnessDenominator' is zero to avoid division errors
data = data[data['HelpfulnessDenominator'] != 0]
# Create a new column for helpfulness ratio
data['HelpfulnessRatio'] = data['HelpfulnessNumerator'] / data['HelpfulnessDenominator']
# Save the cleaned dataset
data.to_csv('dataset/cleaned_dataset.csv', index=False)
print("Cleaned dataset saved as 'cleaned_dataset.csv'")
# EDA using Plotly
# Distribution of Scores
fig1 = px.histogram(data, x='Score', title='Distribution of Review Scores', labels={'Score': 'Review Score'}, nbins=5)
fig1.show()
# Helpfulness Ratio vs. Scores
fig2 = px.box(data, x='Score', y='HelpfulnessRatio', title='Helpfulness Ratio by Review Score',
              labels={'Score': 'Review Score', 'HelpfulnessRatio': 'Helpfulness Ratio'})
fig2.show()
# Number of Reviews over Time
fig3 = px.histogram(data, x='Time', title='Number of Reviews Over Time', labels={'Time': 'Review Time'}, nbins=50)
fig3.show()
# Top 10 Most Reviewed Products
top_products = data['ProductId'].value_counts().head(10).reset_index()
top_products.columns = ['ProductId', 'ReviewCount']
fig4 = px.bar(top_products, x='ProductId', y='ReviewCount', title='Top 10 Most Reviewed Products',
              labels={'ProductId': 'Product ID', 'ReviewCount': 'Number of Reviews'})
fig4.show()