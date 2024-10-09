import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your match data CSV
match_data = pd.read_csv('ProcessedData.csv')

# Load the rankings data CSV (if you're going to use it for validation later)
rankings_data = pd.read_csv('team_rankings.csv')

# Clean the data if necessary (e.g., remove undefined comments, fill missing values)
match_data = match_data.fillna(0)  # Replace undefined or missing values with 0

# Combine match data by team, aggregating necessary values if you want team-based features
# You can sum or average columns based on your model strategy
team_data = match_data.groupby('teamNumber').agg({
    'ampAuto': 'mean',
    'speakerAuto': 'mean',
    'mobility': 'mean',
    'ampTele': 'mean',
    'speakerTele': 'mean',
    'feed': 'mean',
    'trap': 'mean',
    'climb': 'mean',
    'harmony': 'mean'
}).reset_index()

# For now, ignore rankings; we'll use them later for validation
