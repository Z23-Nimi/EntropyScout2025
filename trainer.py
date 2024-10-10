import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Step 1: Load and process the CSV data ---
def process_csv(file_path):
    """
    This function processes the CSV file containing match data.
    It aggregates the data per team by calculating the mean of numeric columns.
    """
    df = pd.read_csv(file_path)

    # Ensure that teamNumber is treated as a column
    if 'teamNumber' in df.columns:
        df['teamNumber'] = df['teamNumber'].astype(str)  # Ensure teamNumber is string type if needed

    # Aggregate data by teamNumber, calculating the mean for numeric columns
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    # Check if 'teamNumber' is in numeric_columns before removing
    if 'teamNumber' in numeric_columns:
        numeric_columns.remove('teamNumber')

    team_data = df.groupby('teamNumber', as_index=False)[numeric_columns].mean()

    return team_data

# --- Step 2: Load the ranking data ---
def load_rankings(file_path):
    """
    This function loads the CSV file containing overall rankings.
    It assumes the CSV has a 'teamNumber' and 'rank' column.
    """
    rankings = pd.read_csv(file_path)
    return rankings[['teamNumber', 'rank']]

# --- Step 3: Merge match data with rankings ---
def merge_data(team_data, rankings):
    """
    This function merges the team data with the overall rankings.
    """
    merged_data = pd.merge(team_data, rankings, on='teamNumber', how='inner')
    return merged_data

# --- Step 4: Train a machine learning model ---
def train_model(X_train, y_train):
    """
    This function trains a Random Forest model on the training data.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# --- Step 5: Main workflow ---
def main(team_data_file, ranking_data_file):
    """
    Main function to process team data and ranking data.
    """
    # Process team data
    team_data = process_csv(team_data_file)
    
    # Process ranking data
    rankings = pd.read_csv(ranking_data_file)

    # Ensure teamNumber is the same type in both dataframes
    team_data['teamNumber'] = team_data['teamNumber'].astype(int)  # Convert to int if rankings are int
    rankings['teamNumber'] = rankings['teamNumber'].astype(int)  # Convert to int if rankings are int

    # Merge the processed data with rankings
    data = merge_data(team_data, rankings)

    # Prepare features (X) and target (y)
    X = data.drop(columns=['teamNumber', 'rank'])
    y = data['rank']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Cross-Validation for better evaluation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Mean Cross-Validated MSE: {-scores.mean()}")
    
    # Plot true values vs predictions
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Rankings')
    plt.show()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.show()

# --- Step 6: Execute main function ---
if __name__ == "__main__":
    team_data_file = 'ProcessedData.csv'  # Replace with the path to your team data CSV
    ranking_data_file = 'team_rankings.csv'  # Replace with the path to your rankings CSV
    main(team_data_file, ranking_data_file)

