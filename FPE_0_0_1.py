#Import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import os
from tslearn.metrics import dtw
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

pairs = ["EURUSD", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "AUDCAD", "AUDCHF",
        "AUDJPY", "AUDNZD", "AUDUSD", "CADCHF", "CADJPY", "CHFJPY", "EURAUD",
        "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "GBPAUD", "GBPCAD",
        "GBPCHF", "GBPJPY", "GBPNZD", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD"]

timeframes = [1, 5, 15, 30, 60, 240, 1440]

def read_forex_data(pairs, timeframes):
    def read_csv_file(file_path):
        dataset = pd.read_csv(file_path, delimiter='\t', 
                              names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                              parse_dates=['Date'])
        dataset = dataset[['Date', 'Close']]
        dataset.set_index('Date', inplace=True)
        return dataset

    dataframes = []
    
    for pair in pairs:
        for timeframe in timeframes:
            file_name = f"Data/{pair}{timeframe}.csv"
            file_path = os.path.join(os.getcwd(), file_name)

            try:
                df = read_csv_file(file_path)
                dataframes.append(df)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return dataframes

tforex_data = read_forex_data(pairs, timeframes)

def create_weekly_splits(dataframes):
    weekly_dataframes = []

    for dataframe in dataframes:
        if dataframe.empty:
            continue

        # Ensure the DataFrame is sorted by the Date index
        dataframe = dataframe.sort_index()

        # Group by week (assuming 'Date' is the DataFrame index)
        weekly_groups = dataframe.groupby(pd.Grouper(freq='W-SAT'))

        for _, group in weekly_groups:
            if not group.empty:
                weekly_dataframes.append(group)

    return weekly_dataframes

all_weekly_splits = create_weekly_splits(tforex_data)

def process_and_plot_groups(dataframes):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    regressor = LinearRegression()

    # List to store DataFrames with the 'Distance' column
    updated_groups = []

    for df in dataframes:
        # Check if the DataFrame is empty
        if df.empty:
            continue

        # Check if 'Date' is the index and convert it to ordinal for regression
        if pd.api.types.is_datetime64_any_dtype(df.index.dtype):
            X = np.array(df.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
        else:
            # If 'Date' is not the index, convert it to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            X = np.array(df.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)

        # Fit the regressor and predict
        regressor.fit(X, df['Close'])
        yi = regressor.predict(X)

        # Calculate the distance and scale it
        distance = df['Close'] - yi
        scaled_distance = scaler.fit_transform(distance.values.reshape(-1, 1))

        # Add the 'Distance' column to the DataFrame
        df = df.reset_index()
        df['Distance'] = scaled_distance.flatten()
        updated_groups.append(df)

    return updated_groups


processed_data = process_and_plot_groups(all_weekly_splits)

def segment_dataframes(dataframes, minArea=5):
    pattern = []

    for df in dataframes:
        checkA = 0
        checkB = 0
        halfA = []
        halfB = []

        for di in range(len(df)):
            if df['Distance'].iloc[di] < 0:  
                if checkA == 0:
                    if len(halfB) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfA.append(df.iloc[di])
                        checkA += 1
                    else:
                        if len(halfA) < minArea:
                            halfA.clear()
                            halfA.append(df.iloc[di])
                            checkA += 1
                        else:
                            full = halfA + halfB
                            pattern.append(full)
                            halfA.clear()
                            halfB.clear()
                            halfA.append(df.iloc[di])
                            checkA += 1 
                else:
                    halfA.append(df.iloc[di])
                checkB = 0
            
            elif df['Distance'].iloc[di] > 0:
                if checkB == 0:
                    if len(halfA) < minArea:
                        halfA.clear()
                        halfB.clear()
                        halfB.append(df.iloc[di])
                        checkB += 1
                    else:
                        if len(halfB) < minArea:
                            halfB.clear()
                            halfB.append(df.iloc[di])
                            checkB += 1
                        else:
                            full = halfB + halfA
                            pattern.append(full)
                            halfA.clear()
                            halfB.clear()
                            halfB.append(df.iloc[di])
                            checkB += 1
                else:
                    halfB.append(df.iloc[di])
                checkA = 0

    return pattern

segmented_patterns = segment_dataframes(processed_data)

def categorize_patterns(segmented_patterns):
    up_trends = []
    down_trends = []

    for pattern in segmented_patterns:
        if not pattern:
            continue  # Skip empty patterns

        # Check the first value of 'Distance'
        first_distance = pattern[0]['Distance'] if isinstance(pattern, list) and pattern else None

        if first_distance is not None:
            if first_distance < 0:
                up_trends.append(pattern)
            elif first_distance > 0:
                down_trends.append(pattern)

    return up_trends, down_trends

up, down = categorize_patterns(segmented_patterns)







#Pattern to find the most similar
def read_and_prepare_csv(file_path):
    dataset = pd.read_csv(file_path, delimiter='\t', 
                          names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], 
                          parse_dates=['Date'])

    # Keep only 'Date' and 'Close' columns
    dataset = dataset[['Date', 'Close']]

    # Limit to the last 120 rows
    dataset = dataset.tail(120)

    return dataset

df = read_and_prepare_csv("XAUUSD240.csv")

def process_and_calculate_distance(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    regressor = LinearRegression()

    # Check if the DataFrame is empty
    if df.empty:
        return None

    # Ensure the 'Date' column is in datetime format and set it as the index
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Convert datetime index to ordinal for regression
    X = np.array(df.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)

    # Fit the regressor and predict
    regressor.fit(X, df['Close'])
    yi = regressor.predict(X)

    # Calculate the distance and scale it
    distance = df['Close'] - yi
    scaled_distance = scaler.fit_transform(distance.values.reshape(-1, 1))

    # Reset index to bring 'Date' back as a column and add the 'Distance' column
    df.reset_index(inplace=True)
    df['Distance'] = scaled_distance.flatten()

    return df

upDistance = process_and_calculate_distance(df)

def segment_dataframe(df, minArea=5):
    checkA = 0
    checkB = 0
    halfA = []
    halfB = []
    last_pattern = None

    for di in range(len(df)):
        if df['Distance'].iloc[di] < 0:
            if checkA == 0:
                halfA.clear()
                halfB.clear()
                halfA.append(df.iloc[di])
                checkA += 1
            else:
                halfA.append(df.iloc[di])
            checkB = 0

        elif df['Distance'].iloc[di] > 0:
            if checkB == 0:
                halfB.clear()
                halfB.append(df.iloc[di])
                checkB += 1
            else:
                halfB.append(df.iloc[di])
            checkA = 0

        if len(halfA) >= minArea and len(halfB) >= minArea:
            last_pattern = halfA + halfB
            halfA = [df.iloc[di]] if df['Distance'].iloc[di] < 0 else []
            halfB = [df.iloc[di]] if df['Distance'].iloc[di] > 0 else []

    return last_pattern

segmented_pattern = segment_dataframe(upDistance)

def find_and_plot_most_similar_pattern_with_next_move(test_pattern, known_patterns):
    min_distance = float('inf')
    most_similar_pattern = None
    most_similar_index = -1

    # Convert the test pattern to a numpy array
    test_array = np.array([p['Distance'] for p in test_pattern])

    # Find the most similar pattern
    for i, pattern in enumerate(known_patterns):
        if len(pattern) == len(test_pattern):
            pattern_array = np.array([p['Distance'] for p in pattern])
            distance = np.linalg.norm(test_array - pattern_array)
            if distance < min_distance:
                min_distance = distance
                most_similar_pattern = pattern
                most_similar_index = i

    if most_similar_pattern is None:
        print("No similar pattern found.")
        return

    # Plotting the patterns
    plt.figure(figsize=(12, 6))
    plt.plot(test_array, label="Test Pattern", marker='o')
    similar_pattern_distances = [p['Distance'] for p in most_similar_pattern]
    plt.plot(similar_pattern_distances, label="Most Similar Pattern", marker='x')

    # Check if there's a next move in the most similar pattern
    if most_similar_index + 1 < len(known_patterns):
        next_move = known_patterns[most_similar_index + 1][0]['Distance']
        similar_pattern_distances.append(next_move)
        plt.plot(similar_pattern_distances, label="Most Similar Pattern with Next Move", marker='x', linestyle='--')

    plt.title("Test Pattern vs Most Similar Pattern with Next Move")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

find_and_plot_most_similar_pattern_with_next_move(segmented_pattern, up)
