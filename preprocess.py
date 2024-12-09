import pandas as pd
import numpy as np


def prepare_data(df):
    """
    Prepare dataset by handling missing values and cleaning data.
    
    Args:
        df: pandas DataFrame with the raw data
    
    Returns:
        Cleaned pandas DataFrame ready for modeling
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # First identify columns with too many missing values (>70%)
    missing_percentages = data.isnull().mean()
    columns_to_drop = missing_percentages[missing_percentages > 0.7].index.tolist()
    data = data.drop(columns=columns_to_drop)
    
    # Handle remaining numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        # Calculate median for the column (excluding NaN)
        median_value = data[col].median()
        
        # Fill NaN with median
        if pd.isna(data[col]).any():
            data[col] = data[col].fillna(median_value)
    
    # Handle categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        # Fill missing values with mode (most frequent value)
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value)
    
    # Special handling for Season columns (convert to numeric)
    season_columns = [col for col in data.columns if col.endswith('-Season')]
    season_mapping = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    
    for col in season_columns:
        data[col] = data[col].map(season_mapping)
        # Fill any remaining NaN with mode
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Handle Sex column
    if 'Basic_Demos-Sex' in data.columns:
        data['Basic_Demos-Sex'] = data['Basic_Demos-Sex'].astype(int)
    
    # Special handling for PCIAT columns
    pciat_columns = [col for col in data.columns if col.startswith('PCIAT-')]
    for col in pciat_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            # Fill NaN with median
            data[col] = data[col].fillna(data[col].median())
    
    # Verify no missing values remain
    assert data.isnull().sum().sum() == 0, "There are still missing values in the dataset"
    
    return data



if __name__=="__main__":
    prepare_data(pd.read_csv("data/train.csv")).to_csv("data/cleaned.csv")