import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

def load_and_preprocess_adult(filepath='adult.csv', save_scaler=True):
    """
    Load Adult dataset, preprocess it, and extract feature values
    Returns the preprocessed features in the same format as your example
    """
    # Load dataset
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, na_values=' ?', skipinitialspace=True)
    else:
        # Download if needed
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, header=None, na_values=' ?', skipinitialspace=True)
        df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    # Drop missing values
    df = df.dropna()
    
    # Encode target
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
    
    # Categorical columns to encode
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
    
    # Label encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Separate features and target
    X = df.drop('income', axis=1).values.astype(np.float32)
    y = df['income'].values
    feature_names = df.drop('income', axis=1).columns.tolist()
    
    # Standardize features (subtract mean, divide by std)
    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    X_scaled = X
    
    # Save preprocessing info if requested
    # if save_scaler:
    #     preprocessing_info = {
    #         'feature_names': feature_names,
    #         'scaler_mean': scaler.mean_.tolist(),
    #         'scaler_std': scaler.scale_.tolist(),
    #         'label_encoders': {col: list(le.classes_) for col, le in le_dict.items()}
    #     }
    #     with open('adult_preprocessing_info.json', 'w') as f:
    #         json.dump(preprocessing_info, f, indent=2)
    #     print("Preprocessing info saved to adult_preprocessing_info.json")
    
    return X_scaled, y, feature_names, scaler, le_dict

def extract_sample_features(X, y, index=0, format='c_array'):
    """
    Extract a single sample's features and format them
    
    Args:
        X: Feature matrix
        y: Labels
        index: Which sample to extract
        format: 'c_array', 'python_list', or 'csv'
    """
    features = X[index]
    label = y[index]
    
    print(f"\nSample {index}:")
    print(f"Label: {label}")
    print(f"Number of features: {len(features)}")
    
    if format == 'c_array':
        # Format as C/C++ array
        print("\nFeatures (C array format):")
        feature_str = ",\n        ".join([f"{f}" for f in features])
        print(f"float features[] = {{\n        {feature_str}\n}};")
    
    elif format == 'python_list':
        # Format as Python list
        print("\nFeatures (Python list):")
        print(features.tolist())
    
    elif format == 'csv':
        # Format as CSV
        print("\nFeatures (CSV format):")
        print(",".join([str(f) for f in features]))
    
    return features, label

def save_multiple_samples(X, y, output_file='adult_samples.txt', num_samples=10):
    """
    Save multiple samples to a file
    """
    with open(output_file, 'w') as f:
        for i in range(min(num_samples, len(X))):
            # f.write(f"Sample {i}, Label: {y[i]}\n")
            feature_str = str(y[i]) + " " + " ".join([f"{feat}" for feat in X[i]])# + ", " + str(y[i]) 
            f.write(f"{feature_str}\n")
            # f.write("\n" + "="*80 + "\n\n")
    
    print(f"\nSaved {num_samples} samples to {output_file}")

def convert_raw_to_scaled(raw_features, scaler_mean, scaler_std):
    """
    Convert raw features to scaled features using mean and std
    
    Args:
        raw_features: List or array of raw feature values
        scaler_mean: Mean values from training
        scaler_std: Std values from training
    """
    raw_features = np.array(raw_features)
    scaled = (raw_features - scaler_mean) / scaler_std
    return scaled

def main():
    print("="*80)
    print("ADULT DATASET FEATURE EXTRACTION")
    print("="*80)
    
    # Load and preprocess
    X, y, feature_names, scaler, le_dict = load_and_preprocess_adult()
    
    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"\nFeature names: {feature_names}")
    
    # Extract first sample in different formats
    print("\n" + "="*80)
    print("FIRST SAMPLE - C ARRAY FORMAT")
    print("="*80)
    extract_sample_features(X, y, index=0, format='c_array')
    
    print("\n" + "="*80)
    print("FIRST SAMPLE - PYTHON LIST FORMAT")
    print("="*80)
    extract_sample_features(X, y, index=0, format='python_list')
    
    print("\n" + "="*80)
    print("FIRST SAMPLE - CSV FORMAT")
    print("="*80)
    extract_sample_features(X, y, index=0, format='csv')
    
    # Save multiple samples
    save_multiple_samples(X, y, output_file='test/ai/data/inputs/adult_test_1.txt', num_samples=1e9)
    
    # Example: Extract specific samples
    print("\n" + "="*80)
    print("SAMPLES WITH DIFFERENT LABELS")
    print("="*80)
    
    # Find one sample for each class
    class_0_idx = np.where(y == 0)[0][0]
    class_1_idx = np.where(y == 1)[0][0]
    
    print("\nClass 0 sample (income <=50K):")
    extract_sample_features(X, y, index=class_0_idx, format='c_array')
    
    print("\nClass 1 sample (income >50K):")
    extract_sample_features(X, y, index=class_1_idx, format='c_array')
    
    # Show raw vs scaled example
    print("\n" + "="*80)
    print("RAW VS SCALED FEATURES (First sample)")
    print("="*80)
    
    # Get raw features (before scaling)
    df = pd.read_csv('adult.csv', na_values=' ?', skipinitialspace=True) if pd.io.common.file_exists('adult.csv') else None
    if df is not None:
        df = df.dropna()
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
        
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        raw_features = df.drop('income', axis=1).iloc[0].values
        scaled_features = X[0]
        
        print("\nFeature comparison:")
        print(f"{'Feature':<20} {'Raw Value':<15} {'Scaled Value':<15}")
        print("-" * 50)
        for i, name in enumerate(feature_names):
            print(f"{name:<20} {raw_features[i]:<15.2f} {scaled_features[i]:<15.6f}")

if __name__ == "__main__":
    main()