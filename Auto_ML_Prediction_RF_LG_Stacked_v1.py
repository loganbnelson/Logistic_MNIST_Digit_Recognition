#!/usr/bin/env python
# coding: utf-8

# # Let's predict an output value! This will walk you through the process of loading training and test data and get predictions.

# ## Created by Logan Nelson
# See github for licensing. https://github.com/loganbnelson/Auto_ML_Prediction
# ### logan.b.nelson@gmail.com, nelso566@purdue.edu, 818-925-6426

# In[ ]:


# If you haven't installed the packages, uncomment the below to install

# !pip install pandas
# !pip install scikit-learn
# !pip install matplotlib


# In[ ]:


# Modules required to run everything

import csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
import re


# Let's set up a few funcitons

# In[ ]:


# Load the sets of data using the following function

def read_csv_as_dataframe(file_path):
    """
    Read a CSV file and return its data as a pandas DataFrame.

    Arguments to enter:
    file_path (str): The path to the CSV file. This will default to the current working directory.

    Returns:
    pandas.DataFrame: A DataFrame containing the CSV data.
    """
    try:
        dataframe = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    return dataframe

# Let's define a function to split our data into X and y portions. We'll split into k-folds during the actual model fit.
# At this point the x vs y can be defined by column name. If none are defined it will default to "SalePrice."
# I am temporarily removing the input side of this function for the purpose of this assignment. Recommend adding in for
# other applications.

def split_data(train, test):
    """
    Split a dataset into X_train, y_train, X_test, and y_test.

    Args:
    train (pd.DataFrame): The training dataset.
    test (pd.DataFrame): The testing dataset.

    Returns:
    pd.DataFrame: X_train (features for training).
    pd.Series: y_train (target for training).
    pd.DataFrame: X_test (features for testing).
    pd.Series or None: y_test (target for testing or None if not available).
    """
    print("\nHere are the column names to be selected from the train set.")
    print(train.columns)
    y_train_col = input("\n\nWhat column is the output or y for Training?: ") or "SalePrice"
    
    if y_train_col not in train.columns:
        raise ValueError(f"'{y_train_col}' not found in the training dataset columns.")
    
    print("\n\nHere are the column names to be selected from the test set.")
    print(test.columns)
    y_test_col = input("\n\nWhat column is the output or y for testing?: ") or y_train_col
    
    def extract_numeric(series):
        # Use regular expression to extract numeric parts of the values
        return series.apply(lambda x: float(''.join(re.findall(r'\d+\.?\d*', str(x)))))
    
    try:
        X_train = train.drop(y_train_col, axis=1)
        y_train = extract_numeric(train[y_train_col])
    except:
        raise ValueError("\n\nThe Train set was unable to be split. Did you select a valid column? The Train set must contain a valid y column.")
    
    try:
        X_test = test.drop(y_test_col, axis=1)
        y_test = extract_numeric(test[y_test_col])
    except:
        X_test = test
        print(f"\n\nThe Test does not contain numeric values in '{y_test_col}'. Did you select a valid column? If the Test set does not contain the y column, y_test will be set to None.")
        y_test = None  # Set y_test to None if it's not available

    return X_train, y_train, X_test, y_test

# Let's set up a funciton to split data if only one file is submitted. It must contain all Y output values througout.

def split_dataframe(data, test_size=0.2, random_state=None):
    """
    Split a DataFrame into train and test DataFrames.

    Args:
    data (pd.DataFrame): The input DataFrame to be split.
    test_size (float, optional): The proportion of the data to include in the test split (between 0.0 and 1.0).
    random_state (int or None, optional): Seed for random number generation (for reproducibility).

    Returns:
    pd.DataFrame: The training DataFrame.
    pd.DataFrame: The testing DataFrame.
    """
    if test_size < 0.0 or test_size > 1.0:
        raise ValueError("test_size should be a float between 0.0 and 1.0.")

    # Split the DataFrame into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data

def select_or_create_column(dataframe, prompt="Select the appropriate column as index. If no column represents the index just hit Enter:  ", default_column=None):
    """
    Prompt the user to select an existing column or create a new one with sequential values.

    Args:
    dataframe (pd.DataFrame): The DataFrame to work with.
    prompt (str, optional): The prompt message for the user. Default is "Select or create a column:".
    default_column (str, optional): The default column to use if the user doesn't specify one. Default is None.

    Returns:
    pd.Series: The selected or created column.
    """

    # Display the column names in the DataFrame
    print("Available columns:")
    print(dataframe.columns)

    # Ask the user for their choice
    user_input = input(f"{prompt} ")

    if not user_input.strip():
        # If the user didn't provide a column name, use the default_column or create a new one
        if default_column:
            selected_column = dataframe[default_column]
        else:
            # Create a new column with sequential values
            new_column_name = input("""Enter a name for the new column. 
            It can be the same column name as previous or can be a new column name: """)
            if not new_column_name:
                raise ValueError("A name is required for the new column.")
            
            # Generate sequential values for the new column
            selected_column = pd.Series(range(len(dataframe)), name=new_column_name)
    else:
        # Use the user's input as the column name
        if user_input in dataframe.columns:
            selected_column = dataframe[user_input]
        else:
            raise ValueError(f"'{user_input}' is not a valid column in the DataFrame.")

    return selected_column

# Example usage:
# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
# selected_col = select_or_create_column(df, default_column='A')
# print(selected_col)


# Let's handle categorical data by creating one-hot encoding. We'll do this for both X train and test sets.

def encode_categorical_features(X_train, X_test):
    """
    Encode categorical features using one-hot encoding with consistent columns.

    Args:
    X_train (pd.DataFrame): The feature matrix for the training data.
    X_test (pd.DataFrame): The feature matrix for the testing data.

    Returns:
    pd.DataFrame: The encoded training data with consistent columns.
    pd.DataFrame: The encoded testing data with consistent columns.
    """

    # Combine train and test sets to find common unique categories
    combined_data = pd.concat([X_train, X_test])

    # Identify categorical features based on data types
    categorical_features = combined_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Apply one-hot encoding to combined data
    combined_data_encoded = pd.get_dummies(combined_data, columns=categorical_features, drop_first=True)

    # Split the combined data back into train and test sets
    X_train_encoded = combined_data_encoded.iloc[:len(X_train)]
    X_test_encoded = combined_data_encoded.iloc[len(X_train):]

    # Ensure that both train and test sets have the same columns
    # If a column is missing in one set, add it with zeros
    for column in X_train_encoded.columns:
        if column not in X_test_encoded.columns:
            X_test_encoded[column] = 0

    for column in X_test_encoded.columns:
        if column not in X_train_encoded.columns:
            X_train_encoded[column] = 0

    return X_train_encoded, X_test_encoded

# Let's define a simple linear regression to get results of the train data set. It will be run K_folds number of times.

def train_simple_linear_regression_kfold(X_train, y_train, X_test, k_folds=5):
    """
    Train a simple linear regression model with k-fold cross-validation.

    Args:
    X_train (pd.DataFrame): The feature matrix for training data.
    y_train (pd.Series): The target vector for training data.
    X_test (pd.DataFrame): The feature matrix for testing data.
    k_folds (int, optional): The number of folds for k-fold cross-validation. Default is 5.

    Returns:
    sklearn.linear_model.LinearRegression: The trained linear regression model.
    float: The average mean squared error (MSE) from k-fold cross-validation.
    float: The average coefficient of determination (R-squared) from k-fold cross-validation.
    pd.Series: The predictions from the best-performing fold (according to validation set).
    pd.Series: The y predictors from the best-performing fold.
    np.ndarray: The feature importances from the best-performing fold.
    """

    if k_folds <= 1:
        raise ValueError("Number of folds (k_folds) must be greater than 1.")

    # Initialize variables for k-fold metrics and predictions
    mse_values = []
    r2_values = []
    best_fold_mse = float('inf')
    best_fold_r2 = -float('inf')
    best_fold_predictions = None
    best_fold_y_predictors = None
    best_fold_importances = None

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=None)

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Create a linear regression model
        model = LinearRegression()

        # Train the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation fold
        y_pred_val = model.predict(X_val_fold)

        # Calculate mean squared error (MSE) and R-squared for validation fold
        mse_fold = mean_squared_error(y_val_fold, y_pred_val)
        r2_fold = r2_score(y_val_fold, y_pred_val)

        # Update best fold metrics and predictions if necessary
        if mse_fold < best_fold_mse:
            best_fold_mse = mse_fold
            best_fold_r2 = r2_fold
            best_fold_predictions = model.predict(X_train)  # Predict on X_train
            best_fold_y_predictors = model.predict(X_test) # Predict on X_test
            best_fold_importances = np.array([0])

        # Append fold metrics to lists
        mse_values.append(mse_fold)
        r2_values.append(r2_fold)

    # Calculate average validation metrics
    avg_mse_val = (sum(mse_values) / k_folds)
    avg_r2_val = (sum(r2_values) / k_folds)

    return model, avg_mse_val, avg_r2_val, best_fold_predictions, best_fold_y_predictors, best_fold_importances

# Let's define a Random Forest model to get results of the train data set. It will be run K_folds number of times.

def train_random_forest_regression(X_train, y_train, X_test=None, n_estimators=100, max_depth=None, min_samples_split=2
                                   , min_samples_leaf=1, k_folds=5, random_state=None):
    """
    Train a regression-enhanced random forest model with k-fold cross-validation on the training data.

    Args:
    X_train (pd.DataFrame): The feature matrix for training data.
    y_train (pd.Series): The target vector for training data.
    X_test (pd.DataFrame, optional): The feature matrix for testing data. Default is None.
    n_estimators (int, optional): The number of trees in the forest. Default is 100.
    max_depth (int or None, optional): The maximum depth of the tree. None means no maximum depth. Default is None.
    min_samples_split (int, optional): The minimum number of samples required to split an internal node. Default is 2.
    min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Default is 1.
    k_folds (int, optional): The number of folds for k-fold cross-validation. Default is 5.
    random_state (int or None, optional): The random seed for reproducibility. Default is None.

    Returns:
    sklearn.ensemble.RandomForestRegressor: The trained random forest regression model.
    float: The mean squared error (MSE) from k-fold cross-validation on the training data.
    float: The coefficient of determination (R-squared) from k-fold cross-validation on the training data.
    pd.Series: The predicted values on the training data using the best fold.
    pd.Series: The predicted values on the test data (if provided).
    np.array: The feature importances of the best fold.
    """
    
    # Create a random forest regression model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    # Initialize variables for validation metrics and predictions
    mse_val = []
    r2_val = []
    y_pred_train = []
    best_mse = float('inf')  # Initialize with a high value
    best_fold_predictions = None
    best_fold_importances = None

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Fit the model on the training fold
        model.fit(X_train_fold, y_train_fold)

        # Make predictions on the validation fold
        y_pred_val = model.predict(X_val_fold)
        y_pred_train.extend(y_pred_val)  # Collect predicted values

        # Calculate mean squared error (MSE) and R-squared for validation fold
        mse_fold = mean_squared_error(y_val_fold, y_pred_val)
        r2_fold = r2_score(y_val_fold, y_pred_val)

        # Append validation metrics
        mse_val.append(mse_fold)
        r2_val.append(r2_fold)

        # Track the fold with the lowest MSE (or other metric)
        if mse_fold < best_mse:
            best_mse = mse_fold
            best_fold_predictions = model.predict(X_train)
            best_fold_importances = model.feature_importances_
            best_r2_val = r2_fold
        # If test data is provided, evaluate the model on the test data
            if X_test is not None:
                y_pred_test = model.predict(X_test)
            else:
                y_pred_test = None
    
    # Calculate average validation metrics
    avg_mse_val = (sum(mse_val) / k_folds)
    avg_r2_val = (sum(r2_val) / k_folds)

    return model, best_mse, best_r2_val, pd.Series(best_fold_predictions), pd.Series(y_pred_test), best_fold_importances

# Let's created a stacked model based on all the models created.

def stack_models(base_models, base_predictions, X_holdout, y_holdout, X_test, meta_model=LinearRegression()):
    """
    Stack (combine) multiple base models using a meta-model and calculate MSE and R2.

    Args:
    base_models (list): List of base models (already trained).
    base_predictions (list): List of predicted values from the base models.
    meta_model (sklearn model): The meta-model (e.g., Linear Regression) to combine base model predictions.
    X_holdout (pd.DataFrame): Feature matrix of the holdout set.
    y_holdout (pd.Series): Target vector of the holdout set.
    X_test (pd.DataFrame): Feature matrix of the test set.

    Returns:
    pd.Series: Predicted values on the test set using the stacked model.
    float: Mean Squared Error (MSE) of the stacked model on the holdout set.
    float: R-squared (R2) of the stacked model on the holdout set.
    """
    # Combine predicted values from base models into a new feature matrix
    X_meta = np.column_stack(base_predictions)

    # Train the meta-model on the combined predictions of the holdout set
    meta_model.fit(X_meta, y_holdout)

    # Use base models to predict on the test data
    base_predictions_test = [model.predict(X_test) for model in base_models]

    # Combine test data predictions into a new feature matrix
    X_meta_test = np.column_stack(base_predictions_test)

    # Use the meta-model to make final predictions on the test data
    final_predictions = meta_model.predict(X_meta_test)

    # Calculate MSE and R2 on the holdout set
    y_pred_holdout = meta_model.predict(X_meta)
    mse_holdout = mean_squared_error(y_holdout, y_pred_holdout)
    r2_holdout = r2_score(y_holdout, y_pred_holdout)

    return final_predictions, mse_holdout, r2_holdout

# If needed, repalce NA with Nulls in the data sets.

def replace_na_with_nulls(data):
    """
    Replace all "NA" string values in a DataFrame with nulls (NaN).

    Args:
    data (pd.DataFrame): The DataFrame in which to replace string "NA" values.

    Returns:
    pd.DataFrame: The DataFrame with "NA" values replaced by nulls.
    """
    # Replace "NA" values with NaN
    data = data.replace("NA", pd.NA)

    return data

# Imputate data as means for Null values. Changing this to median for this iteration.

def impute_missing_values(data, numeric_strategy='median', categorical_strategy='constant', constant_value=None):
    """
    Impute missing values in a DataFrame.

    Args:
    data (pd.DataFrame): The DataFrame with missing values to be imputed.
    numeric_strategy (str, optional): Imputation strategy for numerical features ('mean', 'median', or 'mode'). 
        Default is 'mean'.
    categorical_strategy (str, optional): Imputation strategy for categorical features ('constant' or 'mode'). 
        Default is 'constant'.
    constant_value (any, optional): The constant value to use for categorical imputation. Default is None.

    Returns:
    pd.DataFrame: The DataFrame with missing values imputed.
    """
    imputed_data = data.copy()

    # Impute missing values for numeric features
    numeric_columns = data.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if imputed_data[col].isna().any():
            if numeric_strategy == 'mean':
                imputed_data[col].fillna(data[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                imputed_data[col].fillna(data[col].median(), inplace=True)
            elif numeric_strategy == 'mode':
                imputed_data[col].fillna(data[col].mode()[0], inplace=True)

    # Impute missing values for categorical features
    categorical_columns = data.select_dtypes(exclude=['number']).columns
    for col in categorical_columns:
        if imputed_data[col].isna().any():
            if categorical_strategy == 'constant':
                imputed_data[col].fillna(constant_value, inplace=True)
            elif categorical_strategy == 'mode':
                imputed_data[col].fillna(data[col].mode()[0], inplace=True)

    return imputed_data

# Ensure that Train and Test (or any two dataframes) sets contain the same columns. Essentially an inner join. Un-used for now.

def align_columns(df1, df2):
    """
    Align the columns of two DataFrames based on a common set of columns.

    Args:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.

    Returns:
    pd.DataFrame: DataFrame 1 with columns aligned.
    pd.DataFrame: DataFrame 2 with columns aligned.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))

    df1_aligned = df1[common_columns]
    df2_aligned = df2[common_columns]

    return df1_aligned, df2_aligned

# Write lists or dataframes desired to csv.

def write_to_csv(data, file_name):
    """
    Write data (list or DataFrame) to a CSV file.

    Args:
    data (list or pd.DataFrame): The data to be written to the CSV file.
    file_name (str): The name of the CSV file (including the .csv extension).

    Returns:
    None, but creates CSV file.
    """
    if isinstance(data, list):
        # If data is a list, convert it to a DataFrame
        data = pd.DataFrame(data)

    # Write the data to the CSV file
    pd.DataFrame(data).to_csv(file_name, index=False)
    print(f"Data has been written to {file_name}")
    
# Get a histogram of results to visualize a set of values. Could be useful to compare predictions vs y train data.

def plot_histogram(data, column_name=None, bins=10, title="Histogram"):
    """
    Generate a histogram plot for a list, NumPy array, DataFrame with one column, or a specific column of a DataFrame.

    Args:
    data (list, np.ndarray, pd.DataFrame, pd.Series): The data to create a histogram for.
    column_name (str, optional): If data is a DataFrame, specify the column name to plot. Default is None.
    bins (int, optional): The number of bins for the histogram. Default is 10.
    title (str, optional): The title for the histogram plot. Default is "Histogram".

    Returns:
    None
    """
    # Handling for different data types (list, NumPy array, DataFrame, Series)
    if isinstance(data, (list, np.ndarray)):
        plt.hist(data, bins=bins)
    elif isinstance(data, pd.DataFrame):
        if column_name is not None:
            if column_name in data.columns:
                plt.hist(data[column_name], bins=bins)
            else:
                print(f"Column '{column_name}' not found in the DataFrame.")
                return
        else:
            print("Please specify a column name for DataFrame input.")
            return
    elif isinstance(data, pd.Series):
        plt.hist(data, bins=bins)
    else:
        print("Unsupported data type. Please provide a list, NumPy array, DataFrame, or Series.")
        return

    # Add labels and title
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)

    # Show the histogram plot
    plt.show()

# Comined two arguments that are Single-column DF, single-row DF, arrays, lists, or series into a single DF with two columns.    
    
def combine_to_dataframe(arg1, arg2, column_names=('Column 1', 'Column 2')):
    """
    Combine two arguments (single-column DataFrames, single-row DataFrames, NumPy arrays, lists, or Pandas Series) into a
    single DataFrame with two columns.

    Args:
    arg1, arg2 (pd.DataFrame, np.ndarray, list, pd.Series): The two arguments to combine.
    column_names (tuple, optional): A tuple containing the names for the two columns. Default is ('Column 1', 'Column 2').

    Returns:
    pd.DataFrame: A DataFrame with two columns.
    """
    # Convert arguments to DataFrames if they are not already
    if not isinstance(arg1, (pd.DataFrame, pd.Series)):
        arg1 = pd.DataFrame(arg1, columns=[column_names[0]])
    if not isinstance(arg2, (pd.DataFrame, pd.Series)):
        arg2 = pd.DataFrame(arg2, columns=[column_names[1]])

    # Combine the two DataFrames side by side
    combined_df = pd.concat([arg1, arg2], axis=1)

    return combined_df

# A function to evaluate all R2 values from models and return the best predicted results based on the training data.

def select_best_model_and_predictions(r2_values, predictions_list, model_names):
    """
    Select the best-performing model based on R2 values and return its predicted values.

    Args:
    r2_values (list): List of R2 values for each model.
    predictions_list (list): List of predicted values (pd.Series) from different models.
    model_names (list): List of model names or identifiers corresponding to each R2 value and prediction.

    Returns:
    pd.Series: Predicted values from the best-performing model.
    str: Name or identifier of the best-performing model.
    """

    if not (r2_values and predictions_list and model_names):
        raise ValueError("No R2 values, predicted values, or model names provided.")

    if len(r2_values) != len(predictions_list) or len(r2_values) != len(model_names):
        raise ValueError("Inconsistent number of R2 values, predicted values, or model names provided.")

    # Find the index of the model with the highest R2
    best_model_index = max(range(len(r2_values)), key=lambda i: r2_values[i])

    # Get the predicted values from the best-performing model
    best_predictions = predictions_list[best_model_index]
    best_model_name = model_names[best_model_index]
    best_r2 = r2_values[best_model_index]

    return best_predictions, best_model_name, best_r2


# The following executes functions according to user input. It should work for any tabular data set(s) in csv form.

# In[ ]:


# Constants and Configuration
INPUT_DATA_IS_SPLIT = input("Is your data already split into a Train and Test set? Y or N: ").upper() or "N"
if INPUT_DATA_IS_SPLIT == "N":
    DATA_CSV_PATH = input("""Please enter the name of the file if it is in the same folder as this program.
    If it is not in the same folder, please type the file path with the file name.   """)
else:
    TRAIN_CSV_PATH = input("""Please enter the name of the file for your TRAIN dataset. 
    If it is in the same folder as this program, just enter the name.
    If it is not in the same folder, please type the file path with the file name. """)
    TEST_CSV_PATH = input("""Please enter the name of the file for your TRAIN dataset. 
    If it is in the same folder as this program, just enter the name.
    If it is not in the same folder, please type the file path with the file name. """)

N_ESTIMATORS = int(input("How many estimators would you like to use? Suggested between 64 and 128.   ") or 64)
# MAX_DEPTH = int(input("What Max depth would you like for the forest model? E.g. None or 1+   ") or 0) # Would love to make this work, but on hold for now.
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = int(input("What mimimum samples do you want for the random forest splits?   ") or 2)
MIN_SAMPLES_LEAF = int(input("What mimimum samples do you want for each leaf for the random forest?   ") or 1)
K_FOLDS = int(input("How many folds would you like for the cross validation?   ") or 20)
RANDOM_STATE = None


# In[ ]:


# Use read_csv_as_dataframe to read in the csvs needed as dataframes
if INPUT_DATA_IS_SPLIT == "N":
    try: combined_data = read_csv_as_dataframe(DATA_CSV_PATH)
    except: print("The file cannot be found. Please check the file path and/or name and try again.")
    try: train_data, test_data = split_dataframe(combined_data, test_size=0.2, random_state=None)
    except: print("""The data file was found, but is unable to be split into train and test sets. 
    Please check the data to confirm it is structured as a delimited file.""")
else:
    train_data = read_csv_as_dataframe(TRAIN_CSV_PATH)
    test_data = read_csv_as_dataframe(TEST_CSV_PATH)


# In[ ]:


# Impute values to handle Null or NaN entries.

train_data = impute_missing_values(train_data, numeric_strategy='median', categorical_strategy='constant', constant_value=0)
test_data = impute_missing_values(test_data, numeric_strategy='median', categorical_strategy='constant', constant_value=0)


# In[ ]:


# Check the first 5 rows for the train data.

test_data.head()


# In[ ]:


# Let's split the test and train sets now. Storing the result in a tuple for future use.
# For this competition you can just hit enter without any value as it is coded for the data used. 
# For other purposes enter the column that is the output of the training and test set.

data_split = split_data(train_data, test_data)

X_test_encoded_index = select_or_create_column(data_split[2], prompt="Select or create a column:", default_column=None)


# In[ ]:


# Set the Training and Testing Data
X_train_not_encoded, y_train, X_test_not_encoded, y_test = data_split

# Encode Categorical Features
X_train_encoded, X_test_encoded = encode_categorical_features(X_train_not_encoded, X_test_not_encoded)


# In[ ]:


# Let's check the shape of our split datasets.

print(f"X_train_encoded shape: {X_train_encoded.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test_encoded shape: {X_test_encoded.shape}")

if y_test is not None:
    print(f"y_test shape: {y_test.shape}")
else:
    print("y_test is None")


# In[ ]:


# Now we get the random forest model results and store them.

train_model_results = train_random_forest_regression(
    X_train_encoded, y_train, X_test_encoded, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT
    , min_samples_leaf=MIN_SAMPLES_LEAF
    , k_folds=K_FOLDS, random_state=RANDOM_STATE)


# In[ ]:


# Print the results of our forest model and a small subset of the values.

print("The model used is " + str(train_model_results[0]) + ". Here are"
      + " the best mse test is: " + str(train_model_results[1]) + ". The best R2 of the model is " + 
     str(train_model_results[2]))


# In[ ]:


# Now we get a simple model results and store them. This will be a linear regression.

simple_model_results = train_simple_linear_regression_kfold(X_train_encoded, y_train, X_test_encoded, k_folds=K_FOLDS)


# In[ ]:


# Print the results of our model and a small subset of the values.

print("The model used is " + str(simple_model_results[0]) + ". Here are"
      + " the best mse test is: " + str(simple_model_results[1]) + ". The best R2 of the model is " +
    str(simple_model_results[2]))


# In[ ]:


# Now we call the stacked model function to get a combined value for all the moodel results we've gotten so far.

stacked_model = stack_models([train_model_results[0], simple_model_results[0]],
                             [train_model_results[3], simple_model_results[3]],
                             X_train_encoded, y_train, X_test_encoded, meta_model=LinearRegression())


# In[ ]:


# Let's compare the Linear Regression results to the Random Forest and give the best model.

best_y_predictions = select_best_model_and_predictions([train_model_results[2], simple_model_results[2], stacked_model[2]], 
                                                       [train_model_results[4], simple_model_results[4], stacked_model[0]],
                                                       [train_model_results[0], simple_model_results[0], "Stacked Model"])


# In[ ]:


# Print the Best Model Results
best_model_index = best_y_predictions[1]
best_predictions = best_y_predictions[0]
best_r2 = best_y_predictions[2]

print(f"The best model used is: {best_model_index}.")
print(f"The R-squared (R2) of the best model is: {best_r2}")


# In[ ]:


# Let's get the combination of House ID and Predicted Sale Price from the best model now and store it as a dataframe.

best_predictions_df = combine_to_dataframe(X_test_encoded_index, best_y_predictions[0], column_names=(X_test_encoded_index.name, data_split[1].name))

# Display the resulting DataFrame.
best_predictions_df


# In[ ]:


# Now we write it to a csv so we can hand off the results to any who want to use it.

write_to_csv(best_predictions_df, 'best_y_predictions.csv')


# In[ ]:


# Let's visualize the Histogram of the prices predicted. 
# It will be useful to compare to the prices of the train set to see if it makes some sense.

plot_title = str(("Histogram of Predicted " + data_split[1].name))
plot_histogram(best_y_predictions[0], bins=20, title=plot_title)
plt.show()  # Display the plot


# In[ ]:


# Let's visualize the Histogram of the prices from the train data. 

plot_title = str(("Histogram of Training " + y_train.name))
plot_histogram(y_train, bins=20, title=plot_title)
plt.show()  # Display the plot

