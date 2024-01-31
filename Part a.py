#method 1

import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Function to load data from file and convert to list
def load_data(file_path):
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            data_list.append(ast.literal_eval(line))
    return data_list

# Custom RMSE calculation
def cal_RMSE(true_prob, pred_prob): 
    K = len(true_prob)
    if len(pred_prob) != K:
        raise ValueError("Dimension mismatch")
    sum_error_sq = 0
    total_item = 0
    for a in range(K):
        total_item += len(true_prob[a])
        sum_error_sq += sum((true_prob[a] - pred_prob[a])**2)
    return np.sqrt(sum_error_sq/total_item)

# Load assortment and probability data
assortments = load_data('assortment.txt')  # specify the correct path
probabilities = load_data('probability.txt')  # specify the correct path

# Find the max product ID across all assortments
max_product_id = max([max(assortment) for assortment in assortments if assortment])

# Initialize X and y
X = np.zeros((len(assortments), max_product_id + 1), dtype=np.int)
y = np.zeros((len(assortments), max_product_id + 1))

# Set the features and labels; products not in the assortment have a target probability of 0
for i, (assortment, probs) in enumerate(zip(assortments, probabilities)):
    X[i, assortment] = 1  # feature: 1 if the product is in the assortment, 0 otherwise
    y[i, assortment] = probs  # the probabilities of the products

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary of models with default parameters
models = {
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
    'Support Vector Regression': MultiOutputRegressor(SVR()),
    'Neural Network': MultiOutputRegressor(MLPRegressor(random_state=42)),
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
}

model_performance = {}  # A dictionary to store the performance of each model

for model_name, model in models.items():
    model.fit(X_train, y_train)  # Fit the model using default parameters
    
    # Make predictions using the trained model
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, 1)  # Clip values to the [0,1] range if necessary

    # Normalize the predictions to sum to 1
    normalized_y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)

    # Prepare data for custom RMSE calculation
    y_val_list = [y_val[i, X_val[i] == 1] for i in range(len(X_val))]
    y_pred_list = [normalized_y_pred[i, X_val[i] == 1] for i in range(len(X_val))]
    
    # Calculate custom RMSE
    rmse = cal_RMSE(y_val_list, y_pred_list)
    model_performance[model_name] = rmse  # Store the RMSE
    
    print(f"{model_name}: RMSE: {rmse}")

# Select the best model based on RMSE
best_model_name = min(model_performance, key=model_performance.get)  # get the model with the lowest RMSE
best_model = models[best_model_name]

print(f'Best Model: {best_model_name}')

# Load your test assortments
test_assort = load_data('assortment_test.txt')  # Load test assortment data

# Convert test assortments to the same feature format used for training
X_test = np.zeros((len(test_assort), max_product_id + 1), dtype=np.int)
for i, assortment in enumerate(test_assort):
    X_test[i, assortment] = 1  # Set the product's corresponding column to 1 if it's in the assortment

# Predict using the best model
predictions = best_model.predict(X_test)

# Normalize the predictions to sum to 1
normalized_predictions = []
for pred in predictions:
    if pred.sum() > 0:  # avoid dividing by 0
        normalized_pred = pred / pred.sum()
    else:
        normalized_pred = pred  # if sum is 0, keep original zeros (this is an edge case)
    normalized_predictions.append(normalized_pred)

# Filter out the zero values, which represent products not in the assortment
filtered_predictions = [pred[assortment] for pred, assortment in zip(normalized_predictions, test_assort)]

# Write the filtered predictions to a file
with open('Group6.txt', 'w') as f:
    for preds in filtered_predictions:
        # Write the predicted probabilities to the file, only for the products in the test assortment
        f.write("%s\n" % list(preds))

print('Predictions for method 1 saved to Group6.txt')

#method 2

import numpy as np
from math import *
import ast
import math
from sklearn.model_selection import train_test_split



# load assortment data
assort = []
filename = open("assortment.txt","r")
for line in filename:
    assort.append(ast.literal_eval(line))



# load Choice Probability
prob = []
filename = open("probability.txt","r")
for line in filename:
    prob.append(np.array(ast.literal_eval(line)))



features1 = np.array([[1,0,0,0,0],
            [1,3000,4,3.2,95], [1,2700,4,3.2,95], [1,2400,4,3.2,95], [1,2100,4,3.2,95], [1,800,4,3.2,95],
           [1,3000,8,2.9,60], [1,2700,8,2.9,60], [1,2400,8,2.9,60], [1,2100,8,2.9,60], [1,1800,8,2.9,60],
           [1,3000,8,2.9,95], [1,2700,8,2.9,95], [1,2400,8,2.9,95], [1,2100,8,2.9,95], [1,1800,8,2.9,95],
           [1,3000,4,2.9,60], [1,2700,4,2.9,60], [1,2400,4,2.9,60], [1,2100,4,2.9,60], [1,1800,4,2.9,60],
           [1,3000,4,3.2,60], [1,2700,4,3.2,60], [1,2400,4,3.2,60], [1,2100,4,3.2,60], [1,1800,4,3.2,60],
           [1,3000,4,2.2,135], [1,2700,4,2.2,135], [1,2400,4,2.2,135], [1,2100,4,2.2,135], [1,1800,4,2.2,135]])



assortments = assort
observed_probabilities = prob



X_train, X_test, y_train, y_test = train_test_split(assortments,observed_probabilities, test_size=0.2, random_state=42)


def log_likelihood1(params):
    probabilities = []
    for assortment_indices, assortment_probs in zip(X_train, y_train):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum()) # Calculate choice probabilities for the assortment
        probabilities.append(probabilities_assortment)
    
    probabilities_flat = np.concatenate(probabilities)  # Flatten probabilities for all assortments
    log_likelihood = np.sum(np.log(probabilities_flat + 1e-10) * np.concatenate(y_train))  # Add a small constant to prevent log(0)
    return -log_likelihood  # Return negative log-likelihood for minimization



# Initial guess for coefficients
initial_params = np.zeros(features1.shape[1])


from scipy.optimize import minimize
# Perform the optimization to find the best coefficients
result = minimize(log_likelihood1, initial_params, method='L-BFGS-B')


# Extract the optimal coefficients
optimal_coefficients = result.x
optimal_coefficients



def y_test_result(params):
    probabilities = []
    for assortment_indices, assortment_probs in zip(X_test, y_test):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum())  # Calculate choice probabilities for the assortment
        probabilities.append(probabilities_assortment)
    
    return probabilities  



y_predict = y_test_result(optimal_coefficients)


# Custom RMSE calculation
def cal_RMSE(true_prob, pred_prob): 
    K = len(true_prob)
    if len(pred_prob) != K:
        raise ValueError("Dimension mismatch")
    sum_error_sq = 0
    total_item = 0
    for a in range(K):
        total_item += len(true_prob[a])
        sum_error_sq += sum((true_prob[a] - pred_prob[a])**2)
    return np.sqrt(sum_error_sq/total_item)


print('method 2 rmse')
print(cal_RMSE(y_test, y_predict))


def log_likelihood1(params):
    probabilities = []
    for assortment_indices, assortment_probs in zip(assortments, observed_probabilities):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum()) # Calculate choice probabilities for the assortment
        probabilities.append(probabilities_assortment)
    
    probabilities_flat = np.concatenate(probabilities)  # Flatten probabilities for all assortments
    log_likelihood = np.sum(np.log(probabilities_flat + 1e-10) * np.concatenate(observed_probabilities))  # Add a small constant to prevent log(0)
    return -log_likelihood  # Return negative log-likelihood for minimization



# Initial guess for coefficients
initial_params = np.zeros(features1.shape[1])


from scipy.optimize import minimize
# Perform the optimization to find the best coefficients
result = minimize(log_likelihood1, initial_params, method='L-BFGS-B')



# Extract the optimal coefficients
optimal_coefficients = result.x
optimal_coefficients


test_assort = []
filename = open("assortment_test.txt","r")
for line in filename:
    test_assort.append(ast.literal_eval(line))




def y_test_result(params):
    probabilities = []
    for assortment_indices in zip(test_assort):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum())  # Calculate choice probabilities for the assortment
        probabilities.append(probabilities_assortment)
    
    return probabilities  


y_predict = y_test_result(optimal_coefficients)
y_predict



with open('probability_test1.txt', 'w') as f:
    for item in y_predict:
        f.write("%s\n" % list(item))

print('Predictions for method 2 saved to probability_test1.txt')


