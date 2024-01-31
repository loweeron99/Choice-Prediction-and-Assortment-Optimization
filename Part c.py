#method 1
import pandas as pd
import ast

# Function to calculate expected revenue
def calculate_revenue(assortment, probabilities, prices):
    revenue = sum(prob * prices.get(prod, 0) for prod, prob in zip(assortment, probabilities))
    return revenue

# Load and parse the data from the text files
with open("assortment.txt", 'r') as file:
    assortments = [ast.literal_eval(line) for line in file.readlines()]

with open("probability.txt", 'r') as file:
    probabilities = [ast.literal_eval(line) for line in file.readlines()]

# Define the prices for the specific products
product_prices = {
    **dict.fromkeys(range(1, 6), 2400),    # Products 1-5
    **dict.fromkeys(range(6, 11), 2100),   # Products 6-10
    **dict.fromkeys(range(11, 16), 2700),  # Products 11-15
    **dict.fromkeys(range(16, 21), 2100),  # Products 16-20
    **dict.fromkeys(range(21, 26), 2100),  # Products 21-25
    **dict.fromkeys(range(26, 31), 2700)   # Products 26-30
}

# Initialize variables to track the top revenue and its corresponding assortment
max_revenue = 0
top_assortment = None

# Calculate revenue for each assortment and find the one with the highest revenue
for assortment, prob in zip(assortments, probabilities):
    # Filter out assortments that contain more than 3 products (excluding '0')
    if len([prod for prod in assortment if prod != 0]) <= 3:
        revenue = calculate_revenue(assortment, prob, product_prices)
        if revenue > max_revenue:
            max_revenue = revenue
            top_assortment = assortment
print('Method 1')
# Print the top revenue-generating assortment
print(f"The top revenue-generating assortment is {top_assortment} with a revenue of {max_revenue}.")

#method 2 


import numpy as np
from math import *
import ast
import math



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


product_qnc = [3, 9, 12, 19, 24, 27, 0]  # Example list of product indices

filtered_assortments = []
filtered_probabilities = []


for assortment_indices, assortment_probs in zip(assortments, observed_probabilities):
    total_in_list = 0
    for i in range(len(assortment_indices)):
        if assortment_indices[i] in product_qnc:
            total_in_list += 1
    if total_in_list == len(assortment_indices):
        filtered_assortments.append(assortment_indices)
        filtered_probabilities.append(assortment_probs)


def log_likelihood1(params):
    probabilities = []
    for assortment_indices, assortment_probs in zip(filtered_assortments, filtered_probabilities):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum()) # Calculate choice probabilities for the assortment
        probabilities.append(probabilities_assortment)
    
    probabilities_flat = np.concatenate(probabilities)  # Flatten probabilities for all assortments
    log_likelihood = np.sum(np.log(probabilities_flat + 1e-10) * np.concatenate(filtered_probabilities))  # Add a small constant to prevent log(0)
    return -log_likelihood  # Return negative log-likelihood for minimization


# Initial guess for coefficients
initial_params = np.zeros(features1.shape[1])



from scipy.optimize import minimize
# Perform the optimization to find the best coefficients
result = minimize(log_likelihood1, initial_params, method='L-BFGS-B')


# Extract the optimal coefficients
optimal_coefficients = result.x
optimal_coefficients


import itertools

# List of product indices, including the 0 option
product_indices = [0, 3, 9, 12, 19, 24, 27]

# Generate all combinations of at most 3 products including the 0 option
sku_combinations = []
for r in range(1, 5):
    product_combinations = itertools.combinations(product_indices, r)
    for combination in product_combinations:
        if 0 in combination:  # Ensure the 0 option is included in the combination
            sku_combinations.append(list(combination))
            
# Remove assortments with only the 0 option
combinations_qnc = [assortment for assortment in sku_combinations if len(assortment) > 1]


probabilities = []
revenues = []
for assortment_indices in zip(combinations_qnc):
    utilities = np.dot(features1[assortment_indices], optimal_coefficients)  # Calculate utility for each alternative in the assortment
    max_utility = np.max(utilities)
    exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
    probabilities_assortment = exp_utilities / (exp_utilities.sum())  # Calculate choice probabilities for the assortment
    probabilities.append(probabilities_assortment)
    total_revenue = 0
    
for assortment_indices, prob  in zip(combinations_qnc, probabilities):
    total_revenue = 0
    for i in range(len(assortment_indices)):
        price = features1[assortment_indices][i][1]
        choice_prob = prob[i]
        total_revenue += (price*choice_prob)
    revenues.append(total_revenue)



# Find the index of the maximum element
max_index = revenues.index(max(revenues))


print('Method 2')
print(max(revenues))
print(combinations_qnc[max_index])

#method 3 

import pandas as pd
from itertools import combinations
import ast  # To safely evaluate strings containing Python literals

# Function to process a line from the text files
def process_line(line):
    return ast.literal_eval(line.strip())

# Load and parse the data from the text files
with open("assortment.txt", 'r') as file:
    assortments = [ast.literal_eval(line) for line in file.readlines()]

with open("probability.txt", 'r') as file:
    probabilities = [ast.literal_eval(line) for line in file.readlines()]

# Creating a DataFrame with assortments and probabilities
df_combined = pd.DataFrame({
    'assortment': assortments,
    'probability': probabilities
})

# Prices for each SKU
prices_specific = {'a': 2400, 'b': 2100, 'c': 2700, 'd': 2100, 'e': 2100, 'f': 2700}

# Mapping of product numbers to SKUs
product_to_sku_mapping = {i: 'a' for i in range(1, 6)}
product_to_sku_mapping.update({i: 'b' for i in range(6, 11)})
product_to_sku_mapping.update({i: 'c' for i in range(11, 16)})
product_to_sku_mapping.update({i: 'd' for i in range(16, 21)})
product_to_sku_mapping.update({i: 'e' for i in range(21, 26)})
product_to_sku_mapping.update({i: 'f' for i in range(26, 31)})

# Calculate expected revenue function
def calculate_revenue(row, prices, product_to_sku_mapping):
    revenue = 0
    for product, probability in zip(row['assortment'], row['probability']):
        if product != 0:  # Exclude the outside option
            sku = product_to_sku_mapping.get(product, None)
            if sku:
                revenue += prices[sku] * probability
    return revenue

# Calculate expected revenue for each assortment in the dataset
df_combined['expected_revenue'] = df_combined.apply(
    lambda row: calculate_revenue(row, prices_specific, product_to_sku_mapping), 
    axis=1
)

# Generate combinations of exactly 3 SKUs
skus = ['a', 'b', 'c', 'd', 'e', 'f']
sku_combinations_3 = list(combinations(skus, 3))

# Revised function to calculate revenue for SKU combinations
def calculate_revenue_for_sku_combination(sku_combination, df, prices, product_to_sku_mapping):
    filtered_df = df[df['assortment'].apply(lambda assortment: all(product_to_sku_mapping.get(product, None) in sku_combination for product in assortment if product != 0))]
    if not filtered_df.empty:
        average_revenue = filtered_df['expected_revenue'].mean()
    else:
        average_revenue = 0
    return average_revenue

# Calculate expected revenue for each combination of 3 SKUs
sku_combination_3_revenues = {combination: calculate_revenue_for_sku_combination(combination, df_combined, prices_specific, product_to_sku_mapping) for combination in sku_combinations_3}

# Find the optimal combination of 3 SKUs
optimal_sku_combination_3 = max(sku_combination_3_revenues, key=sku_combination_3_revenues.get)
optimal_revenue_for_3_skus = sku_combination_3_revenues[optimal_sku_combination_3]

print('Method 3')
print("Optimal SKU Combination:", optimal_sku_combination_3)
print("Expected Revenue:", optimal_revenue_for_3_skus)

