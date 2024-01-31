#method 1
import ast
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd


def parse_array_file(file_path):
    """Reads a file containing arrays on each line and returns a list of these arrays."""
    with open(file_path, 'r') as file:
        # Parse each line as a Python literal (array)
        arrays = [ast.literal_eval(line) for line in file]
    return arrays

def func(x, a, b, c):
    """Polynomial function for curve fitting."""
    return a * x**2 + b * x + c

# Parse the data from the files
assortments = parse_array_file('assortment.txt')
probabilities = parse_array_file('probability.txt')

# Define the product IDs for product b
product_b_ids = set(range(6, 11))  # Products 6 to 10 correspond to SKU b

# Filter assortments and probabilities where product b is the only product
b_only_assortments = []
b_only_probabilities = []

for assortment, probability in zip(assortments, probabilities):
    # Filter out '0' which represents no purchase, then check if there's only one product and it's from SKU 'b'
    products = [product_id for product_id in assortment if product_id != 0]
    if len(products) == 1 and products[0] in product_b_ids:
        b_only_assortments.append(assortment)
        b_only_probabilities.append(probability)

# Map product IDs to prices for product b
id_to_price = {
    6: 3000,
    7: 2700,
    8: 2400,
    9: 2100,
    10: 1800
}

# Create a dataset of price and choice probability pairs for product b
price_probability_data = []

for assortment, probability in zip(b_only_assortments, b_only_probabilities):
    product_id = assortment[1]
    price = id_to_price[product_id]
    choice_probability = probability[1]
    price_probability_data.append((price, choice_probability))

# Extract prices and probabilities from the dataset
prices = np.array([item[0] for item in price_probability_data])
probabilities = np.array([item[1] for item in price_probability_data])

# Curve fitting
popt, _ = curve_fit(func, prices, probabilities)

# Function to calculate expected revenue
def revenue(x):
    choice_probability = func(x, *popt)
    return -x * choice_probability  # negative because we want to maximize

# Find the price that maximizes expected revenue
optimal_price = None
max_rev = float('-inf')

for price in range(0, 100000):  # prices from 0 to 100000
    rev = -revenue(price)
    if rev > max_rev:
        max_rev = rev
        optimal_price = price

print('Method 1')
print("Optimal price:", optimal_price)
print("Maximum expected revenue:", max_rev)

#method 2 

import numpy as np
from math import *
import ast
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


product_b_indices = [6, 7, 8, 9, 10]  # Example list of product indices
filtered_assortments = []
filtered_probabilities = []

for assortment_indices, assortment_probs in zip(assortments, observed_probabilities):
    if len(assortment_indices) == 2 and assortment_indices[1] in product_b_indices:
        filtered_assortments.append(assortment_indices)
        filtered_probabilities.append(assortment_probs)


def log_likelihood1(params):
    probabilities = []
    for assortment_indices, assortment_probs in zip(filtered_assortments, filtered_probabilities):
        utilities = np.dot(features1[assortment_indices], params)  # Calculate utility for each alternative in the assortment
        max_utility = np.max(utilities)
        exp_utilities = np.exp(utilities - max_utility)  # Normalize to prevent overflow
        probabilities_assortment = exp_utilities / (exp_utilities.sum())  # Calculate choice probabilities for the assortment
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
optimal_coefficients = np.array(optimal_coefficients, dtype=float)

print('Method 2 is the Qnb png')
img = mpimg.imread('Qnb.png')
plt.imshow(img)
plt.axis('off')  # To turn off axis
plt.show()

#method 3
# Load and parse the data from the text files
with open("assortment.txt", 'r') as file:
    assortments = [ast.literal_eval(line) for line in file.readlines()]

with open("probability.txt", 'r') as file:
    probabilities = [ast.literal_eval(line) for line in file.readlines()]

# Filter out the assortments that contain only product b and its corresponding probabilities
product_b_assortments = []
product_b_probabilities = []

# Product b corresponds to products 6 to 10
product_b_indices = list(range(6, 11))

for assortment, probability in zip(assortments, probabilities):
    if all(product in product_b_indices or product == 0 for product in assortment):
        product_b_assortments.append(assortment)
        product_b_probabilities.append(probability)

# Now we need to analyze these assortments and probabilities to determine the optimal price for product b
# Let's extract the data for product b only
product_b_data = []
for assortment, probability in zip(product_b_assortments, product_b_probabilities):
    # Find the index of product b in the assortment
    for product in product_b_indices:
        if product in assortment:
            # The index in the probability list corresponds to the position in the assortment
            product_index = assortment.index(product)
            product_b_data.append((product, probability[product_index]))

# Let's now display the first few data points for product b
product_b_data[:10]

# Define the price levels for product b
price_levels_b = {6: 3000, 7: 2700, 8: 2400, 9: 2100, 10: 1800}

# Calculate the expected revenue for each price level (price * choice probability)
expected_revenues = []

for product, choice_prob in product_b_data:
    price = price_levels_b[product]
    expected_revenue = price * choice_prob
    expected_revenues.append((product, price, choice_prob, expected_revenue))

# Sort the expected revenues by the expected revenue value to find the optimal price
expected_revenues.sort(key=lambda x: x[3], reverse=True)

print('Method 3')
# Display the sorted expected revenues
print(expected_revenues[0])  # Show the first 10 entries for brevity


