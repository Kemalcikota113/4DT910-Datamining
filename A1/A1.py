# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

# read the data
data = pd.read_csv('data/artificial_dataset.csv') # can be changed to: class_example_dataset.csv, custom_dataset.csv

# Step 1: calculate pairwise distance matrix manually

num_samples = data.shape[0]
distance_matrix = np.zeros((num_samples, num_samples))
for i in range(num_samples):
    for j in range(num_samples):
        distance_matrix[i, j] = np.linalg.norm(data.iloc[i] - data.iloc[j])

# print(distance_matrix)
# Print matrix dimensions
print("Distance matrix shape:", distance_matrix.shape)

# Step 2: Choose k and find k-distance

k = 3 # Change k value here if needed

k_distance = np.zeros(num_samples)
for i in range(num_samples):
    sorted_distances = np.sort(distance_matrix[i])
    k_distance[i] = sorted_distances[k]  # k-th nearest neighbor distance

print(k_distance)

# Step 3: Define the Average Reachability Distance (ARD) of each point

def average_reachability_distance(point_index, neighbors):
    reachability_distances = []
    for neighbor in neighbors:
        reachability_distance = max(k_distance[neighbor], distance_matrix[point_index, neighbor])
        reachability_distances.append(reachability_distance)
    return np.mean(reachability_distances)

# Do it but not in a function
average_reachability_distances = np.zeros(num_samples)
for i in range(num_samples):
    # Find k nearest neighbors
    neighbors = np.argsort(distance_matrix[i])[:k+1]  # +1 to include the point itself
    neighbors = neighbors[neighbors != i]  # Exclude the point itself
    average_reachability_distances[i] = average_reachability_distance(i, neighbors)

print(average_reachability_distances)


# step 3 (cont): Define the Local Average Reachability Distance (LARD) of each point
local_average_reachability_distances = np.zeros(num_samples)
for i in range(num_samples):
    # Find k nearest neighbors (excluding the point itself)
    neighbors = np.argsort(distance_matrix[i])[:k+1]  # +1 to include the point itself
    neighbors = neighbors[neighbors != i]  # Exclude the point itself
    local_average_reachability_distances[i] = np.mean(average_reachability_distances[neighbors])

print(local_average_reachability_distances)

# Step 4: Define Local Outlier Factor (LOF) for each point
local_outlier_factors = np.zeros(num_samples)
for i in range(num_samples):
    if local_average_reachability_distances[i] == 0:
        local_outlier_factors[i] = 0  # Avoid division by zero
    else:
        local_outlier_factors[i] = average_reachability_distances[i] / local_average_reachability_distances[i]
print("Local Outlier Factors:")
for i in range(num_samples):
    print(f"LOF[{i}] = {local_outlier_factors[i]}")

# Calculate LOF using scikit-learn for validation

lof = LocalOutlierFactor(n_neighbors=k)
y_pred = lof.fit_predict(data)
lof_scores = -lof.negative_outlier_factor_
print("LOF scores from scikit-learn:")
for i in range(num_samples):
    print(f"LOF[{i}] = {lof_scores[i]}")
print("\n")

# Compare the two LOF scores
print("Difference between manual LOF and scikit-learn LOF:")
print(local_outlier_factors - lof_scores)

