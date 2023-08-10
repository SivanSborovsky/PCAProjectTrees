import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Step 1: Generating the Dataset
# Read and parse the JSON data from the .txt file
def read_family_tree(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

# Vectorize the family tree data
def vectorize_family_tree(data):
    generations = get_generations(data)
    max_salary = get_max_salary(data)
    return [generations, max_salary]

# Get the number of different generations in the family tree
def get_generations(data):
    # Implement your logic to determine the number of generations
    # Example: Return the depth of the family tree
    return depth_of_family_tree(data["family_tree"])

# Get the maximum salary earned by a family member
def get_max_salary(data):
    salaries = []
    extract_salaries(data, salaries)
    return min(salaries)

# Extract salaries from the family tree data
def extract_salaries(data, salaries):
    salaries.append(data["family_tree"]["root"]["born"])
    salaries.append(data["family_tree"]["spouse"]["born"])
    if "children" in ["family_tree"]:
        children = data["family_tree"]["root"]["children"]
        for child in children:
            salaries.append(child["born"])

# Determine the depth of the family tree
def depth_of_family_tree(node):
    if "root" in node:
        if "children" in node['root']:
            max_depth = 0
            for child in node['root']["children"]:
                depth = depth_of_family_tree(child)
                max_depth = max(max_depth, depth)
            return max_depth + 1
    else:
        if "children" in node:
            max_depth = 0
            for child in node["children"]:
                depth = depth_of_family_tree(child)
                max_depth = max(max_depth, depth)
            return max_depth + 1

    return 1

# Read and vectorize all family trees in the dataset
def process_dataset(file_paths):
    dataset = []
    families = []
    for file_path in file_paths:
        data = read_family_tree(file_path)
        vector = vectorize_family_tree(data)
        dataset.append(vector)
        family_name = data["family_tree"]["root"]["name"]
        ######### append here !!
        families.append(family_name)
    return dataset, families

# Define the file paths for the family tree dataset
# file_paths = ["William_Louis01.txt","House_of_Normandy.txt", "Sborovsky.txt","David01.txt"]
file_paths = ["William_Louis01.txt","House_of_Normandy.txt", "Sborovsky.txt","David01.txt","ernest_levin.txt","Antonio_Venditto.txt"]

# Process the dataset
dataset, families = process_dataset(file_paths)

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_dataset = encoder.fit_transform(dataset)

# Step 3: Applying PCA
# Perform PCA on the dataset
pca = PCA(n_components=2)
pca.fit(encoded_dataset)
reduced_data = pca.transform(encoded_dataset)

# Step 4: Visualizing the Family Trees
# Plot the transformed data using the first two principal components
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel("Number of Generations")
plt.ylabel("Principle Component")
plt.title("PCA - Family Trees")

# Add labels to the data points
for i, family in enumerate(families):
    plt.text(reduced_data[i, 0], reduced_data[i, 1], family)
    print(f"Family Name: {family}")
    # Add additional print statements to display more information about each family
    # Example:
    print(f"Number of Generations: {dataset[i][0]}")
    print(f"earliest year: {dataset[i][1]}")
    # print("")

plt.show()

if __name__ == "__main__":
    # List of file paths containing family tree JSON data
    file_paths = ["William_Louis01.txt", "House_of_Normandy.txt", "Sborovsky.txt", "David01.txt", "ernest_levin.txt", "Antonio_Venditto.txt","Tree01.txt","Tree02.txt","Tree03.txt","Tree04.txt"]

    # Process the dataset
    dataset, families = process_dataset(file_paths)

    # Encode categorical variables using one-hot encoding
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded_dataset = encoder.fit_transform(dataset)

    # Step 3: Applying PCA
    # Perform PCA on the dataset
    pca = PCA(n_components=2)
    pca.fit(encoded_dataset)
    reduced_data = pca.transform(encoded_dataset)

    # Step 4: Visualizing the Family Trees
    # Get the number of generations for the x-axis
    num_generations = [data[0] for data in dataset]

    # Plot the transformed data using the first two principal components
    plt.scatter(num_generations, reduced_data[:, 1])  # x-axis: Number of Generations, y-axis: Principal Component 2
    plt.xlabel("Number of Generations")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Family Trees")

    # Add labels to the data points
    for i, family in enumerate(families):
        plt.text(num_generations[i], reduced_data[i, 1], family)
        print(f"Family Name: {family}")
        # Add additional print statements to display more information about each family
        # Example:
        print(f"Number of Generations: {dataset[i][0]}")
        print(f"Principal Component 1: {reduced_data[i, 0]}")
        print(f"Principal Component 2: {reduced_data[i, 1]}")
        print("")

    plt.show()