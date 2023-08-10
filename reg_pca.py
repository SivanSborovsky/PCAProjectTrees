import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_family_tree(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

def extract_features(node):
    features = [
        node["born"],
        node["dead"],
        node["age"],
        node["Married"],
        node["Divorced"],
        node["Kids"],
        # node["gender"],
        # node["birth"],
        node["immigrated"]

    ]
    return features

def encode_birth_country(node):
    birth_country = node["birth"]
    countries = ["Sweden", "USA", "Germany", "France"]  # Replace with your list of countries
    encoded_country = [1 if country == birth_country else 0 for country in countries]
    return encoded_country

def extract_features_recursive(node, features_list):
    features_list.append(extract_features(node) + encode_birth_country(node))
    if "children" in node:
        for child in node["children"]:
            extract_features_recursive(child, features_list)
def process_dataset(file_paths):
    dataset = []
    families = []
    for file_path in file_paths:
        family_tree = read_family_tree(file_path)
        root_node = family_tree["family_tree"]["root"]
        family_features = []
        extract_features_recursive(root_node, family_features)
        dataset.extend(family_features)
        family_name = root_node["name"]
        families.extend([family_name] * len(family_features))
    return dataset, families

def plot_pca_result(X_transformed, labels, names):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(set(labels)):
        indices = [j for j, x in enumerate(labels) if x == label]
        plt.scatter(X_transformed[indices, 0], X_transformed[indices, 1], label=label)
        for j in indices:
            plt.text(X_transformed[j, 0], X_transformed[j, 1], names[j], fontsize=8)  # Add names as annotations
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.title('PCA Result')
    plt.show()
# if __name__ == "__main__":
#     # List of file paths containing family tree JSON data
#     file_paths = ["ernest_levin.txt", "Sborovsky.txt"]
#
#     # Process the dataset and extract features for PCA
#     dataset, families = process_dataset(file_paths)
#
#     # ... (previous code)
#
#     # Fit and transform the dataset using PCA
#     scaler = StandardScaler()
#     dataset_scaled = scaler.fit_transform(dataset)
#     pca = PCA(n_components=2)
#     X_transformed = pca.fit_transform(dataset_scaled)
#
#     # Extract individual names (as shown in previous code)
#
#     # Access the principal components
#     principal_components = pca.components_
#
#     # Print the principal components and their interpretations
#     num_features = len(dataset[0])
#     for i, component in enumerate(principal_components):
#         print(f"Principal Component {i+1}:")
#         for j in range(num_features):
#             print(f"Feature {j+1}: {component[j]}")
#         print()
if __name__ == "__main__":
    # List of file paths containing family tree JSON data
    file_paths = ["ernest_levin.txt","Antonio_Venditto.txt","Sborovsky.txt","William_Louis01.txt", "David01.txt","House_of_Normandy.txt","Tree01.txt","Tree02.txt","Tree03.txt","Tree04.txt"]

    # Process the dataset and extract features for PCA
    dataset, families = process_dataset(file_paths)

    # Standardize the feature matrix
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    # Perform PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(dataset_scaled)

    # Extract individual names
    names = []
    for file_path in file_paths:
        family_tree = read_family_tree(file_path)
        root_node = family_tree["family_tree"]["root"]
        family_names = [root_node["name"]]
        if "children" in root_node:
            for child in root_node["children"]:
                family_names.append(child["name"])
        names.extend(family_names * len(family_names))
        # Access the principal components
    principal_components = pca.components_

    # Print the principal components and their interpretations
    # num_features = len(dataset[0])
    # for i, component in enumerate(principal_components):
    #     print(f"Principal Component {i + 1}:")
    #     for j in range(num_features):
    #         print(f"Feature {j + 1}: {component[j]:.4f}")
    #     print()

    # Visualize the PCA results with points and labels
    plot_pca_result(X_transformed, families, names)
    for idx, pc in enumerate(pca.components_):
        print(f"Principal Component {idx + 1}:")
        for feature_idx, feature_weight in enumerate(pc):
            print(f"Feature {feature_idx + 1}: {feature_weight:.4f}")
        print()

