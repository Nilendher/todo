import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # Used for ANN, if strictly only numpy/scipy, this needs manual implementation
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Configuration ---
# IMPORTANT: Update this path to where you extracted the 'dataset' folder
DATASET_PATH = 'path/to/your/dataset' 
IMAGE_HEIGHT = 100  # Standardize image height
IMAGE_WIDTH = 100   # Standardize image width
NUM_COMPONENTS_K = 50 # Initial value of k for PCA. This will be varied for evaluation.

# --- Step 1 & 2: Generate Face Database & Mean Calculation ---
def load_images_and_create_database(dataset_path, img_height, img_width):
    """
    Loads images from the dataset, flattens them into column vectors,
    and creates the face database and calculates the mean face.
    
    Args:
        dataset_path (str): Path to the root directory of the dataset.
        img_height (int): Desired height for resizing images.
        img_width (int): Desired width for resizing images.
        
    Returns:
        tuple: (face_db, mean_face, labels_array)
            face_db (np.ndarray): The face database (mn x p).
            mean_face (np.ndarray): The mean face vector (mn x 1).
            labels_array (np.ndarray): Array of integer labels for each face.
    """
    face_images = [] # List to store flattened image vectors
    labels = []      # List to store corresponding labels (person IDs)
    person_id_map = {} # Maps person folder names to integer IDs
    current_person_id = 0

    # Iterate through each person's folder in the dataset
    for person_folder in sorted(os.listdir(dataset_path)): # Sorted for consistent ID assignment
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            person_id_map[person_folder] = current_person_id
            print(f"Loading images for {person_folder} (ID: {current_person_id})...")
            
            # Iterate through each image in the person's folder
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                # Read image in grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize image to a standard size
                    img = cv2.resize(img, (img_width, img_height))
                    # Flatten image to a column vector (mn x 1)
                    face_vector = img.flatten().reshape(-1, 1)
                    face_images.append(face_vector)
                    labels.append(current_person_id)
                else:
                    print(f"Warning: Could not load image {image_path}")
            current_person_id += 1
    
    # Concatenate all face vectors horizontally to form Face_Db (mn x p)
    face_db = np.hstack(face_images)
    print(f"Face database (Face_Db) shape: {face_db.shape} (mn x p)") # Should be (mn, p)
    
    # Calculate the mean face (M) by averaging across all images (axis=1)
    mean_face = np.mean(face_db, axis=1, keepdims=True)
    print(f"Mean face (M) shape: {mean_face.shape} (mn x 1)") # Should be (mn, 1)
    
    return face_db, mean_face, np.array(labels), person_id_map

# --- Step 3: Do Mean Zero ---
def mean_subtract_faces(face_db, mean_face):
    """
    Subtracts the mean face from each face image in the database.
    
    Args:
        face_db (np.ndarray): The face database (mn x p).
        mean_face (np.ndarray): The mean face vector (mn x 1).
        
    Returns:
        np.ndarray: Mean-subtracted face data (Delta) (mn x p).
    """
    # Subtract mean face from each column of Face_Db: Delta (mn x p)
    delta = face_db - mean_face
    print(f"Mean-subtracted faces (Delta) shape: {delta.shape}")
    return delta

# --- Step 4: Calculate Co-Variance of Mean Aligned Faces (Surrogate Covariance) ---
def calculate_surrogate_covariance(delta):
    """
    Calculates the surrogate covariance matrix (C) as Delta_T * Delta.
    
    Args:
        delta (np.ndarray): Mean-subtracted face data (mn x p).
        
    Returns:
        np.ndarray: Surrogate Covariance Matrix (C) (p x p).
    """
    # C (p x p) = Delta_T * Delta
    # Delta (mn x p) -> Delta.T (p x mn)
    # (p x mn) * (mn x p) -> (p x p)
    covariance_matrix = np.dot(delta.T, delta) 
    print(f"Surrogate Covariance Matrix (C) shape: {covariance_matrix.shape} (p x p)")
    return covariance_matrix

# --- Step 5: Do Eigenvalue and Eigenvector decomposition ---
def eigen_decomposition(covariance_matrix):
    """
    Performs eigenvalue and eigenvector decomposition on the covariance matrix.
    Sorts eigenvalues in descending order and rearranges eigenvectors accordingly.
    
    Args:
        covariance_matrix (np.ndarray): The surrogate covariance matrix (p x p).
        
    Returns:
        tuple: (sorted_eigenvalues, sorted_eigenvectors)
            sorted_eigenvalues (np.ndarray): Eigenvalues sorted in descending order.
            sorted_eigenvectors (np.ndarray): Eigenvectors corresponding to sorted eigenvalues.
    """
    # Use np.linalg.eigh for symmetric matrices for better numerical stability
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues in descending order and get the corresponding indices
    sorted_indices = np.argsort(eigenvalues)[::-1] # [::-1] for descending order
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    # Rearrange eigenvectors according to the sorted eigenvalues
    # Each column of eigenvectors is an eigenvector
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    print(f"Sorted Eigenvalues shape: {sorted_eigenvalues.shape}")
    print(f"Sorted Eigenvectors shape: {sorted_eigenvectors.shape}")
    return sorted_eigenvalues, sorted_eigenvectors

# --- Step 6: Find the best direction (Generation of feature vectors) ---
def generate_feature_vectors(sorted_eigenvectors, k):
    """
    Selects the top 'k' eigenvectors to form the feature vector (Psi).
    These are the eigenvectors of the surrogate covariance matrix (C).
    
    Args:
        sorted_eigenvectors (np.ndarray): Eigenvectors sorted by corresponding eigenvalue magnitude (p x p).
        k (int): Number of principal components (eigenvectors) to select.
        
    Returns:
        np.ndarray: Feature Vectors (Psi) (p x k).
    """
    # Select the top k eigenvectors (columns) from the sorted eigenvectors
    psi = sorted_eigenvectors[:, :k]
    print(f"Feature Vectors (Psi) shape: {psi.shape} (p x k)")
    return psi

# --- Step 7: Generating Eigenfaces ---
def generate_eigenfaces(delta, psi):
    """
    Generates the Eigenfaces (Phi) by projecting mean-aligned faces onto the feature vectors.
    
    Args:
        delta (np.ndarray): Mean-subtracted face data (mn x p).
        psi (np.ndarray): Feature Vectors (p x k).
        
    Returns:
        np.ndarray: Eigenfaces (Phi) (mn x k).
    """
    # Eigenfaces (Phi) = Delta (mn x p) * Psi (p x k) -> (mn x k)
    eigenfaces = np.dot(delta, psi)
    
    # Normalize eigenfaces to unit length for better numerical stability and visualization
    # Avoid division by zero for any zero-norm eigenfaces
    norms = np.linalg.norm(eigenfaces, axis=0)
    eigenfaces = eigenfaces / (norms + 1e-10) # Add small epsilon to prevent division by zero
    
    print(f"Eigenfaces (Phi) shape: {eigenfaces.shape} (mn x k)")
    return eigenfaces

# --- Step 8: Generate Signature of Each Face ---
def generate_face_signatures(delta, eigenfaces):
    """
    Generates the signature (omega) for each face by projecting mean-aligned faces onto the eigenfaces.
    
    Args:
        delta (np.ndarray): Mean-subtracted face data (mn x p).
        eigenfaces (np.ndarray): Eigenfaces (mn x k).
        
    Returns:
        np.ndarray: Face Signatures (Omega) (k x p).
    """
    # Omega (k x p) = Eigenfaces_T (k x mn) * Delta (mn x p)
    signatures = np.dot(eigenfaces.T, delta)
    print(f"Face Signatures (Omega) shape: {signatures.shape} (k x p)")
    return signatures

# --- Step 9: Apply ANN for training ---
def train_ann(signatures, labels):
    """
    Trains an Artificial Neural Network (ANN) using the face signatures.
    
    Args:
        signatures (np.ndarray): Face Signatures (k x p).
        labels (np.ndarray): Corresponding integer labels for each face (p,).
        
    Returns:
        tuple: (ann_model, X_test, y_test)
            ann_model: The trained MLPClassifier model.
            X_test (np.ndarray): Test set features (signatures).
            y_test (np.ndarray): Test set labels.
    """
    # Transpose signatures to get (p x k) for ANN input (samples x features)
    X = signatures.T
    y = labels
    
    # Split data into training (60%) and testing (40%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y # stratify ensures balanced classes
    )
    
    print(f"\nANN Training Data Shape: {X_train.shape} (samples x features)")
    print(f"ANN Training Labels Shape: {y_train.shape}")
    print(f"ANN Testing Data Shape: {X_test.shape} (samples x features)")
    print(f"ANN Testing Labels Shape: {y_test.shape}")

    # Initialize and train MLPClassifier (Multi-layer Perceptron)
    # hidden_layer_sizes: Tuple, i-th element represents the number of neurons in the i-th hidden layer.
    # max_iter: Maximum number of iterations for the solver to converge.
    # activation: Activation function for the hidden layer. 'relu' is common.
    # solver: The solver for weight optimization. 'adam' is a good general-purpose optimizer.
    # random_state: For reproducibility.
    ann_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, activation='relu', solver='adam', random_state=1, verbose=True)
    print("\nTraining ANN...")
    ann_model.fit(X_train, y_train)
    print("ANN Training Complete.")
    
    # Evaluate model performance
    y_train_pred = ann_model.predict(X_train)
    y_test_pred = ann_model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"ANN Training Accuracy: {train_accuracy:.4f}")
    print(f"ANN Test Accuracy: {test_accuracy:.4f}")
    
    return ann_model, X_test, y_test

# --- Steps involved in Testing ---
def recognize_face(test_image_path, mean_face, eigenfaces, ann_model, img_height, img_width, person_id_map):
    """
    Recognizes a face in a new test image using the trained PCA-ANN model.
    
    Args:
        test_image_path (str): Path to the test image.
        mean_face (np.ndarray): The mean face vector (mn x 1).
        eigenfaces (np.ndarray): Eigenfaces (mn x k).
        ann_model: The trained ANN model.
        img_height (int): Standardized image height.
        img_width (int): Standardized image width.
        person_id_map (dict): Mapping from integer IDs back to person names.
        
    Returns:
        str: Predicted person's name or "Unknown/Imposter" if not recognized.
    """
    # 1. Load and prepare test image
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return "Error: Image not found"
    
    img = cv2.resize(img, (img_width, img_height))
    test_face_vector = img.flatten().reshape(-1, 1) # (mn x 1)

    # 2. Do mean Zero for the test image
    test_face_mean_aligned = test_face_vector - mean_face # (mn x 1)

    # 3. Project test face to Eigenfaces (Omega_test)
    # Omega_test (k x 1) = Eigenfaces_T (k x mn) * test_face_mean_aligned (mn x 1)
    omega_test = np.dot(eigenfaces.T, test_face_mean_aligned) 

    # 4. Use trained ANN model to predict
    # ANN expects input in shape (1 x k) for a single sample
    predicted_label_id = ann_model.predict(omega_test.T)[0]
    
    # Get prediction probabilities/confidence
    probabilities = ann_model.predict_proba(omega_test.T)[0]
    max_prob = np.max(probabilities)
    
    # Reverse map the predicted ID to a person's name
    reverse_person_id_map = {v: k for k, v in person_id_map.items()}
    predicted_person_name = reverse_person_id_map.get(predicted_label_id, "Unknown Person ID")

    # Simple imposter detection (can be improved with thresholding)
    # If the max probability is below a certain threshold, consider it an imposter
    CONFIDENCE_THRESHOLD = 0.7 # Adjust this threshold based on your dataset and desired strictness
    if max_prob < CONFIDENCE_THRESHOLD:
        return f"Imposter (low confidence: {max_prob:.2f})"
    else:
        return f"Recognized as: {predicted_person_name} (Confidence: {max_prob:.2f})"

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Step 1, 2, 3: Load, Mean Calculate, Mean Zero ---
    print("--- Data Loading and Preprocessing ---")
    face_db, mean_face, labels, person_id_map = load_images_and_create_database(DATASET_PATH, IMAGE_HEIGHT, IMAGE_WIDTH)
    delta = mean_subtract_faces(face_db, mean_face)
    
    # --- Step 4, 5, 6, 7, 8: PCA (Eigenface Generation) ---
    print("\n--- PCA (Eigenface Generation) ---")
    covariance_matrix = calculate_surrogate_covariance(delta)
    sorted_eigenvalues, sorted_eigenvectors = eigen_decomposition(covariance_matrix)
    
    # --- Evaluation Factor a: Change k and plot accuracy ---
    print("\n--- Evaluating Accuracy vs. k ---")
    k_values_to_test = [10, 20, 30, 40, 50, 75, 100, 150, 200] # Example k values
    # Ensure k values do not exceed the number of available samples (p) - 1
    max_k = min(delta.shape[1] - 1, IMAGE_HEIGHT * IMAGE_WIDTH) # p-1 or mn
    k_values_to_test = [k for k in k_values_to_test if k <= max_k]
    
    accuracies = []
    
    for k_val in k_values_to_test:
        print(f"\n--- Processing for k = {k_val} ---")
        # Re-run PCA steps for each k_val
        psi_k = generate_feature_vectors(sorted_eigenvectors, k_val)
        eigenfaces_k = generate_eigenfaces(delta, psi_k)
        signatures_k = generate_face_signatures(delta, eigenfaces_k)
        
        # Train and evaluate ANN for this k_val
        ann_model_k, X_test_k, y_test_k = train_ann(signatures_k, labels)
        
        # Get test accuracy for this k_val
        y_pred_k = ann_model_k.predict(X_test_k)
        test_accuracy_k = accuracy_score(y_test_k, y_pred_k)
        accuracies.append(test_accuracy_k)
        print(f"Test Accuracy for k={k_val}: {test_accuracy_k:.4f}")

    # Plotting Accuracy vs. k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values_to_test, accuracies, marker='o', linestyle='-')
    plt.xlabel('Number of Principal Components (k)')
    plt.ylabel('Classification Accuracy')
    plt.title('Face Recognition Accuracy vs. Number of Principal Components (k)')
    plt.grid(True)
    plt.xticks(k_values_to_test)
    plt.show()

    # --- Final Model Training with a chosen k (e.g., the one with highest accuracy or a good balance) ---
    # For demonstration, let's use NUM_COMPONENTS_K defined at the top
    print(f"\n--- Training Final Model with k = {NUM_COMPONENTS_K} ---")
    psi_final = generate_feature_vectors(sorted_eigenvectors, NUM_COMPONENTS_K)
    eigenfaces_final = generate_eigenfaces(delta, psi_final)
    signatures_final = generate_face_signatures(delta, eigenfaces_final)
    ann_model_final, X_test_final, y_test_final = train_ann(signatures_final, labels)

    # --- Evaluation Factor b: Add imposters ---
    # This part requires actual imposter images not present in the training data.
    # For demonstration, we'll simulate by picking a random image from the test set
    # and showing how the `recognize_face` function works.
    # To truly test imposters, you'd need images of people not in your dataset.
    print("\n--- Testing Recognition with a Sample Image ---")
    
    # Example: Pick a random image from the dataset for testing
    # This assumes your DATASET_PATH has subfolders for each person
    # and each subfolder contains images.
    
    # Get a list of all image paths
    all_image_paths = []
    for person_folder in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_folder)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                all_image_paths.append(os.path.join(person_path, image_name))
    
    if all_image_paths:
        # Pick a random image to test recognition
        np.random.seed(42) # for reproducibility
        random_test_image_path = np.random.choice(all_image_paths)
        print(f"Attempting to recognize: {random_test_image_path}")
        prediction_result = recognize_face(random_test_image_path, mean_face, eigenfaces_final, ann_model_final, IMAGE_HEIGHT, IMAGE_WIDTH, person_id_map)
        print(prediction_result)
        
        # Simulate an imposter (e.g., by using an image from a different source or a very different face)
        # For a real imposter test, you would need an image of someone *not* in your dataset.
        # As a conceptual example, if you had an image 'imposter_face.jpg' outside your dataset:
        # imposter_image_path = 'path/to/an/imposter/image.jpg'
        # print(f"\nAttempting to recognize imposter: {imposter_image_path}")
        # imposter_prediction = recognize_face(imposter_image_path, mean_face, eigenfaces_final, ann_model_final, IMAGE_HEIGHT, IMAGE_WIDTH, person_id_map)
        # print(imposter_prediction)
    else:
        print("No images found in the dataset path. Please check DATASET_PATH.")

