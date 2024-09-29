import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load data from a directory and assign labels based on filename
def load_data_from_dir(directory, img_size):
    data = []
    labels = []
    img_ids = []
    for i, img in enumerate(os.listdir(directory)):
        try:
            img_path = os.path.join(directory, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
            resized_img = cv2.resize(img_array, (img_size, img_size))  # Resize image
            data.append(resized_img)
            img_ids.append(img)  # Store image filename (ID)

            # Assign label based on filename (assuming 'dog' or 'cat' is in the filename)
            if "dog" in img.lower():
                labels.append(1)  # Label as 1 for dogs
            elif "cat" in img.lower():
                labels.append(0)  # Label as 0 for cats
            else:
                continue

            # Show progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} images")

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels), img_ids

# Set parameters
train_dir = "D:/Tehnan/Internship/dogs-vs-cats/train/train".replace("\\", "/")  # Adjust directory path as needed
test_dir = "D:/Tehnan/Internship/dogs-vs-cats/test1/test1".replace("\\", "/")
img_size = 64  # Resize images to 64x64

# Load and preprocess the train and test data
X_train, y_train, _ = load_data_from_dir(train_dir, img_size)
X_test, y_test, img_ids = load_data_from_dir(test_dir, img_size)

# Flatten the images to 1D arrays for SVM
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Train an SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_flattened, y_train)

# Make predictions on the test set with feedback
y_pred = []
for i, test_img in enumerate(X_test_flattened):
    prediction = svm.predict([test_img])
    y_pred.append(prediction[0])

    # Show progress every 100 predictions
    if (i + 1) % 100 == 0:
        print(f"Made {i + 1} predictions")

# Evaluate the model (if test labels are available, assuming y_test is present)
if len(y_test) > 0:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Create a DataFrame with the image IDs and predicted labels
results_df = pd.DataFrame({
    'Image_ID': img_ids,
    'Predicted_Label': y_pred
})

# Save the DataFrame to an Excel file
results_df.to_excel("prediction_results.xlsx", index=False)
print("Results saved to prediction_results.xlsx")
