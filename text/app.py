import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import streamlit as st

# Set up the data directories
image_dir = r'C:\Users\shriz\OneDrive\Desktop\skin disease\images'  # Path to image directory
label_dir = r'C:\Users\shriz\OneDrive\Desktop\skin disease\text'     # Path to label directory
img_height, img_width = 128, 128  # Resize all images to the same dimensions
batch_size = 32

# Caching image and label loading
@st.cache_data
def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    disease_to_index = {}
    index_to_disease = {}
    current_label_index = 0

    # Iterate over image files
    for image_file in os.listdir(image_dir):
        # Load the image
        img_path = os.path.join(image_dir, image_file)
        image = load_img(img_path, target_size=(img_height, img_width))
        image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)

        # Get the corresponding label from the label directory
        label_file = os.path.splitext(image_file)[0] + '.txt'  # Create label file name
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                disease_name = f.readline().strip()  # Read disease name from the label file
            # Convert disease name to a numerical label
            if disease_name not in disease_to_index:
                disease_to_index[disease_name] = current_label_index
                index_to_disease[current_label_index] = disease_name
                current_label_index += 1
            label_index = disease_to_index[disease_name]
            labels.append(label_index)
        else:
            print(f'Label file for {image_file} not found.')

    # Convert lists to numpy arrays
    images = np.array(images)
    if labels:
        labels = np.array(labels)
        labels = to_categorical(labels, num_classes=len(disease_to_index))
    return images, labels, disease_to_index, index_to_disease

# Load images and labels
images, labels, disease_to_index, index_to_disease = load_images_and_labels(image_dir, label_dir)

# Split the data
if len(labels) > 0:
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
else:
    print("No labels found. Please ensure that each image has a corresponding label file.")

# Caching the model creation to avoid recompilation
@st.cache_resource
def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if len(labels) > 0:
    model = create_model(num_classes=len(disease_to_index))
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Adjust epochs as needed
        batch_size=batch_size
    )
    # Save the model and mappings
    model.save('face_disease_detection_model.h5')
    with open('disease_mapping.txt', 'w') as f:
        for index, disease in index_to_disease.items():
            f.write(f'{index}: {disease}\n')
    print("Model training completed and saved successfully.")
else:
    print("Training skipped due to missing labels.")

# Streamlit app interface
st.title("Skin Disease Detection Model")
st.write("Model training completed and saved. Check disease_mapping.txt for disease mappings.")

# File upload for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the uploaded file
    img = load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_disease = index_to_disease[predicted_class_index]

    # Display the prediction result
    st.write(f"Predicted disease: {predicted_disease}")
