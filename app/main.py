import os
import json
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import gdown

# Define the model file ID from Google Drive
google_drive_file_id = "1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"

# Define a local path for the model
local_model_path = "model.h5"

# Check if the model is already downloaded
if not os.path.exists(local_model_path):
    gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", local_model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(local_model_path)
class_indices = json.load(open(class_indices_path))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to Save Image to BytesIO (to handle in-memory image objects)
def save_image_to_buffer(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)  # Reset the pointer to the beginning
    return img_buffer

# Function to Convert the Image to a Temporary File Path
def save_image_to_tempfile(image):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(temp_file, format='JPEG')
    temp_file.close()  # Close the file to ensure it's saved properly
    return temp_file.name  # Return the file path

# Function to Generate a PDF Report with Image Embedded
def generate_pdf_report(prediction, image):
    from reportlab.lib.utils import ImageReader

    # Create a BytesIO object to hold the PDF data
    buffer = BytesIO()

    # Create a canvas for the PDF
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add title and prediction information to the PDF
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Plant Disease Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 130, f"Disease/Pest Identified: {prediction}")
    c.drawString(100, height - 160, "Recommendation:")
    c.drawString(100, height - 190, "- Please follow local agricultural guidelines for treatment.")
    c.drawString(100, height - 220, "- Consult with agricultural experts if necessary.")
    c.drawString(100, height - 250, "The AI model provides this prediction based on the leaf image analysis.")
    
    # Resize and position the image below the text without overlap
    img_width = 250  # Adjust width as needed
    img_height = 250  # Adjust height as needed
    x_position = (width - img_width) / 2  # Center horizontally
    y_position = height - 500  # Adjust vertical position to avoid overlapping text

    # Convert the PIL image to an ImageReader object
    image_reader = ImageReader(image)
    
    # Draw the image on the PDF
    c.drawImage(image_reader, x_position, y_position, width=img_width, height=img_height, preserveAspectRatio=True)

    # Save the PDF to the buffer
    c.save()

    # Get the PDF data from the buffer
    pdf_data = buffer.getvalue()
    buffer.close()

    st.write("""
    'Pepper__bell___Bacterial_spot': "Water-soaked spots on leaves and fruits. Treat with copper-based fungicides. Practice crop rotation and avoid overhead irrigation.",
    'Pepper__bell___healthy': "No issues detected. Continue regular care, ensure optimal watering and nutrient supply.",
    'Potato___Early_blight': "Dark, concentric spots on older leaves. Treat with fungicides containing chlorothalonil or mancozeb. Remove infected leaves and practice crop rotation.",
    'Potato___Late_blight': "Irregular water-soaked spots on leaves. Treat with fungicides containing metalaxyl or mefenoxam. Remove infected plants and avoid wet foliage.",
    'Potato___healthy': "No issues detected. Continue regular care with balanced fertilization and proper irrigation.",
    'Tomato_Bacterial_spot': "Small, dark spots with yellow halos on leaves. Use copper-based bactericides. Avoid overhead watering.",
    'Tomato_Early_blight': "Concentric dark spots on lower leaves. Use fungicides containing chlorothalonil. Remove plant debris.",
    'Tomato_Late_blight': "Water-soaked, irregular lesions on leaves. Use fungicides containing metalaxyl. Remove affected plants.",
    'Tomato_Leaf_Mold': "Yellow spots on upper leaf surfaces. Treat with copper or sulfur-based fungicides. Ensure good ventilation.",
    'Tomato_Septoria_leaf_spot': "Small, gray spots with dark borders. Use fungicides with chlorothalonil. Clear plant debris.",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Fine webbing on leaves. Use insecticidal soaps or miticides. Introduce ladybugs.",
    'Tomato__Target_Spot': "Circular, dark brown spots with concentric rings. Use fungicides with azoxystrobin. Remove infected debris.",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Curling, yellowing leaves. Control whiteflies with insecticidal soaps. Remove infected plants.",
    'Tomato__Tomato_mosaic_virus': "Mottled leaves with mosaic appearance. Remove infected plants. Practice proper sanitation.",
    'Tomato_healthy': "No issues detected. Maintain consistent care with proper watering and fertilization."       
    """)

    return pdf_data


# Streamlit Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Home", "Prediction", "About"])

# Home Page
if page == "Home":
    st.title('Welcome to the Plant Disease Classifier')
    st.write(""" 
    This is an advanced AI-powered system designed to assist farmers and agricultural professionals in the early detection and classification of plant diseases. By simply uploading an image of a plant leaf, our system leverages deep learning models to quickly analyze the image and provide accurate predictions about potential diseases or pest infestations. The aim is to empower farmers with timely insights, enabling them to take appropriate measures before the issue spreads and causes significant crop damage.
    
    Our model is trained on a vast dataset of leaf images, covering various plant species and common plant diseases, ensuring that the predictions are not only accurate but also applicable to a wide range of plants. With a user-friendly interface, the system is accessible even to those with minimal technical expertise, making it an invaluable tool for farmers looking to improve crop health and productivity. Upload an image today and get instant disease classification, along with actionable recommendations for treatment and prevention.
    """)
    st.image("https://via.placeholder.com/800x400.png?text=Plant+Disease+Classifier", use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.title('Plant Disease Prediction')
    
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((150, 150))
            st.image(resized_img)

        with col2:
            if st.button('Classify'):
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')

                # Generate PDF report with image included
                pdf_report = generate_pdf_report(prediction, image)

                # Allow user to download the report as a PDF
                download_button = st.download_button(
                    label="Download Report",
                    data=pdf_report,
                    file_name="plant_disease_report.pdf",
                    mime="application/pdf"
                )

# About Page
elif page == "About":
    st.title("About the Project")
    st.write(""" 
    This project is aimed at providing an AI-powered solution for early detection and classification of plant diseases and pests. Agriculture is one of the most important industries globally, and crop health is a critical aspect of ensuring food security. However, farmers often struggle to identify plant diseases or pests in the early stages, which can lead to substantial crop losses. By utilizing deep learning and computer vision techniques, this system addresses this challenge by offering a fast, accurate, and easy-to-use platform for diagnosing plant health issues based on images of leaves.
    
    The system is powered by a deep learning model that has been trained on a large, diverse dataset of leaf images. This dataset includes a variety of plant species, diseases, and pests, ensuring that the model can make accurate predictions across different plants. The model uses Convolutional Neural Networks (CNNs), which are well-suited for image classification tasks, to extract meaningful patterns from leaf images and classify them into disease or pest categories. As a result, farmers and agricultural professionals can rely on this tool to help them take timely and appropriate action before the issue affects the entire crop.
    
    In addition to disease detection, the system also provides actionable recommendations based on the predictions, such as treatment options and preventive measures. This empowers farmers with the necessary knowledge to address the problem effectively. Furthermore, the system's interface is designed to be user-friendly, making it accessible even to those without technical expertise. Farmers can easily upload images of their plant leaves and get instant results. The user-friendly web interface, built with Streamlit, ensures seamless interaction, allowing for a smooth experience from start to finish.
    
    The following technologies were used to develop this system:
    
    TensorFlow and Keras: These powerful libraries were used for building and training the deep learning model that powers the disease classification. TensorFlow and Keras are highly efficient and widely used in the machine learning community, making them ideal for building a scalable and accurate model.
    
    Streamlit: The web interface of the system is built with Streamlit, allowing for rapid development of interactive web applications. Streamlit’s simplicity and flexibility make it an excellent choice for building applications that are both powerful and easy to use.
    
    Pillow: This library is used for image preprocessing tasks, including resizing and normalizing the images before feeding them into the deep learning model. Pillow ensures that the uploaded images are prepared properly for accurate prediction.
    
    This project serves as a valuable tool for farmers, agricultural scientists, and anyone interested in plant health. With the increasing threat of pests and diseases to crops globally, having such technology can greatly contribute to early detection and effective intervention. By bringing AI to the forefront of agriculture, this system has the potential to reduce crop loss and increase food security, benefiting both farmers and consumers alike.
    """)
