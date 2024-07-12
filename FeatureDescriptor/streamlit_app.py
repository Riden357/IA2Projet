import streamlit as st
import numpy as np
from PIL import Image
import io
from descriptor import glcm, bitdesc
from distances import manhattan, euclidean, chebyshev, canberra

try:
    signatures = np.load('signatures.npy', allow_pickle=True)
    st.write(f"Loaded {len(signatures)} signatures.")
except Exception as e:
    st.error(f"Error loading signatures: {e}")
    signatures = []

def extract_features(image, descriptor_choice):
    img = np.array(image.convert('L'))  
    
    if descriptor_choice == "GLCM":
        features = glcm(img)
    else:
        features = bitdesc(img)
    
    return features

st.title("Image Similarity Search")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

num_images = st.selectbox("Number of similar images to display", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


distance_measure = st.selectbox("Select distance measure", ("manhattan", "euclidean", "chebyshev", "canberra"))

descriptor_choice = st.selectbox("Select descriptor", ("GLCM", "BiT"))

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    features = extract_features(image, descriptor_choice)
    
    st.write("Extracted features:")
    st.write(features)

    if not features:
        st.error("No features were extracted.")
    else:
        st.write("Comparing with stored signatures...")
        distances = []
        for instance in signatures:
            try:
                signature_features, label, img_path = instance[:-2], instance[-2], instance[-1]
                if distance_measure == "manhattan":
                    dist = manhattan(features, signature_features)
                elif distance_measure == "euclidean":
                    dist = euclidean(features, signature_features)
                elif distance_measure == "chebyshev":
                    dist = chebyshev(features, signature_features)
                else:  # canberra
                    dist = canberra(features, signature_features)
                distances.append((img_path, dist, label))
            except Exception as e:
                st.error(f"Error calculating distance for {img_path}: {e}")
        
        if distances:
            distances.sort(key=lambda x: x[1])
            
            st.write(f"Displaying top {num_images} similar images:")
            num_images_to_display = min(num_images, len(distances))
            for i in range(num_images_to_display):
                img_path, dist, label = distances[i]
                st.write(f"Image: {img_path}, Distance: {dist}, Label: {label}")
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=f"Image: {img_path}, Distance: {dist}, Label: {label}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error opening image {img_path}: {e}")
        else:
            st.write("No distances were calculated.")
