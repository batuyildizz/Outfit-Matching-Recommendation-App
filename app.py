import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from io import BytesIO
import zipfile


with st.expander("About"):
    st.write("""
    This application helps you find the best matching outfits based on your uploaded clothing photos.
    
    - Detects colors of your top and bottom clothing.
    - Suggests matching combinations.
    - Allows you to download combined images.
    
    For more information, please contact: [batuyildiz23@outlook.com](mailto:batuyildiz23@outlook.com)

    GitHub: [github.com/batuyildizz](https://github.com/batuyildizz)
    """)
    st.image("pictures/all.png", caption="About Image")  


# Load the CSV file
df = pd.read_csv("data/colour_matrix.csv", index_col=0)

# Function to check color compatibility
def renk_uyumu(ust_renk, alt_renk):
    uyum = df.loc[ust_renk, alt_renk]
    return uyum == 1.0

# Load your color detection model
model = load_model("model.keras")

target_size = (224, 224)
color_labels = {
    'Black': 0,
    'Blue': 1,
    'Darkblue': 2,
    'Gray': 3,
    'Green': 4,
    'Red': 5,
    'White': 6,
}

def combine_images(ust_img, alt_img):
    # Resize images (same width, combined height)
    ust_img = ust_img.resize((300, 300))
    alt_img = alt_img.resize((300, 300))
    
    # Combine two images vertically
    combined_img = Image.new('RGB', (300, 600))  # Width 300, height 600
    combined_img.paste(ust_img, (0, 0))          # Top image at the top
    combined_img.paste(alt_img, (0, 300))        # Bottom image below
    
    return combined_img

# Function to predict the color from an image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    
    return list(color_labels.keys())[predicted_class]

def correct_orientation(image):
    try:
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        print(f"Orientation correction failed: {e}")
    return image

# Title
st.title("Outfit Matching Recommendation App")

# Upload top clothing photos
ust_fotolar = st.file_uploader("Upload multiple top clothing photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Upload bottom clothing photos
alt_fotolar = st.file_uploader("Upload multiple bottom clothing photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# If photos are uploaded
if ust_fotolar and alt_fotolar:
    ust_images = []
    alt_images = []
    
    with st.spinner("Processing images..."):
        for ust_foto in ust_fotolar:
            ust_image_stream = BytesIO(ust_foto.getvalue())  # Obtain byte data using `getvalue`
            ust_image = Image.open(ust_image_stream)
            ust_image = correct_orientation(ust_image)
            
            ust_renk = predict_image(ust_image_stream)  # Re-open if needed
            ust_images.append((ust_image, ust_renk))
        
        for alt_foto in alt_fotolar:
            alt_image_stream = BytesIO(alt_foto.getvalue())  # Obtain byte data using `getvalue`
            alt_image = Image.open(alt_image_stream)
            alt_image = correct_orientation(alt_image)
            
            alt_renk = predict_image(alt_image_stream)  # Re-open if needed
            alt_images.append((alt_image, alt_renk))
        
        combined_images = []

        # Create and check combinations
        if ust_images and alt_images:
            st.write("Suggested Combinations:")
            uyumlu_kombinasyonlar = []
            for ust_image, ust_renk in ust_images:
                for alt_image, alt_renk in alt_images:
                    if renk_uyumu(ust_renk, alt_renk):
                        combined_img = combine_images(ust_image, alt_image)
                        combined_images.append(combined_img)  # Save combined images
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(ust_image, caption=f"Top: {ust_renk}", width=250)
                        with col2:
                            st.image(alt_image, caption=f"Bottom: {alt_renk}", width=250)
                        with col3:
                            st.image(combined_img, caption=f"Combined: {ust_renk} and {alt_renk}", width=250)
                        st.markdown("---")

        # If there are combinations to download
        if combined_images:
            # Create ZIP file
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for idx, img in enumerate(combined_images):
                    # Save image to a byte stream
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    zip_file.writestr(f"combined_{idx+1}.png", img_byte_arr.getvalue())

            zip_buffer.seek(0)

            # Download button for the ZIP file
            st.download_button(
                label="Download Combined Images",
                data=zip_buffer,
                file_name="combined_images.zip",
                mime="application/zip"
            )
else:
    st.write("Please upload photos of tops and bottoms")
