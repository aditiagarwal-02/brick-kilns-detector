import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import tempfile
import os
import shutil
import zipfile
from PIL import Image
import numpy as np
import io
import gdown
import pandas as pd

import tensorflow as tf

url = 'https://drive.google.com/uc?id=1DBl_LcIC3-a09bgGqRPAsQsLCbl9ZPJX'
output = 'model_resnet_fine_ind.h5'
gdown.download(url, output, quiet=True)

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('model_resnet_fine_ind.h5')
    return model

# model = tf.keras.models.load_model('model_resnet_fine_ind.h5')
mapbox_token = 'pk.eyJ1IjoiYWRpdGktMTgiLCJhIjoiY2xsZ2dlcm9zMHRiMzNkcWF2MmFjZTc3biJ9.axO4l5PRwHHn2H3wEE-cEg'

def get_static_map_image(latitude, longitude, api):
 # Replace with your Google Maps API Key
    base_url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
        'center': f'{latitude},{longitude}',
        'zoom': 17,  # You can adjust the zoom level as per your requirement
        'size': '256x276',  # You can adjust the size of the image as per your requirement
        'maptype': 'satellite',
        'key': api,
    }
    response = requests.get(base_url, params=params)
    return response.content

def create_map():
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        control_scale=True
    )

    # Add Mapbox tiles with 'Mapbox Satellite' style
    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={mapbox_token}",
        attr="Mapbox Satellite",
        name="Mapbox Satellite"
    ).add_to(india_map)

    return india_map

def imgs_input_fn(images):
    img_size = (224, 224)
    img_size_tensor = tf.constant(img_size, dtype=tf.int32)
    images = tf.convert_to_tensor(value = images)
    images = tf.image.resize(images, size=img_size_tensor)
    return images

def main():

    hide_st_style = """
            <style>
            body {
            background-color: black;
            color: white;
        }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    model = load_model()

    st.title("Brick Kiln Detector")
    st.write("This app uses a deep learning model to detect brick kilns in satellite images. The model was trained on satellite images of India. The app allows you to select a location on the map and download the images of brick kilns and non-brick kilns in that region.")

    st.sidebar.title("Search Location")
    lat = st.sidebar.number_input("Latitude:", value=20.5937, step=0.000001)
    lon = st.sidebar.number_input("Longitude:", value=78.9629, step=0.000001)

    india_map = create_map()
    india_map.location = [lat, lon]

    # Add marker for selected latitude and longitude
    folium.Marker(
        location=[lat, lon],
        popup=f"Latitude: {lat}, Longitude: {lon}",
        icon=folium.Icon(color='blue')
    ).add_to(india_map)

    # Initialize variables to store user-drawn polygons
    drawn_polygons = []

    # Specify the latitude and longitude for the rectangular bounding box
    st.sidebar.title("Bounding Box")
    box_lat1 = st.sidebar.number_input("Latitude 1:", value=28.74, step=0.000001)
    box_lon1 = st.sidebar.number_input("Longitude 1:", value=77.60, step=0.000001)
    box_lat2 = st.sidebar.number_input("Latitude 2:", value=28.90, step=0.000001)
    box_lon2 = st.sidebar.number_input("Longitude 2:", value=77.90, step=0.000001)

    # Add the rectangular bounding box to the map
    bounding_box_polygon = folium.Rectangle(
        bounds=[[box_lat1, box_lon1], [box_lat2, box_lon2]],
        color='red',
        fill=True,
        fill_opacity=0.2,
    )
    bounding_box_polygon.add_to(india_map)
    drawn_polygons.append(bounding_box_polygon.get_bounds())

    df = pd.DataFrame(columns = ['Latitude', 'Longitude'])

    
    # Display the map as an image using st.image()
    folium_static(india_map)

    ab = st.text_input("API key?", "")

    

    if ab and st.button("Submit"):
        st.session_state.ab = ab
        image_array_list = []
        latitudes = []
        longitudes = []
        idx = 0
        lat_1 = drawn_polygons[0][0][0]
        lon_1 = drawn_polygons[0][0][1]
        lat_2 = drawn_polygons[0][1][0]
        lon_2 = drawn_polygons[0][1][1]
        delta_lat = 0.138
        delta_lon = 0.0023
        latitude = lat_1
        longitude = lon_1


        with st.spinner('Please wait while we process your request...'):
            while latitude <= lat_2:
                while longitude <= lon_2:
                    image_data = get_static_map_image(latitude, longitude, ab)
                    image = Image.open(io.BytesIO(image_data))

        
                    # Get the size of the image (width, height)
                    width, height = image.size
        

                    new_height = height - 20
        
                    # Define the cropping box (left, upper, right, lower)
                    crop_box = (0, 0, width, new_height)
                    
                    # Crop the image
                    image = image.crop(crop_box)

                    new_width = 224
                    new_height = 224

                    # Define the resizing box (left, upper, right, lower)
                    resize_box = (0, 0, new_width, new_height)

                    # Resize the image
                    image = image.resize((new_width, new_height), Image.LANCZOS)

                    if image.mode != 'RGB':
                        image = image.convert('RGB')


                    image_np_array = np.array(image)
                    
                    # image_np_array = np.array(image)
                    
                    
                    image_array_list.append(image_np_array)
                    latitudes.append(latitude)
                    longitudes.append(longitude)
        
                    
                    idx += 1
                    longitude += delta_lon
                    
                    print(idx)
                latitude += delta_lat
            

        images = np.stack(image_array_list, axis=0)


        # images = imgs_input_fn(image_array_list)
        predictions = model.predict(images)
        predictions = [[1 if element >= 0.5 else 0 for element in sublist] for sublist in predictions]

        flat_modified_list = [element for sublist in predictions for element in sublist]
        indices_of_ones = [index for index, element in enumerate(flat_modified_list) if element == 1]
        indices_of_zeros = [index for index, element in enumerate(flat_modified_list) if element == 0]

        temp_dir1 = tempfile.mkdtemp()  # Create a temporary directory to store the images
        with zipfile.ZipFile('images_kiln.zip', 'w') as zipf:
            for i in indices_of_ones:
                temp_df = pd.DataFrame({'Latitude': [latitudes[i]], 'Longitude': [longitudes[i]]})
    
                # Concatenate the temporary DataFrame with the main DataFrame
                df = pd.concat([df, temp_df], ignore_index=True)
    
                image_filename = f'kiln_{latitudes[i]}_{longitudes[i]}.png'
                image_path = os.path.join(temp_dir1, image_filename)

                pil_image = Image.fromarray(image_array_list[i])

                pil_image.save(image_path, format='PNG')
                zipf.write(image_path, arcname=image_filename)

        temp_dir2 = tempfile.mkdtemp()  # Create a temporary directory to store the images
        
        with zipfile.ZipFile('images_no_kiln.zip', 'w') as zipf:
            for i in indices_of_zeros:
                image_filename = f'kiln_{latitudes[i]}_{longitudes[i]}.png'
                image_path = os.path.join(temp_dir2, image_filename)

                pil_image = Image.fromarray(image_array_list[i])

                pil_image.save(image_path, format='PNG')
                zipf.write(image_path, arcname=image_filename)
        
        
        


        csv = df.to_csv(index=False).encode('utf-8')

             

        count_ones = sum(1 for element in flat_modified_list if element == 1)
        count_zeros = sum(1 for element in flat_modified_list if element == 0)

        st.write("The number of brick kilns in the selected region is: ", count_ones)
        st.write("The number of non-brick kilns in the selected region is: ", count_zeros)

        
        with st.expander("Download Options"):
            with open('images_kiln.zip', 'rb') as zip_file:
                zip_data = zip_file.read()
            st.download_button(
                label="Download Kiln Images",
                data=zip_data,
                file_name='images_kiln.zip',
                mime="application/zip"
            )
            with open('images_no_kiln.zip', 'rb') as zip_file:
                zip_data = zip_file.read()
            st.download_button(
                label="Download Non-Kiln Images",
                data=zip_data,
                file_name='images_no_kiln.zip',
                mime="application/zip"
            )
            st.download_button(label =
                "Download CSV of latitude and longitude of brick kilns",
                data = csv,
                file_name = "lat_long.csv",
                mime = "text/csv"
                ) 

        # Cleanup: Remove the temporary directory and zip file
        shutil.rmtree(temp_dir1)
        os.remove('images_kiln.zip')
        shutil.rmtree(temp_dir2)
        os.remove('images_no_kiln.zip')

    else:
        st.sidebar.warning("Please enter an API key.")

if __name__ == "__main__":
    main()
