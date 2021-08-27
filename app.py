import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.xception import preprocess_input


def pneumonia_classifier(input_image):
    #Load the model
    model = load_model('Xception.h5')
    #Convert the image to numpy array
    image = img_to_array(input_image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    result = model.predict(image)
    return result[0][0]
   

def main():

    #Sidebar
    with st.sidebar.beta_expander('About'):
        st.write('This Deep learning  web app uses Transfer Learning(Xception CNN) to classify the uploaded chest X-ray image as a pneumonia case or a healthy one. The front end is built using Streamlit.')

    with st.sidebar.beta_expander('Contact'):
        st.write('[GitHub](https://github.com/VaisakNair7/Stroke-Prediction)')
        st.write('[LinkedIn](https://www.linkedin.com/in/vaisaksnair/)')
        st.write('Mail : vaisaksnair98@gmail.com')
    
    html_temp = """
    <div style="background-color:#34568B;padding:2px">
    <h1 style="color:white;text-align:center;">Pneumonia/Normal lungs classifier</h1>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    #Get image from user.
    st.write('Upload chest X-ray image below.')
    uploaded_image = st.file_uploader(' ', type = ['png', 'jpg', 'jpeg'])

    #Prediction
    if st.button('Predict'):

        if uploaded_image is not None:

            #Read the image
            image = Image.open(uploaded_image)
            image.save('Images/tmp.jpeg')
            image = load_img('Images/tmp.jpeg', target_size = (299, 299))

            result = pneumonia_classifier(image)

            if result == 1:
                st.error('PNEUMONIA LUNGS')
            else:
                st.success('NORMAL LUNGS')

            #Display the uploaded image
            st.image(image, caption = 'Uploaded Image.', use_column_width = True)

        else:
            st.error('Please uplaod chest X-ray image.')
    

if __name__ == '__main__':
    main()

