# Pneumonia Classification using Tensorflow Transfer Learning

This Deep learning project uses Transfer Learning technique to classify 
the uploaded chest X-ray image as a pneumonia case or a healthy one. The tensorflow.keras application used 
here is Xception, which is a smalll and efficient model. The model was trained on 
around 5000 images of both 'NORMAL' and 'PNEUMONIA' X-ray images and an 91.67% accuracy was achieved on test set.
The front end is built using Streamlit, which helps in building web page quickly and the web app is deployed on Amazon EC2 instance.

Web app - http://ec2-3-15-4-32.us-east-2.compute.amazonaws.com:8501

Dataset - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Demo Screenshot

![adasd](https://user-images.githubusercontent.com/37840005/131123746-4795c637-71dd-4067-840b-97b89396683d.PNG)

## How to run
1. Download the code and the dataset from the given link
2. Install requirements.txt using command 'pip install -r requirements.txt'
3. Open the Jupyter Notebook 'Classifier.ipynb' and run all the cells which will train the images and create h5 model (Xception.h5) in the same directory.
4. Run app.py using command 'streamlit run app.py' and then follow the link.

