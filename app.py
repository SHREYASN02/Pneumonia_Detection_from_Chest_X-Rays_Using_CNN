import logging
import os
from flask import Flask,render_template,request
from tensorflow.keras.utils import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

# Configure logging to integrate with Gunicorn
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# from gevent.pywsgi import WSGIServer


app = Flask(__name__)

# Define absolute paths for model and static files
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, 'models/pneu_cnn_model.h5')
STATIC_PATH = os.path.join(APP_ROOT, 'static')

app.logger.info("Loading model...")
model = load_model(MODEL_PATH)
app.logger.info("Model loaded successfully.")

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def predict():
    try:
        imagefile= request.files["imagefile"]
        image_path = os.path.join(STATIC_PATH, imagefile.filename)
        imagefile.save(image_path)
        img=load_img(image_path,target_size=(500,500),color_mode='grayscale')
        x=img_to_array(img)
        x=x/255
        x=np.expand_dims(x, axis=0)
        classes=model.predict(x)
        result1=classes[0][0]
        result2='Negative'
        if result1>=0.5:
            result2='Positive'
        classification ='%s (%.2f%%)' %(result2,result1*100)
        return render_template('index.html',prediction=classification,imagePath=image_path)
    except Exception as e:
        app.logger.exception(f"Error in predict endpoint {request.path}: {e}")
        return render_template('index.html', prediction="Error during prediction. Please try again.", imagePath=None)