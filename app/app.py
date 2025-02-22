from flask import Flask, flash, request, redirect, send_file, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import io
import base64

import os
import logging
import pathlib

import torch
from PIL import Image

#from model.vits import ViT
from app.model.wastemaskrcnn import WasteMaskRCNN

from torchvision import transforms

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FILE_NAME = "sample.jpg"
MASK_RCNN_CHECKPOINT = "checkpoint_epoch_8_2025_2_18_21_53.pt"
IDX2CLASS = None
MODEL=None

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FILE_NAME']=FILE_NAME

def _load_MASK_RCNN():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", MASK_RCNN_CHECKPOINT)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    IDX2CLASS = checkpoint['idx2classes']
    MODEL=WasteMaskRCNN(num_classes=len(IDX2CLASS))
    MODEL.load_state_dict(checkpoint['model_state_dict'])


with app.app_context():
    # First load into memory the variables that we will need to predict
    _load_MASK_RCNN()
    

def predict_images(images):
    detections, processed_images = MODEL.evaluate(images=images, idx2class=IDX2CLASS, preprocessing=True)
    return detections, processed_images


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(request):
    images = request.files.to_dict(flat=False)

    saved_files = []

    for image in images:
        # Use secure_filename to avoid directory traversal and other filename issues
        filename = secure_filename(image)
        if filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if not image or not allowed_file(filename):
            return jsonify({"error": f"File '{filename}' has an invalid extension."}), 400
        else:
            try:
                encoded_image = images[image][0].read()
    
                image_bytes = base64.b64decode(encoded_image)
                # Wrap the bytes in a BytesIO object
                image_stream = io.BytesIO(image_bytes)
                # Open the image using Pillow
                img = Image.open(image_stream)
                print(type(img))
                img.verify()  # Verifies that it's a valid image format
                img = Image.open(image_stream)
                # If you need to use the image again (e.g., for saving), reset the stream pointer
                image_stream.seek(0)

                saved_files.append(img)
                """
                # Read the image bytes for validation
                img_bytes = images[image][0].read()
                
                image_stream = io.BytesIO(img_bytes)
                
                # Attempt to open the image using Pillow
                img = Image.open(image_stream)
                img.verify()  # Verifies that it's a valid image format
                img = Image.open(image_stream)
                # If you need to use the image again (e.g., for saving), reset the stream pointer
                image_stream.seek(0)

                saved_files.append(img)
                """
            except Exception as e:
                return jsonify({"error": f"File '{filename}' is not a valid image: {str(e)}"}), 400

    return saved_files, 200

    
@app.route('/')
def hello():
	return "Waste Detection AIDL API"


@app.route('/predict', methods=['POST'])
def run_user_request():
    output, status = process_file(request)
    if status == 400:
        return output, 400
    else:
        images = output
        print(images)
    detections, processed_images = predict_images(images=images) 
    #[Image. for img in processed_images]
    #
    #"images": processed_images,
    
    encoded_imgs = []
    for img in processed_images:
        rawBytes = io.BytesIO()
        img.save(rawBytes, "PNG")
        rawBytes.seek(0)
        encoded_img = base64.b64encode(rawBytes.read())
        encoded_imgs.append(str(encoded_img, encoding='utf-8'))
    
    return jsonify({"detections": detections, "encoded_images": encoded_imgs}), 200


@app.errorhandler(400)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 400

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)
 