from flask import Flask, flash, request, render_template, redirect, send_file, url_for, send_from_directory, jsonify, flash
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
from app.model.wastemask2former import WasteMask2Former
from dotenv import load_dotenv
load_dotenv()

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FILE_NAME = "sample.jpg"
MASK_RCNN_CHECKPOINT = "checkpoint_epoch_16_2025_2_28_10_40.pt"#"checkpoint_epoch_8_2025_2_18_21_53.pt"
MASK2FORMER_CHECKPOINT = "checkpoint_epoch_14_mask2former_taco28.pt"#"checkpoint_mask2former_cloud.pt"
IDX2CLASS = None
MODEL = None
PREDICT_IMAGE = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['FILE_NAME']=FILE_NAME

def _load_MASK_RCNN():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", MASK_RCNN_CHECKPOINT)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    IDX2CLASS = checkpoint['idx2classes']
    MODEL=WasteMaskRCNN(num_classes=len(IDX2CLASS))
    MODEL.load_state_dict(checkpoint['model_state_dict'])

def _load_MASK2FORMER():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", MASK2FORMER_CHECKPOINT)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    IDX2CLASS = checkpoint['idx2classes']
    MODEL=WasteMask2Former(num_classes=len(IDX2CLASS))
    MODEL.model.load_state_dict(checkpoint['model_state_dict'])


with app.app_context():
    #_load_MASK_RCNN()
    _load_MASK2FORMER()
    

def predict_images(images):
    detections, processed_images = MODEL.evaluate(images=images, idx2class=IDX2CLASS, preprocessing=True)
    return detections, processed_images


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file(request):
    images = request.files.to_dict(flat=False)
    print(images)
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

            except Exception as e:
                return jsonify({"error": f"File '{filename}' is not a valid image: {str(e)}"}), 400

    return saved_files, 200

    
@app.route('/', methods=['GET'])
def hello():
    global PREDICT_IMAGE
    if PREDICT_IMAGE is None:
        print("here in hello")
        return render_template("index.html", image="")
    else:
        return render_template("index.html", image=PREDICT_IMAGE)


@app.route('/predict', methods=['POST'])
def run_user_request():
    output, status = process_file(request)
    if status == 400:
        return output, 400
    else:
        images = output
        print(images)
    detections, processed_images = predict_images(images=images) 
    
    encoded_imgs = []
    for img in processed_images:
        rawBytes = io.BytesIO()
        img.save(rawBytes, "PNG")
        rawBytes.seek(0)
        encoded_img = base64.b64encode(rawBytes.read())
        encoded_imgs.append(str(encoded_img, encoding='utf-8'))
    
    return jsonify({"detections": detections, "encoded_images": encoded_imgs}), 200


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image_web():
    "Saves one image in memory to predict" 
    global PREDICT_IMAGE
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            flash("No Image to upload, try selecting one image.")
            return redirect(url_for("hello"))
        file = request.files['file']
        if file.filename == '':
            print('No file attached in request')
            flash("No Image to upload, try selecting one image.")
            return redirect(url_for("hello"))
        if file:
            if not allowed_file(file.filename):
                return jsonify({"error": f"File '{file.filename}' has an invalid extension."}), 400
            #filename = secure_filename(file.filename)
            img = Image.open(file.stream)
            with io.BytesIO() as buf:
                img.save(buf, 'jpeg')
                image_bytes = buf.getvalue()
            encoded_string = base64.b64encode(image_bytes).decode("utf-8")
            PREDICT_IMAGE = encoded_string
        return render_template('index.html', image=encoded_string), 200
    else:
        return redirect(url_for("hello")), 200


@app.route('/predict_web_image', methods=['POST'])
def run_user_request_web():
    """ Makes a prediction using a model for image segmentation"""
    global PREDICT_IMAGE
    if PREDICT_IMAGE is None:
        return redirect(url_for("hello"))
    else:
        image_bytes = base64.b64decode(PREDICT_IMAGE)
        # Wrap the bytes in a BytesIO object
        image_stream = io.BytesIO(image_bytes)
        # Open the image using Pillow
        img = Image.open(image_stream)
        img.verify()  # Verifies that it's a valid image format
        img = Image.open(image_stream)
                    # If you need to use the image again (e.g., for saving), reset the stream pointer
        image_stream.seek(0)

        _, processed_images = predict_images(images=[img]) 
        
        rawBytes = io.BytesIO()
        processed_images[0].save(rawBytes, "PNG")
        rawBytes.seek(0)

        encoded_string = base64.b64encode(rawBytes.read()).decode("utf-8")
    return render_template('index.html', image=encoded_string), 200

@app.route('/restart', methods=['POST'])
def restart():
    global PREDICT_IMAGE
    PREDICT_IMAGE = None
    return redirect(url_for("hello"))

@app.errorhandler(400)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 400

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)
 