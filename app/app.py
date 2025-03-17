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

MODEL_NAME = os.getenv("MODEL_NAME") #MASK2FORMER or MASK_R-CNN
CHECKPOINT = os.getenv("CHECKPOINT") # Checkpoint name

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
FILE_NAME = "sample.jpg"
IDX2CLASS = None
MODEL = None
PREDICT_IMAGE = None

# Hardcoded checkpoint paths for each model
MASK2FORMER_CHECKPOINT = "mask2former_checkpoint_epoch_30_2025_3_5_16_59.pt" 
MASK_R_CNN_CHECKPOINT = "maskrcnn_checkpoint_epoch_10_2025_3_10_11_30.pt"
VIT_CHECKPOINT = "vit_checkpoint_epoch_75_2025_3_15.pt"
RESNET_CHECKPOINT = "resnet_checkpoint_epoch_25_2025_3_12.pt"

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['FILE_NAME']=FILE_NAME

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def flash_with_log(message, category='message'):
    """Flash a message and log it"""
    logger.info(f"FLASH MESSAGE: {message}")
    flash(message, category)

def _load_MASK_RCNN():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", CHECKPOINT)

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))

    IDX2CLASS = checkpoint['idx2classes']
    MODEL=WasteMaskRCNN(num_classes=len(IDX2CLASS))
    MODEL.load_state_dict(checkpoint['model_state_dict'])

def _load_MASK2FORMER():
    global MODEL, IDX2CLASS
    try:
        checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", CHECKPOINT)
        print(f"Loading Mask2Former from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        
        IDX2CLASS = checkpoint['idx2classes']
        print(f"Loaded classes: {list(IDX2CLASS.values())}")
        
        MODEL = WasteMask2Former(num_classes=len(IDX2CLASS))
        MODEL.model.load_state_dict(checkpoint['model_state_dict'])
        print("Mask2Former model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading Mask2Former model: {str(e)}")
        return False

def _load_VIT():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", CHECKPOINT)
    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    
    IDX2CLASS = checkpoint['idx2classes']
    # Update with your actual VIT model implementation
    # MODEL = WasteVIT(num_classes=len(IDX2CLASS))
    # MODEL.load_state_dict(checkpoint['model_state_dict'])
    
def _load_RESNET():
    global MODEL, IDX2CLASS
    checkpoint_path = pathlib.Path(__file__).parent.absolute() / os.path.join("checkpoint", CHECKPOINT)
    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    
    IDX2CLASS = checkpoint['idx2classes']
    # Update with your actual ResNet model implementation
    # MODEL = WasteResNet(num_classes=len(IDX2CLASS))


with app.app_context():
    if MODEL_NAME == "MASK_R-CNN":
        _load_MASK_RCNN()
    elif MODEL_NAME == "MASK2FORMER":
        _load_MASK2FORMER()
    else:
        print("Model Not found")
        exit(0)
    

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


@app.route('/upload_image', methods=['POST'])
def upload_image_web():
    logger.info("Received image upload request")
    global PREDICT_IMAGE
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            flash_with_log("No Image to upload, try selecting one image.")
            return redirect(url_for("hello"))
        file = request.files['file']
        if file.filename == '':
            print('No file attached in request')
            flash_with_log("No Image to upload, try selecting one image.")
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
    """ Makes a prediction using the loaded model """
    global PREDICT_IMAGE, MODEL, MODEL_NAME
    logger.info(f"ROUTE CALLED: /predict_web_image - MODEL: {MODEL_NAME}")
    
    if MODEL is None:
        flash_with_log("No model loaded. Please select a model first.")
        return redirect(url_for("hello"))
    
    try:
        # Check if we have an image to process
        if PREDICT_IMAGE is None:
            flash_with_log("No image selected. Please upload an image first.")
            return redirect(url_for("hello"))
        
        # Decode and process image
        image_bytes = base64.b64decode(PREDICT_IMAGE)
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream)
        
        # Run segmentation prediction using the loaded model
        try:
            _, processed_images = predict_images(images=[img])
            
            # Process and return the result
            rawBytes = io.BytesIO()
            processed_images[0].save(rawBytes, "PNG")
            rawBytes.seek(0)
            encoded_string = base64.b64encode(rawBytes.read()).decode("utf-8")
            
            # Debug info
            print(f"Prediction successful, returning image with length: {len(encoded_string)}")
            
            return render_template('index.html', image=encoded_string), 200
            
        except Exception as e:
            print(f"Error during model prediction: {str(e)}")
            flash_with_log(f"Error during prediction: {str(e)}")
            return redirect(url_for("hello"))
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        flash_with_log(f"Error processing image: {str(e)}")
        return redirect(url_for("hello"))


@app.route('/restart', methods=['POST'])
def restart():
    global PREDICT_IMAGE
    PREDICT_IMAGE = None
    return redirect(url_for("hello"))

@app.route('/load_model/<model_name>', methods=['GET'])
def load_model_endpoint(model_name):
    """Load a specific model based on model name parameter"""
    logger.info(f"ROUTE CALLED: /load_model/{model_name}")
    try:
        global MODEL_NAME, CHECKPOINT, MODEL, IDX2CLASS
        
        if model_name == "MASK2FORMER":
            MODEL_NAME = "MASK2FORMER"
            CHECKPOINT = MASK2FORMER_CHECKPOINT
            _load_MASK2FORMER()
        elif model_name == "MASK_R-CNN":
            MODEL_NAME = "MASK_R-CNN"
            CHECKPOINT = MASK_R_CNN_CHECKPOINT
            _load_MASK_RCNN()
        elif model_name == "VIT":
            # Add VIT model loading functionality
            return jsonify({"error": "VIT model loading not yet implemented"}), 501
        elif model_name == "RESNET":
            # Add RESNET model loading functionality
            return jsonify({"error": "RESNET model loading not yet implemented"}), 501
        else:
            return jsonify({"error": f"Unknown model: {model_name}. Supported models: MASK2FORMER, MASK_R-CNN, VIT, RESNET"}), 400
            
        return jsonify({
            "success": True,
            "message": f"Successfully loaded {model_name} model",
            "model_name": model_name,
            "checkpoint": CHECKPOINT,
            "classes": list(IDX2CLASS.values()) if IDX2CLASS else []
        }), 200
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to load model: {str(e)}"
        }), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    """Test route to verify model works with a basic image"""
    global MODEL, IDX2CLASS, MODEL_NAME
    
    if MODEL is None:
        return jsonify({"error": "No model loaded"}), 400
        
    try:
        # Create a simple test image (solid color)
        test_img = Image.new('RGB', (224, 224), color='white')
        
        # Run prediction
        _, processed_images = predict_images(images=[test_img])
        
        # Return processed image
        rawBytes = io.BytesIO()
        processed_images[0].save(rawBytes, "PNG")
        rawBytes.seek(0)
        encoded_string = base64.b64encode(rawBytes.read()).decode("utf-8")
        
        return jsonify({
            "success": True, 
            "model": MODEL_NAME,
            "test_image": encoded_string
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/log_action', methods=['POST'])
def log_action():
    """Endpoint to receive logs from client"""
    try:
        log_data = request.get_json()
        logger.info(f"CLIENT LOG: {log_data['action']} - {log_data['details']}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logger.error(f"Error logging client action: {str(e)}")
        return jsonify({"success": False}), 500

@app.errorhandler(400)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 400

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)
