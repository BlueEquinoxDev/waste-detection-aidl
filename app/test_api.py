import base64
import json 
import io
from PIL import Image

import requests
import pathlib
import os

api = 'http://localhost:8000/predict'
image_file = 'sample.jpg'

image_path = pathlib.Path(__file__).parent.absolute() / image_file

with open(image_path, "rb") as f:
    buffer = f.read()
    img_base64 = base64.b64encode(buffer).decode("utf-8")       
#im_b64 = base64.b64encode(im_bytes).decode("utf8")

headers = {'Content-type': 'application/json'}
  
files = [
    (image_file, img_base64),
]

response = requests.post(url=api, files=files)
print("HERE!")
if response.status_code == 200:
    data = response.json()
    #print(data)
    # print(data["detections"][0]["boxes"])
    for i, encoded_image in enumerate(data["encoded_images"]):
        
        image_bytes = base64.b64decode(encoded_image)
        # Wrap the bytes in a BytesIO object
        image_stream = io.BytesIO(image_bytes)
        # Open the image using Pillow
        img = Image.open(image_stream)
        img.save(os.path.join(pathlib.Path(__file__).parent.absolute(), "results", f"image_{i}.png"))
        img.show()
    
    json_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "results", f"detections.json")
    with open(json_path, "w") as f: 
        json.dump(data["detections"], f)
    print("Detections saved in ")

else:
    try:
        print(response.json())
    except requests.exceptions.RequestException:
        print(response.text)

     