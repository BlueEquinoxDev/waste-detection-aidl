'''
This script downloads TACO's images from Flickr given an annotation json file
Code written by Pedro F. Proenza, 2019
'''

import os.path
import argparse
import json
from PIL import Image, ImageOps
import requests
from io import BytesIO
import sys

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default= './data/annotations.json', help='Path to annotations')
parser.add_argument('--images_path', required=False, default= './data/images', help='Path to store images')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

# FIX annotations issue
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())
    ann_ids = []
    for i, ann in enumerate(annotations["annotations"], start=1):
        ann["id"]=i

with open(args.dataset_path, 'w') as f:
    f.write(json.dumps(annotations))

# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):

        image = annotations['images'][i]

        file_name = image['file_name']
        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_name = file_name.replace(os.sep, '_')  # Avoid generating unnecessary subdirectories
        annotations['images'][i]['file_name'] = file_name # Update the file name in the annotations

        file_path = os.path.join(args.images_path, file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            # Load and Save Image
            response = requests.get(url_original)
            img = Image.open(BytesIO(response.content))
            if img._getexif():
                img = ImageOps.exif_transpose(img)  # Avoid issues of rotated images by rotating it accoring to EXIF info
                img.save(file_path) # img.save(file_path, exif=img.info["exif"]) --> Previous code
            else:
                img.save(file_path)

        # Show loading bar
        bar_size = 30
        x = int(bar_size * i / nr_images)
        sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i, nr_images))
        sys.stdout.flush()
        i+=1


    sys.stdout.write('Finished\n')

with open(os.path.join(dataset_dir, "annotations.json"), 'w') as f:
    f.write(json.dumps(annotations))
