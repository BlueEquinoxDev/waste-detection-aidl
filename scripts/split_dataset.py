import json
from sklearn.model_selection import train_test_split
import argparse
import os

# Define a function to filter data based on image IDs
def filter_coco_data(image_ids, coco_data):
    return {
        'info': coco_data['info'],
        'images': [image for image in coco_data['images'] if image['id'] in image_ids],
        'annotations': [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in image_ids],
        'scene_annotations': [scene_annotation for scene_annotation in coco_data['scene_annotations'] if scene_annotation['image_id'] in image_ids],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'scene_categories': coco_data['scene_categories']
    }
    
def make_taco_annotations_28_categories():

    taco28_annotation={
        'info': coco_data['info'],
        'images': coco_data['images'],
        'annotations':None,        
        'licenses': coco_data['licenses'],
        'categories': None,
        'scene_categories':[],
        'scene_annotations':[]
    }
    
    #categories
    categories_names=sorted(set([c['supercategory'] for c in coco_data['categories']]))
    taco28_annotation['categories']=[{'supercategory': o,'id':i+1,'name':o}  for i, o in enumerate(categories_names)]
    taco28_annotation['categories'].insert(0,{'supercategory': 'Background','id':0,'name':'Background'})
    
    #annotations
    map=[{'id':o['id'],'category':o['supercategory']} for o in coco_data['categories']]
    def find_supercategory_name(id):
        return list(filter(lambda o: o['id']==id,map))[0]['category']
    
    def find_supercategory_28_id(supercategory_name):
        return list(filter(lambda o: o['supercategory']==supercategory_name,taco28_annotation['categories']))[0]['id']
        
    def change_annotation(a):
        old_category_id=a['category_id']
        new_category_id=find_supercategory_28_id(find_supercategory_name(old_category_id))
        a['category_id']=new_category_id
        return a
    
    taco28_annotation['annotations']=[change_annotation(a) for a in coco_data['annotations']]
    
    return taco28_annotation


def make_taco_annotations_5_categories():
    taco5_annotation={
        'info': coco_data['info'],
        'images': None,
        'annotations':None,        
        'licenses': coco_data['licenses'],
        'categories': None,
        'scene_categories':[],
        'scene_annotations':[]
    }
    
    # Load the taco5_categories.json file
    taco5_categories_path = os.path.join(args.dataset_dir, 'taco5_categories.json')
    with open(taco5_categories_path, 'r') as f:
        taco5_categories = json.load(f)
    
    #categories
    cluster_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(taco5_categories.keys())}
    idx_to_cluster_class = {idx: cls_name for idx, cls_name in enumerate(cluster_class_to_idx.keys())}
    idx_to_keep = set(cluster_class_to_idx.values())
    taco5_annotation['categories']=[{'supercategory': idx_to_cluster_class[c['id']], 'id':i+1,'name':idx_to_cluster_class[c['id']]} for i, c in enumerate(coco_data['categories']) if c['id'] in idx_to_keep]
    taco5_annotation['categories'].append({'supercategory': 'Background', 'id':0, 'name':'Background'})
    map2taco5={c['id']:i+1 for i, c in enumerate(coco_data['categories']) if c['id'] in idx_to_keep}
    
    #annotations
    taco5_annotation['annotations'] = [a for a in coco_data['annotations'] if a['category_id'] in idx_to_keep]
    for a in taco5_annotation['annotations']:
        a['category_id']=map2taco5[a['category_id']]
    
    #images
    image_ids = set([a['image_id'] for a in taco5_annotation['annotations']])
    taco5_annotation['images'] = [i for i in coco_data['images'] if i['id'] in image_ids]
    
    return taco5_annotation

def make_taco_annotations_1_categories():
    taco1_annotation={
        'info': coco_data['info'],
        'images': coco_data['images'],
        'annotations':None,        
        'licenses': coco_data['licenses'],
        'categories': None,
        'scene_categories':[],
        'scene_annotations':[]
    }
    
    taco1_annotation['categories']=[{'supercategory': "waste", 'id':1, 'name': "waste"}]
    taco1_annotation['categories'].append({'supercategory': 'Background', 'id':0, 'name':'Background'})
        
    #annotations
    taco1_annotation['annotations'] = [a for a in coco_data['annotations']]
    for a in taco1_annotation['annotations']:
        a['category_id']=1
        
    return taco1_annotation


# Parse arguments
parser = argparse.ArgumentParser(description='Split dataset into training, validation and testing sets')
parser.add_argument('--dataset_dir', required=True, help='Path to dataset annotations', type=str)
parser.add_argument('--test_percentage', required=False, help='Percentage of images used for the testing set', type=float, default=0.10)
parser.add_argument('--val_percentage', required=False, help='Percentage of images used for the validation set', type=float, default=0.10)
parser.add_argument('--seed', required=False, help='Random seed for the split', type=int, default=123)
parser.add_argument('--verbose', required=False, help='Print information about the split', type=bool, default=False)
parser.add_argument('--dataset_type', required=True, help='Type of dataset to be used, it can be taco1, taco28, taco5 or classification', type=str)

args = parser.parse_args()

# Get annotations path
ann_input_path = os.path.join(args.dataset_dir, 'annotations.json')
    
# Check if the annotations file exists
assert os.path.exists(ann_input_path), 'Annotations file not found'
if args.verbose: print('Annotations file found...')

# Load COCO annotations
with open(ann_input_path, 'r') as f:
    coco_data = json.load(f)
if args.verbose: print('Annotations file loaded...')

custom_annotations_path = None
#Create a file of annotations with 28 supercategories
if args.dataset_type.lower() == "taco28": 
    if args.verbose: print('Create a file of annotations with 28 supercategories annotations28.json...')
    custom_annotations_path = os.path.join(args.dataset_dir, 'annotations28.json')
    with open(custom_annotations_path, 'w') as f:
        annotationns28=make_taco_annotations_28_categories()
        json.dump(annotationns28, f)

elif args.dataset_type.lower() == "taco1":
    if args.verbose: print('Create a file of annotations with 1 supercategory annotations1.json...')
    custom_annotations_path = os.path.join(args.dataset_dir, 'annotations1.json')
    with open(custom_annotations_path, 'w') as f:
        annotationns1=make_taco_annotations_1_categories()
        json.dump(annotationns1, f)

elif args.dataset_type.lower() == "taco5":
    if args.verbose: print('Create a file of annotations with 5 categories annotations5.json...')
    custom_annotations_path = os.path.join(args.dataset_dir, 'annotations5.json')
    with open(custom_annotations_path, 'w') as f:
        annotationns5=make_taco_annotations_5_categories()
        json.dump(annotationns5, f)

elif args.dataset_type.lower() == "classification":
    custom_annotations_path = ann_input_path
    
else:
    raise ValueError('No annotation file selected, select one of the following: --taco28, --taco5 or --classification')

# Load COCO annotations for the dataset to be used
with open(custom_annotations_path, 'r') as f:
    coco_data = json.load(f)
if args.verbose: print(f'Annotations file {custom_annotations_path} loaded...')

# Get image IDs
image_ids = [image['id'] for image in coco_data['images']]

# Split COCO annotations based on image IDs in training, validation and testing sets
train_val_ids, test_ids = train_test_split(image_ids, test_size=args.test_percentage, random_state=args.seed)
train_ids, val_ids = train_test_split(train_val_ids, test_size=args.val_percentage/(1-args.test_percentage), random_state=args.seed)

if args.verbose: print('Annotations split...')

# Create new annotations for training, validation and test sets
if args.verbose: print('Filtering annotations acording to the split...')
train_dataset = filter_coco_data(train_ids, coco_data)
val_dataset = filter_coco_data(val_ids, coco_data)
test_dataset = filter_coco_data(test_ids, coco_data)
if args.verbose: print('Filtering completed...')


# Save the splited COCO annotations in different files
if args.verbose: print('Creating train_annotations.json...')
train_output_path = os.path.join(args.dataset_dir, 'train_annotations.json')
with open(train_output_path, 'w') as f:
    json.dump(train_dataset, f)

if args.verbose: print('Creating validation_annotations.json...')
val_output_path = os.path.join(args.dataset_dir, 'validation_annotations.json')
with open(val_output_path, 'w') as f:
    json.dump(val_dataset, f)

if args.verbose: print('Creating test_annotations.json...')
test_output_path = os.path.join(args.dataset_dir, 'test_annotations.json')
with open(test_output_path, 'w') as f:
    json.dump(test_dataset, f)

if args.verbose: print(f'jsons created in {args.dataset_dir}. Completed!')