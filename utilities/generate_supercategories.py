import json

# Load the original data from the annotations file
with open('data/annotations.json', 'r') as infile:
    data = json.load(infile)

# Create a dictionary to store unique supercategories
supercategories = {}
current_id = 0  # Start incremental ID from 0

# Iterate over categories to create unique supercategories with incremental ids
for category in data["categories"]:
    supercategory_name = category["supercategory"]
    if supercategory_name not in supercategories:
        # Assign a new incremental id to the supercategory
        supercategories[supercategory_name] = current_id
        current_id += 1  # Increment the id for the next supercategory

# Prepare the final structure for supercategories
supercategories_data = [{"supercategory": name, "id": id} for name, id in supercategories.items()]

# Create the new JSON document
with open("data/taco28_categories.json", "w") as outfile:
    json.dump(supercategories_data, outfile, indent=4)

print("data/taco28_categories.json file created successfully!")