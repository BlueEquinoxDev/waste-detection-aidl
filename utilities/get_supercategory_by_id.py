import json

def get_supercategory_by_id(input_id):

    # Load the original categories data (input JSON) from the annotations.json file
    with open('data/annotations.json', 'r') as infile:
        annotations_data = json.load(infile)

    # Extract the 'categories' data from the loaded JSON
    categories_data = annotations_data["categories"]

    # Load the new JSON with supercategories and their corresponding ids
    with open('data/supercategories.json', 'r') as infile:
        supercategories_data = json.load(infile)

    # Create a mapping of supercategory to id from the new JSON
    supercategory_map = {item["supercategory"]: item["id"] for item in supercategories_data}
    
    # Find the supercategory corresponding to the input id in the categories data
    for category in categories_data:
        if category["id"] == input_id:
            supercategory = category["supercategory"]
            break
    else:
        return f"ID {input_id} not found in categories data"
    
    # Return the id from the new JSON using the supercategory
    return supercategory_map.get(supercategory, "Supercategory not found in the mapping data")


def get_supercategory_map():

    # Load the original categories data (input JSON) from the annotations.json file
    with open('data/annotations.json', 'r') as infile:
        annotations_data = json.load(infile)

    # Extract the 'categories' data from the loaded JSON
    categories_data = annotations_data["categories"]

    # Load the new JSON with supercategories and their corresponding ids
    with open('data/supercategories.json', 'r') as infile:
        supercategories_data = json.load(infile)

    # Create a mapping of supercategory to id from the new JSON
    supercategory_map = {item["supercategory"]: item["id"] for item in supercategories_data}

    # Create a mapping from the old category ids to the new supercategory ids
    # category_map = {item["id"]: supercategory_map.get(item["supercategory"], "Supercategory not found in the mapping data") for item in supercategories_data}

    # Create a mapping from the old category in categories_data to the new supercategory ids in supercategory_map
    category_map = {item["id"]: supercategory_map.get(item["supercategory"], "Supercategory not found in the mapping data") for item in categories_data}

    return category_map

print(get_supercategory_map()) # Print the supercategory map

# Example usage:
# input_id = 32  # The ID you want to search for
# result = get_supercategory_by_id(input_id)
# print(f"The corresponding ID for supercategory with input ID {input_id} is: {result}")
