# find out the creation date from each json i.e. telegram channel
# input the json files
# output the dictionary with all the names and creation date

import glob
import json
import os

# Specify the directory path containing JSON files
path = os.path.join('json_files')

# Get a list of all JSON files in the directory
file_list = glob.glob(os.path.join(path, '*.json'))

# Initialize a dictionary to store creation information
creation_dict = {}

# Iterate over each JSON file
for file in file_list:
    with open(f'{file}', 'r') as f:
        try:
            # Load the JSON data from the file
            data = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            pass
        
        # Iterate over each JSON object in the data
        for s in data:
            try:
                # Check if the 'id' field is equal to 1
                if s['id'] == 1:
                    # Store the title and date in the creation_dict
                    creation_dict[s['action']['title']] = s['date']
            except KeyError:
                pass

# Iterate over the items in the creation_dict and print them
for key, value in creation_dict.items():
    print(key + '   ' + value)
