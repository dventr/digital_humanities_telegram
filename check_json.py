# check for faulty downloads of channels
# if the case, it will print out the channel number
# input json_files
# output prints out faulty json files
import json
import os
import glob
import re

# Specify the directory path containing JSON files
path = os.path.join('json_files')

# Get a list of all JSON files in the directory
file_list = glob.glob(os.path.join(path, '*.json'))

# Initialize a list to store channel IDs that need to be redownloaded
redo_download = []

# Iterate over each JSON file
for file in file_list:
    with open(f'{file}', 'r') as f:
        try:
            # Try to load the JSON data from the file
            json.loads(f.read())
            print('correct')  # JSON file is correctly formatted
        except ValueError:
            print(f'{file} was incorrectly downloaded')  # JSON file is incorrectly downloaded
            
            # Extract the channel ID from the file name using regular expressions
            number = re.search('\d{5,}', file)
            channel_id = number.group()
            
            # Add the channel ID to the redo_download list
            redo_download.append(f'{channel_id}')

# Print the redo_download list
print(redo_download)
