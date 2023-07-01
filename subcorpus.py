# Create a subcorpus with a keyword
# input are json files in the folder json_files
# output is a .pkl file with dates and messages

import glob
import json
import os
import re
import json
import pickle

# Define the path to the directory containing JSON files
path = os.path.join('json_files')

# Get a list of all JSON files in the directory
file_list = glob.glob(os.path.join(path, '*.json'))

# Initialize lists and dictionaries
messages_list = []
date_list = []
channel_dict = {}
channel_id = []
x = 0

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
                # Check if the length of the 'message' field is greater than or equal to 60
                if len(s['message']) >= 60:
                    # Remove URLs from the 'message' field
                    telegram_messages = re.sub('(?:https?:\/\/|www\.)\S+', '', s['message'])
                    
                    # Check if the 'message' field contains the word 'freiheit' (case-insensitive)
                    if re.search('.*freiheit.*', telegram_messages, re.IGNORECASE):
                        # Append the message, date, and channel ID to the respective lists
                        messages_list.append(telegram_messages)
                        date_list.append(s['date'])
                        channel_id.append(s['peer_id']['channel_id'])
                        
                        # Update the channel dictionary with the channel ID count
                        if s['peer_id']['channel_id'] in channel_dict:                     
                            channel_dict[s['peer_id']['channel_id']] += 1
                        else:
                            channel_dict[s['peer_id']['channel_id']] = 1
                    else:
                        pass
                else:
                    pass
            except KeyError:
                pass

# Create an output folder if it doesn't exist
output_folder = 'output_pkl'
os.makedirs(output_folder, exist_ok=True)

# Save the messages_list, date_list, and channel_id lists to pickle files in the output folder
with open(os.path.join(output_folder, 'messages_list_freiheit.pkl'), 'wb') as f:
    pickle.dump(messages_list, f)
with open(os.path.join(output_folder, 'date_list_freiheit.pkl'), 'wb') as f:
    pickle.dump(date_list, f)

# Print the lengths of messages_list, date_list, and x
print(len(messages_list))
print(len(date_list))
print(x)

# Print the channel_dict: which channel has how many messages with the keyword 
print(channel_dict)
