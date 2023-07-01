# make a .csv data, that can be used to create a Gephi file

import json
import csv
import re
import glob
import os

# Define the path where the JSON files are located
path = os.path.join('json_files')

# Get a list of all JSON files in the specified path
file_list = glob.glob(os.path.join(path, '*.json'))

# Write the header
channel_list = []
forwarded_list = []

# Loop through each file
for file in file_list:
    with open(file, 'r') as f:
        try:
            files = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            pass

        # Function to extract channel numbers from the JSON data
        def channel_number(files):
            channel_id = []
            for s in files:
                try:
                    if s['fwd_from'] is None:
                        pass
                    else:
                        channel_id.append(s['peer_id']['channel_id'])
                except (KeyError, TypeError):
                    pass
            return channel_id

        # Iterate through the channel numbers and add them to the channel_list
        for item in channel_number(files):
            try:
                channel_list.append(item)
            except TypeError:
                pass

        # Function to extract forwarded channel IDs from the JSON data
        def forwarded_channel_id(files):
            forwarded_messages = []
            for s in files:
                try:
                    if s['fwd_from'] is None:
                        pass
                    else:
                        a = re.search('\d{5,}', str(s['fwd_from']['from_id']))
                        try:
                            forwarded_messages.append(a.group())
                        except AttributeError:
                            pass
                except (KeyError, TypeError):
                    pass
            return forwarded_messages

        # Iterate through the forwarded channel IDs and add them to the forwarded_list
        for item in forwarded_channel_id(files):
            forwarded_list.append(item)

# Create a directory if it doesn't exist
if not os.path.exists("network_csv"):
    os.mkdir(f"network_csv")

# Write to CSV file for network data
network_file = os.path.join('network_csv', 'netzwerk.csv')
with open(network_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Source', 'Target'])
    for item in zip(channel_list, forwarded_list):
        writer.writerow(item)

# Write the header
channel_name = {}

# Loop through each file again
for file in file_list:
    with open(file, 'r') as f:
        try:
            files = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            pass
        for s in files:
            try:
                # Extract channel name and ID from the JSON data
                value = (s['action']['title'])
                key = (s['peer_id']['channel_id'])
                channel_name[key] = value
            except (KeyError, TypeError):
                pass

# Write to CSV file for channel names
channel_name_file = os.path.join('network_csv', 'channel_name.csv')
with open(channel_name_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'Label'])
    for item in channel_name.items():
        writer.writerow(item)
