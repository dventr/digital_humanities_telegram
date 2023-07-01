# check the language of each .json file
# pip install langdetect
# takes as input the json_files folder
# prints out the language of each message, that are not german
# if there is German nothing will show up, however if there are
# other languages, manual checking of the data is necessary


from langdetect import detect
import glob
import os
import json
import re

# Define the path to the directory containing JSON files
path = os.path.join('json_files')

# Get a list of all JSON files in the directory
file_list = glob.glob(os.path.join(path, '*.json'))


for files in file_list:
    with open(f'{files}', 'r') as f:
        number = re.search("\d{5,}", files)
        number1 = number.group()
        files = json.loads(f.read())
        messages = []
        for s in files:
            try:
                messages.append(s['message'])
                x = 0
                if len(messages) > 5:
                    try:
                        if detect(messages[0]) != 'de':
                            print(f'{number1} ' + detect(messages[0]), detect(messages[1]), detect(messages[2]))
                    except:
                        language = "error"
                        try:
                            if detect(messages[2]) != 'de':
                                print(f'{number1} ' + detect(messages[2]))
                        except:
                            language = "error"
                            try:
                                if detect(messages[3]) != 'de':
                                    print(f'error: check language of {number1}')
                            except:
                                language = 'error'
                    x += 1
                    break
            except KeyError:
                messages.append('NONE')