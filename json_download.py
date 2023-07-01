# based on https://github.com/amiryousefi/telegram-analysis
# modified to loop through the telegram channels and find linked channels
# one has to sign up with the Telegram API and sign in credentials in config.ini

import configparser  # Library for reading configuration files
import json  # Library for working with JSON data
import os  # Library for interacting with the operating system
import re  # Library for regular expressions
import glob  # Library for retrieving file paths using patterns
from datetime import date, datetime  # Libraries for working with dates and times

from telethon import TelegramClient  # Library for interacting with the Telegram API
from telethon.errors import ChannelPrivateError, SessionPasswordNeededError  # Exceptions for handling Telegram API errors
from telethon.tl.functions.messages import GetHistoryRequest  # Telegram API request for retrieving message history
from telethon.tl.types import PeerChannel  # Telegram API type for representing a channel

base_dir = os.path.join(os.getcwd())

# Folder for storing all the data
folder_path = os.path.join(base_dir, 'data')
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_list = glob.glob(os.path.join(base_dir, 'data', 'jsondata', '*.json'))
channel_downloaded = []
for file in file_list:
    search_id = re.search('\d{5,}', file)
    channel_id = search_id.group()
    channel_downloaded.append(channel_id)
# some functions to parse json date
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()

        if isinstance(o, bytes):
            try:
                return list(o)
            except ChannelPrivateError:
                pass
        return json.JSONEncoder.default(self, o)


# Reading Configs
config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']

api_hash = str(api_hash)

phone = config['Telegram']['phone']
username = config['Telegram']['username']

# Create the client and connect

client = TelegramClient(username, api_id, api_hash)
# starting channel for the loop
# 1556925582 = FreeMichaleBallweg (German)
# 1170416613 = Mass-Voll (Swiss)
first_channel = ['1227574204']
id_number = first_channel
async def main(phone):
    await client.start()
    print("Client Created")
    # Ensure you're authorized
    if await client.is_user_authorized() == False:
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone, input('Enter the code: '))
        except SessionPasswordNeededError:
            await client.sign_in(password=input('Password: '))
    me = await client.get_me()
    for number in id_number:
        print('currently working on ' + number)
        user_input_channel = number
        if user_input_channel in channel_downloaded:
            print('already downloaded')
        else:
            if user_input_channel.isdigit():
                try:
                    entity = PeerChannel(int(user_input_channel))
                except ChannelPrivateError:
                    pass
            else:
                entity = user_input_channel
            try:
                my_channel = await client.get_entity(entity)
                offset_id = 0
                limit = 100
                all_messages = []
                total_messages = 0
                total_count_limit = 0
                date_of_post = datetime(2023, 3, 15)
                while True:
                    print("Current Offset ID is:", offset_id, "; Total Messages:", total_messages)
                    history = await client(GetHistoryRequest(
                        peer=my_channel,
                        offset_id=offset_id,
                        offset_date=date_of_post,
                        add_offset=0,
                        limit=limit,
                        max_id=0,
                        min_id=0,
                        hash=0
                    ))
                    if not history.messages:
                        break
                    messages = history.messages
                    for message in messages:
                        all_messages.append(message.to_dict())
                    offset_id = messages[len(messages) - 1].id
                    total_messages = len(all_messages)
                    if total_count_limit != 0 and total_messages >= total_count_limit:
                        break
                      # create outfile with channel number

                    directory_path = os.path.join(base_dir, 'data', 'jsondata')
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)

                    file_path = os.path.join(directory_path, f'{number}.json')
                    with open(file_path, 'w') as outfile:
                        json.dump(all_messages, outfile, cls=DateTimeEncoder)

                    with open(f'{directory_path}/{number}.json',
                              'r') as f:
                        # read the contents of the file
                        data = json.loads(f.read())
                    # make a list with channels that are linked in a channel
                    channel_list = []
                    for s in data:
                        try:
                            if s['fwd_from'] is None:
                                channel_list.append(s['fwd_from'])
                            else:
                                a = re.search('\d{5,}', str(s['fwd_from']['from_id']))
                                try:
                                    channel_list.append(a.group())
                                except AttributeError:
                                    pass
                        except KeyError:
                            channel_list.append(None)
                    # dictionary for the occurrences of each channel
                    channel_count = {}
                    for item in channel_list:
                        if item == None:
                            pass
                        else:
                            if item in channel_count:
                                channel_count[item] += 1
                            else:
                                channel_count[item] = 1
                    # add the channel_id into the list of channels to download. If they're already in the list, then skip
                    for key, value in channel_count.items():
                        if value >= 0:
                            if key in id_number:
                                pass
                            else:
                                id_number.append(key)
                        else:
                            pass
                        folder_path = os.path.join(base_dir, f'jsondata/{number}.json')
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
            except ValueError:
                pass
            except ChannelPrivateError:
                print(f'{number} could not be downloaded')


with client:
    client.loop.run_until_complete(main(phone))

