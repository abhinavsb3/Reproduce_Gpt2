import os
import io
from googleapiclient.http import MediaIoBaseDownload
from Google import Create_Service

CLIENT_SECRET_FILE = 'drive_config/client_secret_827351215080-ghfiqr1eknimkcce7nd30gljbb5279oj.apps.googleusercontent.com.json'#google secret key .json folder path

API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

file_ids = ['1QMQGqf5HUhE11L88YHFQjHGPD6qBCLcx'] #file id of the file in google drive you need to download
file_names = ['model_00350.pt'] #names of files you need to download(this is for mention in what names you want to see the folder in your system)

for file_id, file_name in zip(file_ids, file_names):
    request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fd=fh, request=request)
    done = False

    while not done:
        status, done = downloader.next_chunk()
        print('Download progress {0}'.format(status.progress() * 100))

    fh.seek(0)

    with open(os.path.join('dwnld_folder', file_name), 'wb') as f:
        f.write(fh.read())
        f.close()
