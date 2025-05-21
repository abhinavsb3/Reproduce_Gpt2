###############chunked ulpoad wit ptogress bar############################
from googleapiclient.http import MediaFileUpload
from Google import Create_Service
import os

CLIENT_SECRET_FILE = 'client_secret_827351215080-ghfiqr1eknimkcce7nd30gljbb5279oj.apps.googleusercontent.com.json' #google secret key .json folder path

API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
folder_id = ''    #folder id of google drive folder you need to upload
file_names = [''] #your fioe name here
mime_types = [''] #mime type of the file format here

for file_name, mime_type in zip(file_names, mime_types):
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }

    media = MediaFileUpload(
        os.path.join('upld_folder', file_name), 
        mimetype=mime_type,
        resumable=True,
        chunksize=256 * 1024 #256kb chunks
    )

    request = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    )
    
    response = None
    last_progress = -1  # Initialize with a value that will trigger first progress update
    
    while response is None:
        status, response = request.next_chunk()
        if status:
            current_progress = int(status.progress() * 100)
            # Only print if progress increases by at least 1%
            if current_progress > last_progress:
                print('Upload progress: {0}%'.format(current_progress))
                last_progress = current_progress
    
    print(f'Upload of {file_name} Complete!')
    print(f'File ID: {response.get("id")}')
####################without chunks and without progres bar##############################
# from googleapiclient.http import MediaFileUpload
# from Google import Create_Service
# import os

# CLIENT_SECRET_FILE = 'client_secret_827351215080-ghfiqr1eknimkcce7nd30gljbb5279oj.apps.googleusercontent.com.json'

# API_NAME = 'drive'
# API_VERSION = 'v3'
# SCOPES = ['https://www.googleapis.com/auth/drive']

# service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
# folder_id = ''#folder id of google drive folder you need to upload
# file_names = [''] #your file name here
# mime_types = [''] #mime type of file format here

# for file_name, mime_type in zip(file_names, mime_types):
#     file_metadata = {
#         'name': file_name,
#         'parents': [folder_id]
#     }

#     # Upload as a single file without chunking
#     media = MediaFileUpload(
#         os.path.join('upld_folder', file_name), 
#         mimetype=mime_type,
#         resumable=False  # Set to False for single file upload
#     )

#     print('Starting upload of {0}...'.format(file_name))
    
#     # Execute the request directly
#     response = service.files().create(
#         body=file_metadata,
#         media_body=media,
#         fields='id'
#     ).execute()
    
#     print('Upload of {0} Complete!'.format(file_name))
#     print(f'File ID: {response.get("id")}')