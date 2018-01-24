
from __future__ import print_function

import os
import sys
import requests


# def download_file_from_google_drive(id, destination):
#     URL = "https://docs.google.com/uc?export=download"
#
#     session = requests.Session()
#
#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     token = get_confirm_token(response)
#
#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)
#
#     save_response_content(response, destination)
#
#
# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value
#
#     return None
#
#
# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768
#
#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)


def download_file(file_source, filename):
    """ Download requested file """

    # check which python version is running
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(source, file_name):
        print("Downloading %s" % file_name, end="...")
        urlretrieve(source, file_name)
        print("done!")

    # download file if it does not exist yet
    if not os.path.exists(filename):
        download(file_source, filename)

if __name__ == """__main__""":
    """ main """
    download_file("http://www.cp.jku.at/people/dorfer/cca_layer/iapr.npz", "data/iapr.npz")
    download_file("http://www.cp.jku.at/people/dorfer/cca_layer/audio_sheet_music.npz", "data/audio_sheet_music.npz")

