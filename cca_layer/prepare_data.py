
from __future__ import print_function

import os
import sys
from config.settings import DATA_ROOT


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

    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)

    download_file("http://www.cp.jku.at/people/dorfer/cca_layer/iapr.npz",
                  os.path.join(DATA_ROOT, "iapr.npz"))
    download_file("http://www.cp.jku.at/people/dorfer/cca_layer/audio_sheet_music.npz",
                  os.path.join(DATA_ROOT, "audio_sheet_music.npz"))

