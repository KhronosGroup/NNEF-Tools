import os
import tarfile


def compress(dir_path, file_path, compression_level=0):
    target_directory = os.path.dirname(file_path)
    if target_directory and not os.path.exists(target_directory):
        os.makedirs(target_directory)

    with tarfile.open(file_path, 'w:gz', compresslevel=compression_level) as tar:
        for file_ in os.listdir(dir_path):
            tar.add(dir_path + '/' + file_, file_)


def extract(file_path, dir_path):
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(dir_path)
