from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import numpy as np

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def download_file_to_cwd(url, filename):
    if not isfile(filename):
        with DLProgress(unit='B', unit_scale = True, miniters=1, desc='Downloading {}'.format(filename)) as pbar:
            urlretrieve(url, filename, pbar.hook)
    print('{} is downloaded to ./{}'.format(url, filename))

def untar_to_cwd(filename, outdir):
    if not isdir(outdir):
        with tarfile.open(filename) as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
    print('./{} is untar to ./{}'.format(filename, outdir))


class Dataset(object):
    def __init__(self, X_train, y_train, batch_size, dtype = np.float32, seed = None):
        self.X_train_ = X_train.astype(dtype)
        self.y_train_ = y_train
        self.N_ = self.X_train_.shape[0]

        self.batch_size_ = batch_size
        self.iters_per_epoch_ = max(1, self.N_ // self.batch_size_)
        self.seed_ = seed

        if self.seed_ is not None:
            np.random.seed(self.seed_)

    def batch_size(self):
        return self.batch_size_

    def reset(self):
        if self.seed_ is not None:
            np.random.seed(self.seed_)

    def next_batch(self):
        if self.batch_size_ == self.N_:
            return self.X_train_, self.y_train_
        else:
            idx = np.random.choice(self.N_, self.batch_size_)
            return self.X_train_[idx], self.y_train_[idx]

    def get_nb_iters(self, epochs):
        return self.iters_per_epoch_ * epochs

    def is_epoch_end(self, i):
        epoch_end = (i % self.iters_per_epoch_) == 0
        epoch = i // self.iters_per_epoch_
        return epoch_end, epoch