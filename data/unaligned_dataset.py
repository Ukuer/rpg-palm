import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        if opt.isTrain:
            self.dir_A = os.path.join(opt.dataroot, "line")  # get the image directory
            self.dir_B = os.path.join(opt.dataroot, "palm")

            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

            random.shuffle(self.A_paths)
            random.shuffle(self.B_paths)

            self.A_size = len(self.A_paths)  # get the size of dataset A
            self.B_size = len(self.B_paths)  # get the size of dataset B
            self.length = min(self.A_size, self.B_size)

            # assert len(self.A_paths) == len(self.B_paths)
            assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        else:
            self.dir_test = os.path.join(opt.dataroot, "test")
            self.test_paths = make_dataset(self.dir_test, opt.max_dataset_size)


        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        if self.opt.isTrain:
            # read a image given a random integer index
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]

            A = Image.open(A_path).convert('RGB')
            B = Image.open(B_path).convert('RGB')

            # apply the same transform to both A and B
            A_transform_params = get_params(self.opt, A.size)
            B_transform_params = get_params(self.opt, B.size)
            A_transform = get_transform(self.opt, A_transform_params, grayscale=(self.input_nc == 1))
            B_transform = get_transform(self.opt, B_transform_params, grayscale=(self.output_nc == 1))

            A = A_transform(A)
            B = B_transform(B)

            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        else:
            # read a image given a random integer index
            test_path = self.test_paths[index]
            A = Image.open(test_path).convert('L')

            # apply the same transform to both A and B
            A_transform_params = get_params(self.opt, A.size)
            A_transform = get_transform(self.opt, A_transform_params, grayscale=(self.input_nc == 1))
            A = A_transform(A)
  

            return {'A': A, 'B': A, 'A_paths': test_path, 'B_paths': test_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.isTrain:
            return self.length
        else:
            return len(self.test_paths)
