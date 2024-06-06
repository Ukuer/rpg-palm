import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import torch

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
# model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

opt.num_test = min(opt.num_test, len(dataset))

if __name__ == '__main__':
# test stage
    for i, data in enumerate(islice(dataset, opt.num_test)):
        model.set_input(data, test_mode=True)
        print('process input image %3.3d/%3.3d' % (i, opt.num_test))

        z_samples = model.get_z_random(opt.n_samples, opt.nz)
        images = []
        names = []
        for nn in range(0, opt.n_samples):
            real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=False)
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)

        # img_path = 'input_%3.3d' % i
        img_path = os.path.basename(data['A_paths'][0]).split('.')[0]
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    webpage.save()
