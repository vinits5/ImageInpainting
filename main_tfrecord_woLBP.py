# gsutil -m cp -r gs://experiments_logs/gmm/TOPS/gl/dataset/generator_layers_v2.2_categories_lbp.record /content/

import cv2
import os
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from options import MyOptions
from skimage.measure import compare_ssim, compare_psnr
from model import MyModel
cmd = 'pip install tensorboardX'
os.system(cmd)
cmd = 'pip install torch-summary'
os.system(cmd)
from tensorboardX import SummaryWriter

opt = MyOptions().parse()


import cv2
import os
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# from skimage.measure import compare_ssim, compare_psnr

class Preprocess:
    def __init__(self):
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __call__(self, batch_image, batch_mask):
        batch_image = (batch_image+1)*0.5
        batch_image = tf.image.resize_with_pad(batch_image, 256, 256)
        batch_mask = tf.image.resize_with_pad(batch_mask, 256, 256)

        data = {'I_i': [], 'I_g': [], 'M': [], 'L_i': [], 'L_g': []}
        for image, mask in zip(batch_image, batch_mask):
            # I_i = (image.numpy()+1)*0.5
            I_i = image.numpy()
            mask = mask.numpy()

            I_i = np.array(I_i*255, dtype=np.uint8)
            I_g = copy.deepcopy(I_i)

            I_i = self.transform(I_i)
            I_g = self.transform(I_g)

            mask = transforms.ToTensor()(mask)
            data['I_i'].append(I_i.unsqueeze(0))
            data['I_g'].append(I_g.unsqueeze(0))
            data['M'].append(mask.unsqueeze(0))
        data['I_i'] = torch.cat(data['I_i'])
        data['I_g'] = torch.cat(data['I_g'])
        data['M'] = torch.cat(data['M'])
        return data


def postprocess(img):
    img = img.detach().to('cpu')
    img = img * 127.5 + 127.5
    img = img.permute(0, 2, 3, 1)
    return img.int()


def metrics(real, fake):
    real = postprocess(real)
    fake = postprocess(fake)
    m = (torch.sum(torch.abs(real.float() - fake.float())) / torch.sum(real.float())).float().item()

    a = real.numpy()
    b = fake.numpy()
    ssim = []
    psnr = []
    for i in range(len(a)):
        ssim.append(compare_ssim(a[i], b[i], win_size=11, data_range=255.0, multichannel=True))
        psnr.append(compare_psnr(a[i], b[i], data_range=255))
    return np.mean(ssim), np.mean(psnr), m


def train():
    opt.device = 'cuda:0'

    if not os.path.exists(opt.checkpoints_dir): os.mkdir(opt.checkpoints_dir)
    log_dir = 'tensorboard'
    writer = SummaryWriter(f"{opt.checkpoints_dir}/{log_dir}")

    from dataset_tfrecord import define_dataset
    tfrecord_path = "/content/generator_layers_v2.2_categories_lbp.record"
    batch_size = opt.batchSize
    trainset, trainset_length = define_dataset(tfrecord_path, batch_size, train=True)
    valset, valset_length = define_dataset(tfrecord_path, batch_size, train=False)

    model = MyModel()
    model.initialize(opt)
    dpp = Preprocess()      # data pre-process (dpp)

    from torchsummary import summary
#     summary(model.netG, (4, 256, 256))

    print('Train/Val with %d/%d' % (trainset_length, valset_length))
    for epoch in range(1, 35):
        print('Epoch: %d' % epoch)
        
        train_iterator = iter(trainset)
        num_iterations = int(trainset_length/batch_size)

        epoch_iter = 0
        losses_G, ssim, psnr, mae = [], [], [], []
        for i in range(num_iterations):
            # if i == 1: break
            epoch_iter += opt.batchSize

            data, model_inputs = next(train_iterator)
            inpaint_region = data["inpaint_region"]

            person_cloth = data["person_cloth"]

            data = dpp(person_cloth, inpaint_region)
            try:
                model.set_input(data)
                I_g, I_o, loss_G, loss_G_L2, loss_G_GAN, loss_style, loss_perceptual, loss_multi, loss_D_I_o, loss_D_I_g = model.optimize_parameters()
                s, p, m = metrics(I_g, I_o)
                ssim.append(s)
                psnr.append(p)
                mae.append(m)
                losses_G.append(loss_G.detach().item())
                if i % 100 == 0:
                    print('Tra (%d/%d) G:%5.4f, S:%4.4f, P:%4.2f, M:%4.4f' %
                        (epoch_iter, trainset_length, np.mean(losses_G), np.mean(ssim), np.mean(psnr), np.mean(mae)))
                    print('D1_loss_gen: %4.4f | D1_loss_real: %4.4f | D1_G_loss: %4.4f | recon_loss: %4.4f | \
                        vgg_loss: %4.4f | features_loss: %4.4f | style_loss: %4.4f'%(loss_D_I_o.item(), loss_D_I_g.item()\
                            loss_G_GAN.item(), loss_G_L2.item(), loss_perceptual.item(), loss_multi.item(), loss_style.item()))
            except:
                print("Error")
                pass
        model.save_networks(epoch)
        cmd = f"gsutil -m cp -r {opt.checkpoints_dir}/{log_dir} gs://vinit_helper/cloth_inpainting_gan/cloth_inpainting_local_binary_pattern/{opt.checkpoints_dir.split('/')[1]}"



def test():
    opt.device = 'cuda:0'
    result_dir = 'results'
    if not os.path.exists(result_dir): os.mkdir(result_dir)

    from dataset_tfrecord import define_dataset
    tfrecord_path = "/content/generator_layers_v2.2_categories_lbp.record"
    batch_size = 1
    testset, testset_length = define_dataset(tfrecord_path, batch_size, train=False, test=True)
    dpp = Preprocess()      # data pre-process (dpp)

    print('Test with %d' % (testset_length))

    model = MyModel()
    model.initialize(opt)
    model.load_networks(str(32))     # For irregular mask inpainting

    val_ssim, val_psnr, val_mae, val_losses_G = [], [], [], []
    ids = []

    test_iterator = iter(testset)
    num_iterations = int(testset_length/batch_size)

    def tensor2array(xx):
        xx = xx.detach().cpu()[0]
        xx = xx.permute(1, 2, 0).numpy()
        xx = (xx+1)*0.5
        xx = np.clip(xx, 0, 1)
        if xx.shape[2] == 1:
            return np.concatenate([xx, xx, xx], -1)[:, 32:32+192, :]
        if xx.shape[2] == 3:
            return xx[:, 32:32+192, :]

    with torch.no_grad():
        for i in range(num_iterations):
            # if i == 2: break
            try:
                data, model_inputs = next(test_iterator)
            except:
                break
            inpaint_region = data["inpaint_region"]

            person_cloth = data["person_cloth"]
            cloth_no = int(data['clothno'].numpy()[0])
            person_no = int(data['personno'].numpy()[0])

            data = dpp(person_cloth, inpaint_region)

            model.set_input(data)
            try:
                I_g, I_o, I_i, val_loss_G, I_raw, L_o, mask = model.optimize_parameters(val=True)
            except: 
                print("Error")
                continue

            fig = plt.figure(figsize=(12,10))
            plt.subplot(2, 3, 1)
            plt.imshow(tensor2array(I_i))
            plt.title("Input", fontsize=20)
            plt.subplot(2, 3, 2)
            plt.imshow(tensor2array(mask))
            plt.title("Inpaint Region (M)", fontsize=20)
            plt.subplot(2, 3, 3)
            plt.imshow(tensor2array(I_g))
            plt.title("Ground Truth (GT)", fontsize=20)
            plt.subplot(2, 3, 5)
            plt.imshow(tensor2array(I_raw))
            plt.title("Generator Output (GO)", fontsize=20)
            plt.subplot(2, 3, 6)
            plt.imshow(tensor2array(I_o))
            plt.title("GO*M+GT*(1-M)", fontsize=20)
            plt.savefig(f"{result_dir}/{i}_result.jpg")
            plt.close(fig)
            plt.imsave(f"{result_dir}/{i}_input.jpg", tensor2array(I_i))
            plt.imsave(f"{result_dir}/{i}_mask.png", tensor2array(mask))
            plt.imsave(f"{result_dir}/{i}_output.jpg", tensor2array(I_o))

            val_s, val_p, val_m = metrics(I_g, I_o)
            val_ssim.append(val_s)
            val_psnr.append(val_p)
            val_mae.append(val_m)
            val_losses_G.append(val_loss_G.detach().item())
            
            ids.append(str(cloth_no)+'_'+str(person_no))
    losses = {'ssim': val_ssim, 'val_mae': val_mae, 'psnr': val_psnr, 'loss_G': val_losses_G, 'ids': ids}
    import pandas as pd 
    csv = pd.DataFrame(losses)
    csv.to_csv(f"{result_dir}/losses.csv")

    cmd = f"gsutil -m cp -r {result_dir} gs://vinit_helper/cloth_inpainting_gan/cloth_inpainting_local_binary_pattern/{opt.checkpoints_dir.split('/')[1]}"
    os.system(cmd)  

if __name__ == '__main__':
    cmd = "gsutil -m cp -r gs://experiments_logs/gmm/TOPS/gl/dataset/generator_layers_v2.2_categories_lbp.record /content/"
    if not os.path.exists('/content/generator_layers_v2.2_categories_lbp.record'):
        os.system(cmd)
    if opt.type == 'train':
        train()
    elif opt.type == 'test':
        test()
