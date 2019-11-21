import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from model import ConvLSTM
from dataset import LFdata
to_pil = transforms.ToPILImage()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

val_root = '../Datasets/LF-640/'  # validation dataset
check_root = './parameters/fold1/'
output_root = './Results/'
num_levels = 5
if not os.path.exists(output_root):
    os.mkdir(output_root)

def main():
    model = ConvLSTM(input_size=[(56, 56), (56, 56), (56, 56), (112, 112), (224, 224)],
                    input_dim=[512, 512, 256, 128, 64],
                    lstm_input_dim=64,
                    hidden_dim=64,
                    kernel_size=(3, 3),
                    num_levels=num_levels,
                    batch_size=1,
                    bias=True,
                    return_all_levels=False,
                    light_field=True)


    model.load_state_dict(torch.load(os.path.join(check_root, 'model.pth') ))
    model.eval()
    model.cuda()

    val_loader = torch.utils.data.DataLoader(
        LFdata(val_root, datatag='test', is_training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    start_time = time.time()
    with torch.no_grad():
        validation(val_loader, output_root, model)
    diff_time = time.time() - start_time
    print ('Detection took {:.3f}s'.format(diff_time))

def validation(val_loader, output_root, model):
    for (img, gt, img_name, img_size) in tqdm(val_loader):
        img = Variable(img).cuda()
        pred = model(img)
        sal_map = torch.sigmoid(pred[-1]).data.squeeze().cpu().numpy()
        # sal_map = cv2.resize(sal_map, dsize=tuple(img_size))
        plt.imsave(os.path.join(output_root, img_name[0] + '.png'), sal_map, cmap='gray')


if __name__ == '__main__':
    main()