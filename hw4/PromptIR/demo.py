import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from net.model import PromptIR

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import Image

def pad_input(input_,img_multiple_of=8):
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        return input_,height,width

def tile_eval(model,input_,tile=128,tile_overlap =32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    restored = E.div_(W)

    restored = torch.clamp(restored, 0, 1)
    return restored
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        # self.loss_fn  = nn.L1Loss()

    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        # self.log("train_loss", loss)
        # return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--test_path', type=str, default="test/demo/", help='save path of test images, can be directory or an image')
    parser.add_argument('--output_path', type=str, default="output/demo/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="./epoch=199-step=32000.ckpt", help='checkpoint save path')
    parser.add_argument('--tile',type=bool,default=False,help="Set it to use tiling")
    parser.add_argument('--tile_size', type=int, default=128, help='Tile size (e.g 720). None means testing on the original resolution image')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    opt = parser.parse_args()


    ckpt_path = "train_ckpt/" + opt.ckpt_name
    # ckpt_path = "/swim-pool/yicheng/NYCU_Visual_Recognition2024-/hw4/PromptIR/wandb/run-20250526_171247-p0jfsbt3/train_ckpt_v2/epoch=158-step=25440.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # construct the output dir
    subprocess.check_output(['mkdir', '-p', opt.output_path])

    np.random.seed(0)
    torch.manual_seed(0)

    # Make network
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)
    net  = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    net.eval()

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)

            if opt.tile is False:
                restored = net(degrad_patch)
            else:
                print("Using Tiling")
                degrad_patch,h,w = pad_input(degrad_patch)
                restored = tile_eval(net,degrad_patch,tile = opt.tile_size,tile_overlap=opt.tile_overlap)
                restored = restored = restored[:,:,:h,:w]

            save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')



    print("Generating pred.npz for submission...")

    images_dict = {}
    for filename in sorted(os.listdir(opt.output_path)):
        if filename.endswith(".png") and filename[:-4].isdigit():
            file_path = os.path.join(opt.output_path, filename)
            image = Image.open(file_path).convert('RGB')
            img_array = np.array(image)
            img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
            img_array = img_array.astype(np.uint8)
            images_dict[filename] = img_array

    np.savez("pred.npz", **images_dict)
print(f"✅ Saved {len(images_dict)} images to pred.npz")