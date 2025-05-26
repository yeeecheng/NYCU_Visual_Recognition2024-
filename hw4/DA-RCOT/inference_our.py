import argparse
import os
import torch
import numpy as np
import time, math, glob
from PIL import Image
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import ToTensor

# np.random.seed(1850)
# torch.manual_seed(1850)

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")

parser.add_argument("--model", default="./checkpoint/model_allMRCNet128_255_300_1.0.pth", type=str, help="model path")
parser.add_argument("--save", default="./results/restore_imgs", type=str, help="savepath, Default: results")
parser.add_argument("--saveres", default="./results/res_imgs", type=str, help="savepath, Default: residual")
parser.add_argument("--degset", default="./data/test/degraded/", type=str, help="degraded data")
parser.add_argument("--gpus", default="5", type=str, help="gpu ids")


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

os.makedirs(opt.save, exist_ok=True)
os.makedirs(opt.saveres, exist_ok=True)

Tnet = torch.load(opt.model)["Tnet"]
deg_list = sorted(glob.glob(opt.degset + "*"))

toTensor = ToTensor()

with torch.no_grad():
    for deg_name in deg_list:
        print("Processing ", deg_name)
        deg_img = np.array(Image.open(deg_name).convert('RGB'))
        deg_img = toTensor(deg_img)

        data_degraded = deg_img.unsqueeze(0)

        if cuda:
            Tnet = Tnet.cuda()
            data_degraded = data_degraded.cuda()

        im_output, _ = Tnet(data_degraded)
        im_output = torch.clamp(im_output, 0, 1)

        # Save results
        filename = os.path.basename(deg_name)
        save_image((data_degraded - im_output).data * 3, os.path.join(opt.saveres, filename))
        save_image(im_output.data, os.path.join(opt.save, filename))

print("Generating pred.npz for submission...")

images_dict = {}
for filename in sorted(os.listdir(opt.save)):
    if filename.endswith(".png") and filename[:-4].isdigit():
        file_path = os.path.join(opt.save, filename)
        image = Image.open(file_path).convert('RGB')
        img_array = np.array(image)
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        img_array = img_array.astype(np.uint8)
        images_dict[filename] = img_array

np.savez("pred.npz", **images_dict)
print(f"✅ Saved {len(images_dict)} images to pred.npz")