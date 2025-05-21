import argparse, os, glob
import torch, pdb
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import math, random, time
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model_res import *
from util.universal_dataset import TrainDataset
from torchvision.utils import save_image
from utils import unfreeze, freeze
from scipy import io as scio
import torch.nn.functional as F
import random
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default=None, type=str,
                    help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=16, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pairnum", default=10000000, type=int, help="num of paired samples")

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='./data/train/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of dehazing saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/lowlight/',
                    help='where training images of deraining saves.')
parser.add_argument('--desnow_dir', type=str, default='data/Train/Desnow/',
                    help='where training images of deraining saves.')


parser.add_argument("--Sigma", default=10000, type=float)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--optimizer", default="RMSprop", type=str, help="optimizer type")
parser.add_argument("--backbone", default="MRCNet", type=str, help="architecture name")
parser.add_argument("--type", default="Deraining", type=str, help="to distinguish the ckpt name ")
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/', help='where clean images of denoising saves.')

parser.add_argument('--lambda_pixel', type=float, default=1.0, help='weight of pixel loss')
parser.add_argument('--lambda_freq', type=float, default=1.0, help='weight of frequency loss')
parser.add_argument('--lambda_contrast', type=float, default=0.05, help='weight of contrastive loss')

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def main():
    global opt, Tnet, writer
    opt = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("./logs", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)
    os.makedirs(os.path.join("runs", opt.type + "_" + opt.backbone), exist_ok=True)
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    patch_size = opt.patch_size
    batch_size = opt.batchSize
    cudnn.benchmark = True

    print("------Datasets loaded------")
    if opt.backbone == 'RCNet':
        Tnet = RCNet(decoder=True)
    elif opt.backbone == 'MRCNet':
        Tnet = MRCNet(decoder=True)
    else:
        Tnet = PromptIR(decoder=True)

    print("*****Using " + opt.backbone + " as the backbone architecture******")
    Fnet = F_net(patch_size)

    print("------Network constructed------")
    if cuda:
        Tnet = Tnet.cuda()
        Fnet = Fnet.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            Tnet.load_state_dict(checkpoint["Tnet"].state_dict())
            Fnet.load_state_dict(checkpoint["Fnet"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            Tnet.load_state_dict(weights['model'].state_dict())
            Fnet.load_state_dict(weights['discr'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("------Using Optimizer: '{}' ------".format(opt.optimizer))

    if opt.optimizer == 'Adam':
        T_optimizer = torch.optim.Adam(Tnet.parameters(), lr=opt.lr / 2)
        F_optimizer = torch.optim.Adam(Fnet.parameters(), lr=opt.lr)
    elif opt.optimizer == 'RMSprop':
        T_optimizer = torch.optim.RMSprop(Tnet.parameters(), lr=opt.lr / 2)
        F_optimizer = torch.optim.RMSprop(Fnet.parameters(), lr=opt.lr)

    T_scheduler = CosineAnnealingLR(T_optimizer, T_max=opt.nEpochs, eta_min=1e-6)
    F_scheduler = CosineAnnealingLR(F_optimizer, T_max=opt.nEpochs, eta_min=1e-6)
    print("------Training------")

    train_set = TrainDataset(opt)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                                      batch_size=opt.batchSize, shuffle=True, pin_memory=True)
    os.makedirs("checksample/all", exist_ok=True)
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        mes_loss, t_loss, p_loss = train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch)

        T_scheduler.step()
        F_scheduler.step()

        save_checkpoint(Tnet, Fnet, epoch)
        writer.add_scalar("Epoch/Loss_T", t_loss, epoch)
        writer.add_scalar("Epoch/Loss_F", p_loss, epoch)
        writer.add_scalar("Epoch/Loss_mse", mes_loss, epoch)
    writer.close()

def train(training_data_loader, T_optimizer, F_optimizer, Tnet, Fnet, epoch):
    pixel_losses, T_losses, D_losses = [], [], []

    print("Epoch={}, lr={}".format(epoch, F_optimizer.param_groups[0]["lr"]))
    pbar = tqdm(enumerate(training_data_loader), total=len(training_data_loader), desc=f"Epoch {epoch}")

    for iteration, batch in pbar:
        ([_, de_id], degraded, target) = batch

        if opt.cuda:
            degraded, target = degraded.cuda(), target.cuda()

        ####### 1. Discriminator update (F-sub optimization) #######
        freeze(Tnet);
        unfreeze(Fnet);
        for _ in range(1):
            Fnet.zero_grad()

            out_real = Fnet(target).squeeze()
            F_real_loss = -out_real.mean()

            out_restored, _ = Tnet(degraded)
            out_fake = Fnet(out_restored.data).squeeze()
            F_fake_loss = out_fake.mean()

            F_train_loss = F_real_loss + F_fake_loss
            F_train_loss.backward()
            F_optimizer.step()

            # gradient penalty
            Fnet.zero_grad()
            alpha = torch.rand(target.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(target)
            interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * out_restored.data, requires_grad=True)

            out_interp = Fnet(interpolated1).squeeze()

            # Computes and returns the sum of gradients of outputs with respect to the inputs.
            grad = \
                torch.autograd.grad(outputs=out_interp,  # outputs (sequence of Tensor) – outputs of the differentiated function
                                    inputs=interpolated1,
                                    # inputs (sequence of Tensor) – Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
                                    grad_outputs=torch.ones(out_interp.size()).cuda(),
                                    # grad_outputs (sequence of Tensor) – The “vector” in the vector-Jacobian product. Usually gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            f_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = 10 * f_loss_gp

            gp_loss.backward()
            F_optimizer.step()
            del  gp_loss, f_loss_gp

        D_losses.append(F_train_loss.item())

        ####### 2. Generator update (T-sub optimization) #######
        freeze(Fnet);
        unfreeze(Tnet)
        Tnet.zero_grad();

        out_restored, res_emd = Tnet(degraded)
        res = target - out_restored

        # Pixel-wise L1 Loss (better for PSNR)
        pixel_loss = F.l1_loss(out_restored, target)

        # Frequency-domain residual penalty
        res_fft = torch.fft.fft2(res)
        if (de_id < 3).any():
            freq_loss = torch.mean(torch.abs(res_fft) ** 2)
        else:
            freq_loss = torch.mean(torch.abs(res_fft))

        # Contrastive loss over residual embeddings
        pos_sim, neg_sim = 0.0, 0.0
        for i in range(res_emd.size(0)):
            z1 = F.normalize(res_emd[i].mean(dim=(1, 2)), dim=0)
            for j in range(i + 1, res_emd.size(0)):
                z2 = F.normalize(res_emd[j].mean(dim=(1, 2)), dim=0)
                sim = torch.exp(torch.dot(z1, z2) / 0.07)
                if de_id[i] == de_id[j]:
                    pos_sim += sim
                else:
                    neg_sim += sim
        contrastive_loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)) if pos_sim > 0 and neg_sim > 0 else torch.tensor(0.0).cuda()

        # Adversarial loss
        adv_loss = -Fnet(out_restored).squeeze().mean()

        # Final total loss with PSNR-oriented weighting
        T_train_loss = (
            adv_loss +
            opt.lambda_pixel * pixel_loss +
            opt.lambda_freq * freq_loss +
            opt.lambda_contrast * contrastive_loss
        )

        T_train_loss.backward()
        T_optimizer.step()

        pixel_losses.append(pixel_loss.item())
        T_losses.append(T_train_loss.item())

        global writer
        global_step = epoch * len(training_data_loader) + iteration
        writer.add_scalar("Loss/Loss_F", F_train_loss.item(), global_step)
        writer.add_scalar("Loss/Loss_T", T_train_loss.item(), global_step)
        writer.add_scalar("Loss/Loss_mse", pixel_loss.item(), global_step)

        pbar.set_postfix({
            'Loss_F': F_train_loss.item(),
            'Loss_T': T_train_loss.item(),
            'Loss_mse': pixel_loss.item()
        })

        del T_train_loss, F_train_loss, z1, z2

    # Evaluate PSNR on random training samples
    def calculate_psnr(pred, target, max_val=1.0):
        mse = F.mse_loss(pred, target)
        psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
        return psnr.item()

    Tnet.eval()
    with torch.no_grad():
        sampled_psnrs = []
        num_samples = 4
        for i, batch in enumerate(training_data_loader):
            if i >= num_samples:
                break
            ([_, _], degraded, target) = batch
            if opt.cuda:
                degraded, target = degraded.cuda(), target.cuda()
            restored, _ = Tnet(degraded)
            psnr = calculate_psnr(restored, target)
            sampled_psnrs.append(psnr)

        avg_sample_psnr = sum(sampled_psnrs) / len(sampled_psnrs)
        writer.add_scalar("Eval/TrainSample_PSNR", avg_sample_psnr, epoch)
        print(f"Sample PSNR (Train): {avg_sample_psnr:.2f} dB")
    Tnet.train()

    return torch.mean(torch.FloatTensor(pixel_losses)), torch.mean(torch.FloatTensor(T_losses)), torch.mean(
        torch.FloatTensor(D_losses))


def save_checkpoint(Tnet, Fnet, epoch):
    model_out_path = "checkpoint/" + "model_" + str(opt.type) + opt.backbone + str(opt.patch_size) + "_" + str(epoch) + "_" + str(
        opt.nEpochs) + "_" + str(
        opt.sigma) + ".pth"
    state = {"epoch": epoch, "Tnet": Tnet, "Fnet": Fnet}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    main()
