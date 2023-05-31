import torch
from torch import nn
import itertools as it
import time
from data import StyleTransferDataset
from utils import LambdaLR, weights_init_normal, ReplayBuffer
from torch.utils.data import DataLoader
from Generator import Generator
from Discriminator import Discriminator
from config import Config
from tqdm import tqdm
import os
import json
import warnings
from utils import set_weights_init
import math as mt
warnings.filterwarnings('ignore')

configuration = Config()
def main(config):
    train_on_gpu = config.train_on_gpu
    if train_on_gpu:
        print("CUDA is available, Training on GPU")
    else:
        print("CUDA isn't available, Training on CPU")
    print("本次训练的图像风格为", configuration.style)
    device = config.device

    batch_size = 1
    learning_rate = 0.0002
    n_epochs = config.n_epochs
    epoch = 0
    decay_epoch = config.decay_epoch

    train_dataset = StyleTransferDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G1 = Generator().to(device)
    G2 = Generator().to(device)
    D1 = Discriminator(config.attention).to(device)
    D2 = Discriminator(config.attention).to(device)

    if config.weights:
        print("正在加载预训练模型-", config.attention)
        if not config.attention:
            model_name = os.path.join(config.model, config.style + '_model.pth')
        else:
            model_name = os.path.join(config.model, config.style + '_attention_model.pth')
        checkpoint = torch.load(model_name)
        G1.load_state_dict(checkpoint['G1'])
        G2.load_state_dict(checkpoint['G2'])
        D1.load_state_dict(checkpoint['D1'])
        D2.load_state_dict(checkpoint['D2'])
        start_epoch = checkpoint['start_epoch']
        print("预训练模型加载完毕")
    else:
        print("正在进行模型初始化-", config.attention)
        G1.apply(weights_init_normal)
        G2.apply(weights_init_normal)
        D1.apply(weights_init_normal)
        D2.apply(weights_init_normal)
        start_epoch = 0
        print("模型初始化完毕")

    loss_MSE = nn.MSELoss()
    loss_MAE = nn.L1Loss()

    fake_content_buffer = ReplayBuffer(50)
    fake_style_buffer = ReplayBuffer(50)

    opt_G = torch.optim.Adam(it.chain(G1.parameters(), G2.parameters()), lr=learning_rate, betas=(0.5, 0.9999))
    opt_D = torch.optim.Adam(it.chain(D1.parameters(), D2.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    schedule_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=LambdaLR(n_epochs, epoch,
                                                                             decay_epoch).step)

    schedule_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=LambdaLR(n_epochs, epoch,
                                                                             decay_epoch).step)

    info = {'G_Loss': [], "D_Loss": [], "Total_Loss": []}

    for epoch in range(start_epoch, n_epochs):
        epoch_time = time.time()
        loop = tqdm(enumerate(train_loader), total=mt.ceil(len(train_dataset) / train_loader.batch_size), position=0)
        G1.train()
        D1.train()
        G2.train()
        D2.train()
        for i, batch in loop:
            loop.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            contentImage = batch['Content'].to(device)
            styleImage = batch['Style'].to(device)
            if contentImage.shape[1] != 3 or styleImage.shape[1] != 3:
                continue
            G1_style = G1(contentImage)
            recover_content = G2(G1_style)
            G2_content = G2(styleImage)
            recover_style = G1(G2_content)

            set_weights_init([D1, D2], requires_grad=False)
            opt_G.zero_grad()

            D2_out_fake = D2(G1_style)
            G1_gan_loss = loss_MSE(D2_out_fake, torch.ones(D2_out_fake.size()).to(device))

            D1_out_fake = D1(G2_content)
            G2_gan_loss = loss_MSE(D1_out_fake, torch.ones(D1_out_fake.size()).to(device))

            Gan_Loss = G1_gan_loss + G2_gan_loss
            Cycle_Loss = loss_MAE(contentImage, recover_content) * 10 + loss_MAE(styleImage, recover_style) * 10

            G1_identity = G1(styleImage)
            G2_identity = G2(contentImage)
            Identity_Loss = loss_MAE(styleImage, G1_identity) * 10 * 0.5 + loss_MAE(contentImage, G2_identity) * 10 * 0.5

            loss_G = Gan_Loss + Cycle_Loss + Identity_Loss
            loss_G.backward()
            opt_G.step()

            set_weights_init([D1, D2], requires_grad=True)
            opt_D.zero_grad()
            content_fake_p = fake_content_buffer.query(G2_content)
            style_fake_p = fake_style_buffer.query(G1_style)

            D1_out_fake = D1(content_fake_p.detach()).squeeze()
            D1_out_real = D1(contentImage).squeeze()
            loss_D11 = loss_MAE(D1_out_fake, torch.zeros(D1_out_fake.size()).to(device))
            loss_D12 = loss_MAE(D1_out_real, torch.ones(D1_out_real.size()).to(device))
            loss_D1 = (loss_D11 + loss_D12) * 0.5

            D2_out_fake = D2(style_fake_p.detach()).squeeze()
            D2_out_real = D2(styleImage).squeeze()
            loss_D21 = loss_MAE(D2_out_fake, torch.zeros(D2_out_fake.size()).to(device))
            loss_D22 = loss_MAE(D2_out_real, torch.ones(D2_out_real.size()).to(device))
            loss_D2 = (loss_D21 + loss_D22) * 0.5

            loss_D = loss_D1 + loss_D2

            loss_D.backward()
            opt_D.step()

            total_loss = loss_D + loss_G
            if i % 20 == 0:
                info['G_Loss'].append(round(loss_G.item(), 4))
                info['D_Loss'].append(round(loss_D.item(), 4))
                info['Total_Loss'].append(round(total_loss.item(), 3))
            loop.set_postfix(loss_G=round(loss_G.item(), 3), loss_D=round(loss_D.item(), 3))

            if i == mt.ceil(len(train_dataset) / train_loader.batch_size) - 1:
                end = time.time()
                spend = end - epoch_time
                h = int((spend // 60) // 60)
                m = int((spend // 60) % 60)
                s = round(spend % 60, 3)
                loop.set_postfix(loss_G=round(loss_G.item(), 3), loss_D=round(loss_D.item(), 3),
                                 epoch_elapsed=f"{h}h: {m}m: {s}s")

        schedule_G.step()
        schedule_D.step()

        checkpoint = {'G1': G1.state_dict(), "G2": G2.state_dict(), "D1": D1.state_dict(),
                      "D2": D2.state_dict(), 'start_epoch': epoch}

        if not os.path.exists(config.model):
            os.mkdir(config.model)
        if not config.attention:
            torch.save(checkpoint, os.path.join(config.model, config.style + '_channel_model.pth'))
        else:
            torch.save(checkpoint, os.path.join(config.model, config.style + '_attention_spacial_model.pth'))

        json_file = json.dumps(info)
        if not config.attention:
            json_name = config.style + '_channel_info.json'
        else:
            json_name = config.style + '_attention_spacial_info.json'

        with open(json_name, 'w') as f:
            f.write(json_file)

if __name__ == '__main__':
    main(configuration)








