import numpy as np
from config import Config
import torch
import random
import matplotlib.pyplot as plt


configuration = Config()
train_on_gpu = configuration.train_on_gpu

class OverstepException(Exception):
    def __init__(self, message):
        self.message = message

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_epoch):
        if decay_epoch > n_epochs:
            print("开始衰减的epoch应低于总epoch")
            raise OverstepException("请重新设置decay_epoch")
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epoch = decay_epoch
    def step(self, epoch):
        return 1 - max(0, epoch + self.offset - self.decay_epoch) / (self.n_epochs - self.decay_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class ReplayBuffer():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def set_weights_init(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
def trans(name):
    if 'attention' in name:
        x = name.split('_')
        x.pop(1)
        return '_'.join(x)
    else:
        x = name.split('_')
        x.insert(1, 'attention')
        return '_'.join(x)

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    learning_rate = 0.0002
    lr = []
    for i in range(100):
        lr.append(learning_rate * (1 - max(0, i - 70) / (100 - 70)))
    plt.figure()
    plt.grid()
    plt.title('学习率调整')
    plt.plot(range(100), lr)
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.show()

