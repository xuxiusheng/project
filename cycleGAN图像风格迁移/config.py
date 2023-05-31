import torch
import os

class Config():
    def __init__(self, root='./train'):
        self.root = root
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        self.train_on_gpu = torch.cuda.is_available()
        self.style = 'ink'
        self.size = 256
        self.n_epochs = 80
        self.decay_epoch = 50
        self.model = './model'
        self.attention = True
        self.weights = self.exist()

    def exist(self):
        catalogue = os.listdir(self.model)

        if not self.attention:
            if self.style + '_model.pth' in catalogue:
                return True
            return False
        else:
            if self.style + '_attention_spacial_model.pth' in catalogue:
                return True
            return False


if __name__ == '__main__':
    configuration = Config()
    print("数据根目录: ", configuration.root)
    print("训练设备: ", configuration.device)
    print("训练风格: ", configuration.style)
    print("图片切割尺寸: ", configuration.size)
    print("训练轮次: ", configuration.n_epochs)
    print("衰减轮次: ", configuration.decay_epoch)
    print("模型保存路径: ", configuration.model)
    print("是否使用注意力机制: ", configuration.attention)
    print("是否存在预训练模型参数: ", configuration.weights)
