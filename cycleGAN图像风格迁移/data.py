from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
import glob
import os
from config import Config

configuration = Config()
style = configuration.style

class StyleTransferDataset(Dataset):
    def __init__(self, root=configuration.root):
        super(StyleTransferDataset, self).__init__()
        self.root = root
        self.train_content = glob.glob(os.path.join(self.root, 'content/*'))
        self.train_style = glob.glob(os.path.join(self.root, 'style', style, '*'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        content_path = random.choice(self.train_content)
        style_path = self.train_style[item % len(self.train_style)]
        imageA = Image.open(content_path)
        imageB = Image.open(style_path).convert('RGB')
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)

        return {"Content": imageA, "Style": imageB}

    def __len__(self):
        return len(self.train_style)

if __name__ == '__main__':
    dataset = StyleTransferDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    print('数据加载器: ', loader)
    print("数据样本量: ", len(loader.dataset))
    print("内容数据样本: ", dataset.train_content)
    print("风格数据样本: ", dataset.train_style)
