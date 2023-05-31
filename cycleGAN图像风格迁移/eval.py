from Generator import Generator
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as imgplt
from config import Config
from utils import trans
import cv2


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

configuration = Config()
device = configuration.device

def resize(img):
    max_value, max_idx = torch.max(torch.tensor(img.shape), dim=0)
    ratio = 256 / max_value
    img = torch.nn.functional.interpolate(
        img[None], scale_factor=ratio, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]
    boundary = img.shape
    pad_img = img.new_full([3, 256, 256], 0)
    pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return pad_img, boundary, ratio


def test(img_path, path):
    img = Image.open(img_path)

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    img = transforms(img.copy())
    img, boundary, ratio = resize(img)
    img = img[None].to(device)
    G1 = Generator().to(device)
    if "ink_attention_model.pth" in path or "ink_model.pth" in path:
        path = trans(path)
    checkpoint = torch.load(path)
    G1.load_state_dict(checkpoint['G1'], strict=False)

    G1.eval()
    out = G1(img)[0]
    out = out[:, :boundary[1], :boundary[2]]
    out = torch.nn.functional.interpolate(
        out[None], scale_factor=1.0 / ratio, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]
    out = out.permute(1, 2, 0)
    out = (0.5 * (out + 0.8)).cpu().detach().numpy()

    return out

def show(img_path, model_path):
    img = test(img_path, model_path)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig('2-水墨.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.show()

if __name__ == "__main__":
    # test(r'C:\Users\xuxiu\Desktop\西南视角.jpg', './model/ink_attention_model.pth')
    # show(r"C:\Users\xuxiu\Desktop\西南视角.jpg", "./model/fauvism_attention_model.pth")
    image = cv2.imread(r"C:\Users\xuxiu\Desktop\shijiao.jpg")
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)))