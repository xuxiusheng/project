from Generator import Generator
import torch
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as imgplt
from config import Config
from utils import trans


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

configuration = Config()
device = configuration.device

def test(img_path, path):
    img = Image.open(img_path)

    transforms = transform.Compose([
        transform.ToTensor(),
        transform.Resize((256, 256)),
        transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    img = transforms(img.copy())
    img = img[None].to(device)
    G1 = Generator().to(device)
    G2 = Generator().to(device)
    if "ink_attention_model.pth" in path or "ink_model.pth" in path:
        path = trans(path)
    checkpoint = torch.load(path)
    G1.load_state_dict(checkpoint['G1'], strict=False)
    G2.load_state_dict(checkpoint['G2'], strict=False)

    G1.eval()
    out = G1(img)
    out_trans = G2(out)
    out = out[0].permute(1, 2, 0)
    out_trans = out_trans[0].permute(1, 2, 0)
    out = (0.5 * (out + 0.8)).cpu().detach().numpy()
    out_trans = (0.5 * (out_trans + 0.5)).cpu().detach().numpy()
    return out, out_trans

def GatysAnalysis():
    images = [r'E:\code\PYTHON\DL\毕业设计\train\content\2013-11-23 02_19_20.jpg', r'E:\code\PYTHON\DL\毕业设计\train\content\2013-12-16 06_38_39.jpg',
              r'E:\code\PYTHON\DL\毕业设计\train\content\2014-01-30 14_02_47.jpg']
    models = [['./model/ink_model.pth', './model/ink_attention_model.pth'],
              ['./model/Ukiyoe_model.pth', './model/Ukiyoe_attention_model.pth'],
              ['./model/fauvism_model.pth', './model/fauvism_attention_model.pth']]
    names = ['non-attention', 'attention']
    gatys = ['./output/2.jpg', './output/3.jpg', './output/4.jpg']
    plt.figure(figsize=(11, 5))
    for i in range(len(images)):
        plt.subplot(3, 4, 4 * i + 1)
        img = imgplt.imread(images[i])
        plt.xticks([], [])
        plt.yticks([], [])
        if i == 0:
            plt.title('原图', fontsize=20)
        plt.imshow(img)
        for j in range(2):
            plt.subplot(3, 4, 4 * i + j + 2)
            imgG = test(images[i], models[i][j])[0]
            plt.imshow(imgG)
            plt.xticks([], [])
            plt.yticks([], [])
            if i == 0:
                plt.title(names[j], fontsize=20)
        plt.subplot(3, 4, 4 * i + 4)
        if i == 0:
            plt.title('Gatys', fontsize=20)
        img = imgplt.imread(gatys[i])
        plt.imshow(img)
        plt.xticks([], [])
        plt.yticks([], [])
    plt.show()

def Transfer_image():
    images = ['./train/content/2016-12-26 07_36_25.jpg']
    models = './model/ink_attention_model.pth'
    names = ['G1_image', 'G2_image']
    plt.figure(figsize=(15, 3))
    for i in range(1):
        plt.subplot(1, 3, 3 * i + 1)
        img = imgplt.imread(images[i])
        if i == 0:
            plt.title('原图', fontsize=15)
        plt.imshow(img)
        plt.xticks([], [])
        plt.yticks([], [])
        out = test(images[i], models)
        for j in range(2):
            plt.subplot(1, 3, 3 * i + j + 2)
            if i == 0:
                plt.title(names[j], fontsize=15)
            plt.imshow(out[j])
            plt.xticks([], [])
            plt.yticks([], [])
    plt.show()

def display():
    images = ['./train/content/2016-12-26 07_36_25.jpg', './train/content/2016-12-18 15_37_45.jpg']
    # models = ['./model/ink_model.pth', './model/fauvism_model.pth', './model/Ukiyoe_model.pth']
    model = ['./model/ink_model.pth', './model/fauvism_attention_model.pth', './model/Ukiyoe_attention_model.pth']
    names = ['Ink', 'Fauvism', 'Ukiyoe']
    plt.figure(figsize=(12, 3))
    for i in range(1):
        img = imgplt.imread(images[i])
        plt.subplot(1, 4, 4 * i + 1)
        plt.imshow(img)
        if i == 0:
            plt.title('原图', fontsize=20)
        plt.xticks([], [])
        plt.yticks([], [])
        for j in range(3):
            out = test(images[i], model[j])[0]
            plt.subplot(1, 4, 4 * i + j + 2)
            plt.imshow(out)
            if i == 0:
                plt.title(names[j], fontsize=20)
            plt.xticks([], [])
            plt.yticks([], [])
    plt.show()

def show():
    images = ['./train/content/2013-12-01 01_54_08.jpg']
    models = ['./model/Ukiyoe_attention_model.pth', './model/Ukiyoe_model.pth']
    # model = ['./model/Ukiyoe_attention_model.pth', './model/Ukiyoe_model.pth']
    names = ['non-attention', 'attention']
    plt.figure(figsize=(10, 3))
    for i in range(1):
        img = imgplt.imread(images[i])
        plt.subplot(1, 3, 3 * i + 1)
        plt.imshow(img)
        if i == 0:
            plt.title('原图', fontsize=15)
        plt.xticks([], [])
        plt.yticks([], [])
        for j in range(2):
            out = test(images[i], models[j])[0]
            plt.subplot(1, 3, 3 * i + j + 2)
            plt.imshow(out)
            if i == 0:
                plt.title(names[j], fontsize=15)
            plt.xticks([], [])
            plt.yticks([], [])
    plt.show()


def Transf():
    image = './train/content/2016-12-26 07_36_25.jpg'
    models = ['./model/Ukiyoe_model.pth',
              './model/Ukiyoe_attention_channel_model.pth',
              './model/Ukiyoe_attention_spacial_model.pth',
              './model/Ukiyoe_attention_model.pth']
    names = ['non-attention', 'CA', 'SA', 'CBAM']
    plt.figure(figsize=(13, 6))
    for i in range(4):
        plt.subplot(3, 4, i + 1)
        img = imgplt.imread(image)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(names[i], fontsize=20)
    for i in range(4):
        plt.subplot(3, 4, 4 + i + 1)
        out = test(image, models[i])[0]
        plt.imshow(out)
        plt.xticks([])
        plt.yticks([])
    for i in range(4):
        plt.subplot(3, 4, 8 + i + 1)
        out = test(image, models[i])[1]
        plt.imshow(out)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def Analysis():
    images = ["./train/content/2013-12-24 04_08_27.jpg", "./train/content/2013-12-25 15_23_46.jpg",
              "./train/content/2016-08-15 17_46_46.jpg"]
    targets = ["./train/style/ink/puam_4.jpg", r"C:\Users\xuxiu\Desktop\未标题-1.jpg",
               r"C:\Users\xuxiu\Desktop\未标题-2.jpg"]

    channel_model = ["./model/ink_attention_channel_model.pth", "./model/fauvism_attention_channel_model.pth",
                     "./model/Ukiyoe_attention_channel_model.pth"]

    spacial_model = ["./model/ink_attention_spacial_model.pth", "./model/fauvism_attention_spacial_model.pth",
                     "./model/Ukiyoe_attention_spacial_model.pth"]

    CBAM_model = ["./model/ink_attention_model.pth", "./model/fauvism_attention_model.pth",
                     "./model/Ukiyoe_attention_model.pth"]

    non_model = ["./model/ink_model.pth", "./model/fauvism_model.pth",
                     "./model/Ukiyoe_model.pth"]

    CA = [test(images[i], channel_model[i])[0] for i in range(3)]
    SA = [test(images[i], spacial_model[i])[0] for i in range(3)]
    CBAM = [test(images[i], CBAM_model[i])[0] for i in range(3)]
    non = [test(images[i], non_model[i])[0] for i in range(3)]
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(3, 7, i * 7 + 1)
        if i == 0:
            plt.title("原图", fontsize=18)
        img = imgplt.imread(images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 2)
        if i == 0:
            plt.title("风格图", fontsize=18)
        img = imgplt.imread(targets[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 6)
        if i == 0:
            plt.title("CA", fontsize=18)
        plt.imshow(CA[i])
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 7)
        if i == 0:
            plt.title("SA", fontsize=18)
        plt.imshow(SA[i])
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 4)
        if i == 0:
            plt.title("CBAM", fontsize=18)
        plt.imshow(CBAM[i])
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 5)
        if i == 0:
            plt.title("non-Att", fontsize=18)
        plt.imshow(non[i])
        plt.xticks([])
        plt.yticks([])

    for i in range(3):
        plt.subplot(3, 7, i * 7 + 3)
        if i == 0:
            plt.title("Gatys", fontsize=18)
        file = "./output/" + str(i + 1) + ".jpg"
        img = imgplt.imread(file)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    plt.show()

def singleshow():
    img = test(r'C:\Users\xuxiu\Desktop\西南视角.jpg', './model/ink_attention_model.pth')[0]
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig('2-水墨.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.show()

if __name__ == '__main__':
    # Transfer_image()
    # GatysAnalysis()
    # display()
    # show()
    # plt.savefig("./output/ink_大山.jpg", bbox_inches='tight', pad_inches=0.0, dpi=300)
    #
    # plt.show()
    # Transf()
    Analysis()
    # singleshow()
    pass







