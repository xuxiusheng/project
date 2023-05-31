from torch import nn
# from torchsummary import summary

from ResnetBlock import ResNetBlock


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        net = [nn.ReflectionPad2d(3),
               nn.Conv2d(3, 64, 7, 1),
               nn.InstanceNorm2d(64),
               nn.ReLU(inplace=True)]
        in_channel = 64
        net += [
            nn.Conv2d(in_channel, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]

        for i in range(9):
            net += [
                ResNetBlock(256)
            ]
        net += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        net += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    pass
    G = Generator().cuda()
    # summary(G, (3, 256, 256))









