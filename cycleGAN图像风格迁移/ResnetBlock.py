import torch.nn as nn
# from torchsummary import summary

class ResNetBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResNetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=0),
                      nn.InstanceNorm2d(in_channel),
                      nn.Identity(),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3),
                      nn.InstanceNorm2d(in_channel),
                      nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x) + x

if __name__ == '__main__':
    model = ResNetBlock(3).cuda()
    # summary(model, (3, 256, 256))