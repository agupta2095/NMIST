import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels, con_channels):
        super(Discriminator, self).__init__()
    #Input = batch_size X img_channels x 64 x64
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=con_channels, kernel_size=(4,4), stride=2, padding=1),
            nn.LeakyReLU(0.2), #64* 32*32
            self._block(con_channels, con_channels*2, 4,2,1), #128* 16*16
            self._block(con_channels*2, con_channels*4, 4,2,1), #256* 8*8
            self._block(con_channels*4,con_channels*8, 4,2,1), #512* 4*4
            nn.Conv2d(con_channels*8, 1, 4, 2, 0), #1x1
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, features_g, img_channels):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            #Input
            self._block(z_dim, features_g*16, 4, 1, 0),  #1024*4*4
            self._block(features_g*16, features_g*8, 4, 2, 1), #512*8*8
            self._block(features_g*8, features_g*4, 4, 2, 1), #256*16*16
            self._block(features_g*4, features_g *2, 4, 2, 1), #128*32*32
            nn.ConvTranspose2d(features_g*2, img_channels, 4, 2, 1),#3*64*64
            nn.Tanh(), #[-1, 1] => Images are normalized in this range
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)

    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, 8,in_channels)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W)
    print("SUCCESS")


if __name__ == '__main__':
    test()