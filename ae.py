import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # initializers
    def __init__(self, params, z_dim, nfilter=128):
        super(Encoder, self).__init__()

        self.conv1_1 = nn.Conv2d(3, nfilter//2, 4, 2, 1, bias=False)
        self.norm1 = nn.GroupNorm(nfilter//2, nfilter//2, affine=True)

        self.conv1_2 = nn.Conv2d(10, nfilter//2, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(nfilter, nfilter*2, 4, 2, 1, bias=False)
        self.norm2 = nn.GroupNorm(nfilter*2, nfilter*2, affine=True)

        self.conv3 = nn.Conv2d(nfilter*2, nfilter*4, 4, 2, 1, bias=False)
        self.norm3 = nn.GroupNorm(nfilter*4, nfilter*4, affine=True)

        self.conv4 = nn.Conv2d(nfilter * 4, z_dim, 4, 1, 0, bias=False)

        self.fill = torch.zeros([10, 10, 32, 32]).to(params.device)
        for i in range(10):
            self.fill[i, i, :, :] = 1

    def forward(self, input, label):
        label = self.fill[label]

        x = self.norm1(self.conv1_1(input))
        x = F.leaky_relu(x, 0.2)

        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)

        x = self.norm2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)

        x = self.norm3(self.conv3(x))
        x = F.leaky_relu(x, 0.2)

        return self.conv4(x)

class Decoder(nn.Module):
    # initializers
    def __init__(self, z_dim, nfilter=128):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(z_dim, nfilter*4, 4, 1, 0, bias=False)
        self.norm1 = nn.GroupNorm(nfilter*4, nfilter*4, affine=True)

        self.conv2 = nn.ConvTranspose2d(nfilter*4, nfilter*2, 4, 2, 1, bias=False)
        self.norm2 = nn.GroupNorm(nfilter*2, nfilter*2, affine=True)

        self.conv3 = nn.ConvTranspose2d(nfilter*2, nfilter, 4, 2, 1, bias=False)
        self.norm3 = nn.GroupNorm(nfilter, nfilter, affine=True)

        self.conv4 = nn.ConvTranspose2d(nfilter, 3, 4, 2, 1, bias=False)

    # forward method
    def forward(self, input):
        x = self.norm1(self.conv1(input))
        x = F.leaky_relu(x, 0.2)

        x = self.norm2(self.conv2(x))
        x = F.leaky_relu(x, 0.2)

        x = self.norm3(self.conv3(x))
        x = F.leaky_relu(x, 0.2)
        
        return torch.tanh(self.conv4(x))

class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        self.encoder = Encoder(params, z_dim=params.z_dim, nfilter=params.filter)
        self.decoder = Decoder(z_dim=params.z_dim, nfilter=params.filter)

    def forward(self, x, y):
        z = self.encoder(x, y)
        recon_img = self.decoder(z)
        return recon_img, z
