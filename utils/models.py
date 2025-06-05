import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        d4 = self.upconv4(b); d4 = torch.cat((e4, d4), dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat((e3, d3), dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat((e2, d2), dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat((e1, d1), dim=1); d1 = self.dec1(d1)
        return self.out_conv(d1)

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=15, num_features=80):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual

class REDNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=5, num_features=64):
        super(REDNet, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)

        self.conv_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))

        for _ in range(num_layers - 1):
            self.deconv_layers.append(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
        self.deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        encoder_outputs = []
        h = x

        for i in range(self.num_layers):
            h = self.conv_layers[i](h)
            h = self.relu(h)
            if i < self.num_layers - 1:
                encoder_outputs.append(h)

        for i in range(self.num_layers):
            h = self.deconv_layers[i](h)
            if i < self.num_layers - 1:
                skip_val = encoder_outputs[self.num_layers - 2 - i]
                h = h + skip_val
                h = self.relu(h)
        return h
