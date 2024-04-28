import torch
import torch.nn as nn
import torch.nn.functional as F

#Model architecture
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_type='avg', normalization=True):
        super(EncoderBlock, self).__init__()
        self.conv_block = Conv2dBlock(in_channels, out_channels)
        self.normalisation = normalization
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_conv = self.conv_block(x)
        x = self.pool(x_conv)
        
        if self.normalisation:
            x = self.batch_norm(x)
            
        x = self.dropout(x)
        return x, x_conv

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv_block = Conv2dBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=True):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1)
        self.conv_block = Conv2dBlock(in_channels, out_channels, kernel_size)
        self.normalisation = normalization
        
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, skip_connection):
        
        # Apply padding
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_connection], dim=1)
        if self.normalisation:
            x = self.batch_norm(x)
            
        x = self.dropout(x)
        x = self.conv_block(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(in_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256, normalization=False)
        self.encoder4 = EncoderBlock(256, 512, normalization=False)
        self.bottleneck = Bottleneck(512, 1024)
        self.decoder1 = DecoderBlock(1024, 512, normalization=False)
        self.decoder2 = DecoderBlock(512, 256, normalization=False)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1, conv1 = self.encoder1(x)
        enc2, conv2 = self.encoder2(enc1)
        enc3, conv3 = self.encoder3(enc2)
        enc4, conv4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)
        dec1 = self.decoder1(bottleneck, conv4)
        dec2 = self.decoder2(dec1, conv3)
        dec3 = self.decoder3(dec2, conv2)
        dec4 = self.decoder4(dec3, conv1)
        output = self.output(dec4)
#         output = nn.functional.sigmoid(output)
        return output
