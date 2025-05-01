import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import ssl
import os

# Fix SSL verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard UNet architecture for segmentation
    """
    def __init__(self, num_classes=1, bilinear=False, n_channels=3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Initial double convolution
        self.inc = DoubleConv(n_channels, 64)
        
        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Upsampling path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final convolution
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # Store input shape
        input_shape = x.shape[-2:]
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)
        
        # Ensure output matches input resolution
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
            
        return logits


class ResNetUNet(nn.Module):
    """
    UNet with ResNet101 encoder backbone
    """
    def __init__(self, num_classes=1, pretrained=True, weights_path=None):
        super(ResNetUNet, self).__init__()
        self.num_classes = num_classes
        
        # Load ResNet encoder backbone
        try:
            if pretrained and weights_path and os.path.exists(weights_path):
                print(f"Loading ResNet101 weights from local file: {weights_path}")
                resnet_model = models.resnet101(weights=None)
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                resnet_model.load_state_dict(state_dict)
            elif pretrained:
                print("Loading ResNet101 weights from torchvision")
                resnet_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                print("Using random initialization (pretrained=False)")
                resnet_model = models.resnet101(weights=None)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Falling back to random initialization")
            resnet_model = models.resnet101(weights=None)
            
        # Define encoder layers from ResNet
        self.encoder1 = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu
        )  # 64 channels
        self.pool = resnet_model.maxpool
        self.encoder2 = resnet_model.layer1  # 256 channels
        self.encoder3 = resnet_model.layer2  # 512 channels
        self.encoder4 = resnet_model.layer3  # 1024 channels
        self.encoder5 = resnet_model.layer4  # 2048 channels
        
        # Define decoder path
        self.decoder1 = Up(2048 + 1024, 512, bilinear=False)
        self.decoder2 = Up(512 + 512, 256, bilinear=False)
        self.decoder3 = Up(256 + 256, 128, bilinear=False)
        self.decoder4 = Up(128 + 64, 64, bilinear=False)
        
        # Center convolution (optional bottleneck processing)
        self.center = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        # Final output convolution
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Store input resolution for later upsampling
        input_shape = x.shape[-2:]
        
        # Encoder path
        e1 = self.encoder1(x)  # [B, 64, H/2, W/2]
        p1 = self.pool(e1)     # [B, 64, H/4, W/4]
        
        e2 = self.encoder2(p1)  # [B, 256, H/4, W/4]
        e3 = self.encoder3(e2)  # [B, 512, H/8, W/8]
        e4 = self.encoder4(e3)  # [B, 1024, H/16, W/16]
        e5 = self.encoder5(e4)  # [B, 2048, H/32, W/32]
        
        # Center
        e5 = self.center(e5)  # [B, 2048, H/32, W/32]
        
        # Decoder path with skip connections
        d1 = self.decoder1(e5, e4)  # [B, 512, H/16, W/16]
        d2 = self.decoder2(d1, e3)  # [B, 256, H/8, W/8]
        d3 = self.decoder3(d2, e2)  # [B, 128, H/4, W/4]
        d4 = self.decoder4(d3, e1)  # [B, 64, H/2, W/2]
        
        # Final output
        logits = self.final(d4)  # [B, num_classes, H/2, W/2]
        
        # Ensure output matches input resolution
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
            
        return logits


# Specialized UNet for kidney ultrasound segmentation
class KidneyUNet(nn.Module):
    """
    Enhanced UNet specifically designed for kidney ultrasound segmentation
    with deep supervision and attention mechanism
    """
    def __init__(self, num_classes=1, input_channels=3, features=32):
        super(KidneyUNet, self).__init__()
        self.num_classes = num_classes
        
        # Encoder - Downsampling path
        self.enc1 = DoubleConv(input_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(features, features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(features*2, features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(features*4, features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge
        self.bottleneck = DoubleConv(features*8, features*16)
        
        # Decoder - Upsampling path with attention gates
        self.up4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.att4 = AttentionGate(features*8, features*8, features*8)
        self.dec4 = DoubleConv(features*16, features*8)
        
        self.up3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.att3 = AttentionGate(features*4, features*4, features*4)
        self.dec3 = DoubleConv(features*8, features*4)
        
        self.up2 = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(features*2, features*2, features*2)
        self.dec2 = DoubleConv(features*4, features*2)
        
        self.up1 = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.att1 = AttentionGate(features, features, features)
        self.dec1 = DoubleConv(features*2, features)
        
        # Output layer
        self.final = nn.Conv2d(features, num_classes, kernel_size=1)
        
        # Deep supervision outputs
        self.ds3 = nn.Sequential(
            nn.Conv2d(features*4, features*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, num_classes, kernel_size=1)
        )
        
        self.ds2 = nn.Sequential(
            nn.Conv2d(features*2, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Store input shape for final resizing
        input_shape = x.shape[-2:]
        
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path with attention
        d4 = self.up4(b)
        a4 = self.att4(e4, d4)  # Apply attention gate
        d4 = torch.cat([a4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        a3 = self.att3(e3, d3)  # Apply attention gate
        d3 = torch.cat([a3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        a2 = self.att2(e2, d2)  # Apply attention gate
        d2 = torch.cat([a2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)  # Apply attention gate
        d1 = torch.cat([a1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Main output
        out = self.final(d1)
        
        # Deep supervision outputs
        ds3 = self.ds3(d3)
        ds3 = F.interpolate(ds3, size=input_shape, mode='bilinear', align_corners=False)
        
        ds2 = self.ds2(d2)
        ds2 = F.interpolate(ds2, size=input_shape, mode='bilinear', align_corners=False)
        
        # Ensure main output matches input resolution
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        
        # Return main output if training=False or all outputs for deep supervision
        if self.training:
            return out, ds3, ds2
        else:
            return out


class AttentionGate(nn.Module):
    """
    Attention Gate for UNet to focus on relevant features
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Inputs: g (gating signal from coarser layer), x (skip connection features)
        
        # Handling if g and x have different spatial dimensions
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi  # Element-wise multiplication


# Specialized loss function for deep supervision
class DeepSupervisionLoss(nn.Module):
    """
    Loss function for models with deep supervision outputs
    """
    def __init__(self, main_loss_fn, weights=(0.5, 0.3, 0.2)):
        super(DeepSupervisionLoss, self).__init__()
        self.main_loss_fn = main_loss_fn  # Base loss function (e.g., BinaryHybridLoss)
        self.weights = weights  # Weights for main output and deep supervision branches
        
    def forward(self, outputs, target):
        # Unpack outputs from model with deep supervision
        if isinstance(outputs, tuple):
            main_output = outputs[0]
            ds_outputs = outputs[1:]
            
            # Calculate loss for main output
            main_loss = self.main_loss_fn(main_output, target)
            
            # Calculate losses for deep supervision outputs
            ds_losses = [self.main_loss_fn(ds_out, target) for ds_out in ds_outputs]
            
            # Weight losses
            total_loss = self.weights[0] * main_loss
            for i, ds_loss in enumerate(ds_losses):
                total_loss += self.weights[i+1] * ds_loss
                
            return total_loss
        else:
            # If no deep supervision, just return main loss
            return self.main_loss_fn(outputs, target)