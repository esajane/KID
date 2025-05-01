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

class ResNetSegNet(nn.Module):
    """
    Fixed implementation of ResNetSegNet with proper channel handling
    """
    def __init__(self, num_classes=1, pretrained=True, weights_path=None):
        super(ResNetSegNet, self).__init__()
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
            
        # Encoder layers
        self.layer0 = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool
        )
        self.layer1 = resnet_model.layer1  # 256 channels
        self.layer2 = resnet_model.layer2  # 512 channels
        self.layer3 = resnet_model.layer3  # 1024 channels
        self.layer4 = resnet_model.layer4  # 2048 channels
        
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),  
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self._init_decoder_weights()
        
    def _init_decoder_weights(self):
        """Initialize weights for the decoder part"""
        for m in [self.up1, self.decoder1, self.up2, self.decoder2, 
                 self.up3, self.decoder3, self.up4, self.decoder4, self.final_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Encoder
        e0 = self.layer0(x)      # [B, 64, H/4, W/4]
        e1 = self.layer1(e0)     # [B, 256, H/4, W/4]
        e2 = self.layer2(e1)     # [B, 512, H/8, W/8]
        e3 = self.layer3(e2)     # [B, 1024, H/16, W/16]
        e4 = self.layer4(e3)     # [B, 2048, H/32, W/32]
        
        # Decoder with skip connections 
        d1 = self.up1(e4)                            # [B, 1024, H/16, W/16]
        d1 = torch.cat([d1, e3], dim=1)              # [B, 2048, H/16, W/16]
        d1 = self.decoder1(d1)                       # [B, 1024, H/16, W/16]
        
        d2 = self.up2(d1)                            # [B, 512, H/8, W/8]
        d2 = torch.cat([d2, e2], dim=1)              # [B, 1024, H/8, W/8]
        d2 = self.decoder2(d2)                       # [B, 512, H/8, W/8]
 
        d3 = self.up3(d2)                            # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e1], dim=1)              # [B, 512, H/4, W/4]
        d3 = self.decoder3(d3)                       # [B, 256, H/4, W/4]
        
        d4 = self.up4(d3)                            # [B, 64, H/2, W/2]
        
        # Handle size mismatch with bilinear interpolation
        if d4.shape[2:] != e0.shape[2:]:
            d4 = F.interpolate(d4, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e0], dim=1)              # [B, 128, H/2, W/2]
        d4 = self.decoder4(d4)                       # [B, 64, H/2, W/2]
        
        x = self.final_conv(d4)                      
        
        if x.shape[-2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x