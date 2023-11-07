import torch
import torch.nn as nn

# Block of CNNs
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        
        # Simple CNN block 
            # 2DCNN [stride : to  down sample ],
            # Batch Norm : In recent articl they use instance norm 
            # Leaky Relu , or Relu simply 
             
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    # Apply the convolution on the input simply
    def forward(self, x):
        return self.conv(x)


# Descriminator 
class Discriminator(nn.Module):

    
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        
        # Features by default filed with the values to have a 70*70 Batch CNN discriminator 
        # Other possibilities : 1*1 : [64-128]

        super().__init__()

        # They specefied in the paper that there is no Batch Norm in the first Conv Block 
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, # We multiplie * 2 (it get the input X and the output Y  image as input)
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        # Blocks constructions 

        layers = []
        in_channels = features[0]

        for i in range(1,len(features),1):
            if i == len(features)-1 : # In the paper they don't use stride =2 but 1 instead in the last block 
                layers.append(
                    CNNBlock(in_channels, features[i], stride=1),
                )
            else :
                layers.append(
                    CNNBlock(in_channels, features[i], stride=2),
                )

            in_channels = features[i]

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )
        
        self.model = nn.Sequential(*layers)


    def forward(self, x, y):
        # Concatenate x and y
        x = torch.cat([x, y], dim=1)
        # Initial conv block 
        x = self.initial(x)
        # Forward the model
        x = self.model(x)
        return x


def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()