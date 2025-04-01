import torch
import torch.nn as nn
import monai

# Extract encoder part from pre-trained Swin UNETR and build a classifier
# Model and seed detail: https://monai.io/model-zoo -> Swin unetr btcv segmentation
# Original task: volumetric (3D) multi-organ segmentation using CT images
class SwinUNETRClassifier(nn.Module):
    def __init__(self, weight_path, num_classes=3):
        super(SwinUNETRClassifier, self).__init__()
        
        SwinUNETR_model = monai.networks.nets.SwinUNETR(
                    img_size=(128, 128, 128), 
                    in_channels=1, 
                    out_channels=14, 
                    feature_size=48, 
                    drop_rate=0.0, 
                    attn_drop_rate=0.0, 
                    dropout_path_rate=0.0, 
                    use_checkpoint=False, 
                    spatial_dims=3, 
                    norm_name="instance"
                )
        SwinUNETR_model.load_state_dict(torch.load(weight_path))

        # Encoder components from the pretrained SwinUNETR model
        self.swinViT = SwinUNETR_model.swinViT
        self.encoder1 = SwinUNETR_model.encoder1
        self.encoder2 = SwinUNETR_model.encoder2
        self.encoder3 = SwinUNETR_model.encoder3
        self.encoder4 = SwinUNETR_model.encoder4
        self.encoder10 = SwinUNETR_model.encoder10

        # Freeze the pretrained layers
        for param in self.swinViT.parameters():
            param.requires_grad = False
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
        for param in self.encoder10.parameters():
            param.requires_grad = False
            
        self.reduce_dim1 = nn.Sequential(nn.Conv3d(48, 2, 1),
                                         nn.AdaptiveAvgPool3d((4, 4, 4)))
        self.reduce_dim2 = nn.Sequential(nn.Conv3d(48, 2, 1),
                                         nn.AdaptiveAvgPool3d((4, 4, 4)))
        self.reduce_dim3 = nn.Sequential(nn.Conv3d(96, 4, 1),
                                         nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.reduce_dim4 = nn.Sequential(nn.Conv3d(192, 8, 1),
                                         nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.reduce_dim5 = nn.Sequential(nn.Conv3d(768, 32, 1),
                                         nn.AdaptiveAvgPool3d((1, 1, 1)))

        # Linear classifier
        self.classifier = nn.Sequential(
                        nn.Linear(384, 16), 
                        nn.ReLU(),               
                        nn.Dropout(0.5),          
                        nn.Linear(16, num_classes)  
                        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass input through the Swin Transformer and obtain hidden states
        hidden_states = self.swinViT(x, normalize=False)

        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(hidden_states[0])
        x3 = self.encoder3(hidden_states[1])
        x4 = self.encoder4(hidden_states[2])
        x5 = self.encoder10(hidden_states[4])


        # Dimension reduction
        x1 = self.reduce_dim1(x1)
        x2 = self.reduce_dim2(x2)
        x3 = self.reduce_dim3(x3)
        x4 = self.reduce_dim4(x4)
        x5 = self.reduce_dim5(x5)
        
        # Flatten and concatenate reduced outputs
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x3 = torch.flatten(x3, 1)
        x4 = torch.flatten(x4, 1)
        x5 = torch.flatten(x5, 1)
        
        concatenated_x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # Classifier to output the final class predictions
        x = self.classifier(concatenated_x)
        x = self.softmax(x)

        return x