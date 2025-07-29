import torch
import torch.nn as nn

class KinematicModel(nn.Module):
    def __init__(self, n_gestures, input_channels, feat_channel):
        super(KinematicModel, self).__init__()
        self.feat_channel = feat_channel
        # self.cnn_lstm = nn.ModuleList([CNN_LSTM(input_channels, feat_channel) for _ in range(n_gestures)])  
        self.cnn_lstm = CNN_LSTM(input_channels, feat_channel)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.norm1 = nn.BatchNorm1d(feat_channel) 
        self.relu1 = nn.ReLU() 
        
        self.dropout = nn.Dropout(p=0.5)       
        self.fc = nn.Linear(feat_channel, feat_channel)
        
    def forward(self, x):
        # input size: [batch, gestures, channels, length]
        # Encode each gesture's data separately
        batch_size = x.shape[0]
        n_gestures = x.shape[1]
        x = x.view(-1, x.shape[-2], x.shape[-1]) # [batch x gestures, channels, length]
        x = self.cnn_lstm(x) # [batch x gestures, feat_channel]
        
        x = x.view(batch_size, n_gestures, -1) # [batch, gestures, feat_channel]
        x = x.permute(0, 2, 1) # [batch, feat_channel, gestures]
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # [batch, feat_channel]
        
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN_LSTM, self).__init__()
        self.res1 = ResidualBlock(input_channels)
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=21, stride=2, padding=10)
                
        self.res2 = ResidualBlock(input_channels)
        self.conv2 = nn.Conv1d(input_channels, input_channels*2, kernel_size=21, stride=2, padding=10)
                
        self.lstm1 = nn.LSTM(input_channels*2, input_channels, num_layers=2, batch_first=True, bidirectional=True)

        self.norm1 = nn.BatchNorm1d(input_channels)
        self.relu1 = nn.ReLU()
        
        self.norm2 = nn.BatchNorm1d(input_channels*2)
        self.relu2 = nn.ReLU()
        
        self.norm3 = nn.BatchNorm1d(input_channels*2)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(16) # Pool on the temporal dimension

        self.fc = nn.Linear(input_channels*2*16, output_channels)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.res2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = x.permute(0, 2, 1) # [N, C, L] -> [N, L, C]
        x, (hn, cn) = self.lstm1(x)
        x = x.permute(0, 2, 1) # [N, L, C] -> [N, C, L]
        x = self.norm3(x)
        x = self.relu3(x)
        
        x = self.pool(x) # [N, C, pooling_size]
        x = x.view(x.shape[0], -1)  # [N, C x pooling_size]
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=21, stride=1, padding=10)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=21, stride=1, padding=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    
    
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or \
        isinstance(m, nn.Conv2d) or \
        isinstance(m, nn.Conv3d):
        #nn.init.kaiming_uniform_(m.weight.data)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d) or \
        isinstance(m, nn.BatchNorm1d):
        #nn.init.kaiming_uniform_(m.weight.data)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        #nn.init.kaiming_uniform_(m.weight.data)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
