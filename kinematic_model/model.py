import torch
import torch.nn as nn

class KinematicModel(nn.Module):
    def __init__(self, n_gestures, input_channels, feat_channel, num_classes):
        super(KinematicModel, self).__init__()
        # self.cnn_lstm = nn.ModuleList([CNN_LSTM(input_channels, feat_channel) for _ in range(n_gestures)])  
        self.cnn_lstm = CNN_LSTM(input_channels, feat_channel)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.norm1 = nn.BatchNorm1d(feat_channel) 
        self.relu1 = nn.ReLU() 

        self.dropout = nn.Dropout(p=0.5)       
        self.fc = nn.Linear(feat_channel, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # input size: [batch, gestures, channels, length]
        # Encode each gesture's data separately
        x = x.permute(1, 0, 2, 3) # [gestures, batch, channels, length]
        enc_features = []
        for i in range(x.size(0)):
            enc_x = self.cnn_lstm(x[i])
            enc_features.append(enc_x)
        x = torch.stack(enc_features)
        x = x.permute(1, 2, 0) # [batch, gestures, feat_channel]         
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
# Archive
# class KinematicModel(nn.Module):
#     def __init__(self, n_gestures, input_channels, num_classes):
#         super(KinematicModel, self).__init__()
#         self.cnn = nn.Sequential(
#                         nn.Conv1d(input_channels, 4, kernel_size=3, stride=1, padding='same'),
#                         nn.Conv1d(4, 2, kernel_size=21, stride=1, padding='same'),
#                         nn.BatchNorm1d(2),
#                         nn.ReLU()
#                         )
#         self.lstm = nn.LSTM(2, 2, num_layers=1, batch_first=True, bidirectional=False)
#         # self.lstm = nn.ModuleList([nn.LSTM(2, 2, num_layers=1, batch_first=True, bidirectional=False) for _ in range(n_gestures)])
        
#         self.norm1 = nn.BatchNorm1d(10) 
#         self.relu1 = nn.ReLU() 

#         self.dropout = nn.Dropout(p=0.5)       
#         self.fc = nn.Linear(10, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # input size: [batch, gestures, channels, length]
#         # Encode each gesture's data separately
#         x = x.permute(1, 0, 2, 3) # [gestures, batch, channels, length]
#         enc_features = []
#         for i in range(x.size(0)):
#             enc_x = x[i]
#             enc_x = self.cnn(enc_x)
#             enc_x = enc_x.permute(0, 2, 1) # [batch, length, channels]
#             _, (hn, cn) = self.lstm(enc_x)
#             enc_x = hn.permute(1, 0, 2) # [batch, num_layers, hidden_size]
#             enc_x = enc_x.reshape(enc_x.size(0), -1) # [batch, num_layers*hidden_size]
#             enc_features.append(enc_x)
#         x = torch.cat(enc_features, dim=1) 
#         x = self.relu1(x)
#         x = self.norm1(x)
        
#         x = self.dropout(x)
#         x = self.fc(x)
#         x = self.softmax(x)
#         return x
    
# class KinematicModel(nn.Module):
#     def __init__(self, n_gestures, input_channels, num_classes):
#         super(KinematicModel, self).__init__()
#         self.cnn = nn.Sequential(
#                         nn.Conv1d(input_channels, 16, kernel_size=41, stride=1, padding='same'),
#                         nn.Conv1d(16, 32, kernel_size=21, stride=1, padding='same'),
#                         nn.Conv1d(32, 64, kernel_size=11, stride=1, padding='same'),
#                         nn.BatchNorm1d(64),
#                         nn.ReLU()
#                         )
#         self.lstm = nn.ModuleList([nn.LSTM(64, 32, num_layers=2, batch_first=True, bidirectional=False) for _ in range(n_gestures)])
        
#         self.norm1 = nn.BatchNorm1d(320) 
#         self.relu1 = nn.ReLU() 

#         self.dropout = nn.Dropout(p=0.5)       
#         self.fc1 = nn.Linear(320, 64)
#         self.norm2 = nn.BatchNorm1d(64) 
#         self.relu2 = nn.ReLU() 

#         self.fc2 = nn.Linear(64, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         # input size: [batch, gestures, channels, length]
#         # Encode each gesture's data separately
#         x = x.permute(1, 0, 2, 3) # [gestures, batch, channels, length]
#         enc_features = []
#         for i in range(x.size(0)):
#             enc_x = x[i]
#             enc_x = self.cnn(enc_x)
#             enc_x = enc_x.permute(0, 2, 1) # [batch, length, channels]
#             _, (hn, cn) = self.lstm[i](enc_x)
#             enc_x = hn.permute(1, 0, 2) # [batch, num_layers, hidden_size]
#             enc_x = enc_x.reshape(enc_x.size(0), -1) # [batch, num_layers*hidden_size]
#             enc_features.append(enc_x)
#         x = torch.cat(enc_features, dim=1) 
#         x = self.relu1(x)
#         x = self.norm1(x)
        
#         x = self.dropout(x)
#         x = self.fc1(x)
#         x = self.relu2(x)
#         x = self.norm2(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
    
class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN_LSTM, self).__init__()
        self.res1 = ResidualBlock(input_channels)
        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size=21, stride=2, padding=10)
        
        self.res2 = ResidualBlock(8)
        self.conv2 = nn.Conv1d(8, 8, kernel_size=21, stride=2, padding=10)
        
        self.res3 = ResidualBlock(8)
        self.conv3 = nn.Conv1d(8, 8, kernel_size=21, stride=2, padding=10)

        self.norm1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU()
        
        self.lstm1 = nn.LSTM(8, 8, num_layers=2, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=Tr)

        self.norm2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveMaxPool1d(1) # Pool on the temporal dimension

        self.dropout = nn.Dropout(p=0.5)        
        self.fc = nn.Linear(32, output_channels)
        self.relu3 = nn.ReLU()
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = x.permute(0, 2, 1) # [N, C, L] -> [N, L, C]
        x, (hn, cn) = self.lstm1(x)
        # x = x.permute(0, 2, 1) #  [N, L, C] -> [N, C, L]
        # x = self.norm2(x)
        # x = self.relu2(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        x = hn.permute(1, 0, 2) #  [num_layers, N, 2*hidden_size] -> [N, num_layers, 2*hidden_size]
        x = x.reshape(x.size(0), -1)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=21, stride=1, padding=10)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=21, stride=1, padding=10)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

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