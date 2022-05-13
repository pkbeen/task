import torch

class RobustModel(torch.nn.Module):
    def __init__(self):
        super(RobustModel, self).__init__()

#         self.in_dim = 28 * 28 * 3
#         self.out_dim = 10
        
        self.conv_layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d((2, 2, 2))
        )
        
        self.conv_layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d((2, 2, 2))
        )
        
        self.conv_layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d((2, 2, 2), padding=1)
        )
                
        self.fc4 = torch.nn.Linear(4*4*128, 512, bias = True)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc4,
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(p=0.15)
        )
        
        self.fc5 = torch.nn.Linear(512, 128, bias = True)
        torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc4,
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.15)
        )
        
        self.fc6 = torch.nn.Linear(128, 10, bias = True)
        torch.nn.init.xavier_uniform_(self.fc6.weight)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc6(out)
        return out