import torch.nn as nn

class LinearBCE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBCE, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    def reset_all_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()