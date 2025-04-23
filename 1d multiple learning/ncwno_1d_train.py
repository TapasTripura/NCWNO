"""
NCWNO for Multi-Physics operator learning of 1D problems (five examples)...

It requires the package "Pytorch Wavelets"
-- see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html

"""

import os
directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities import *
from ncwno_modules import *

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%     
""" Def: Expert WNO block """
class Expert_WNO(nn.Module):
    def __init__(self, level, width, expert_num, size):
        super(Expert_WNO, self).__init__()

        """
        The Expert Wavelet Integral Blocks

        Input Parameters
        ----------------
        level      : scalar, levels of wavelet decomposition 
        width      : scalar, kernel dimension in lifted space
        expert_num : scalar, number of local wavelet experts 
        size       : scalar, length of input 1D signal

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """

        self.level = level
        self.width = width
        self.expert_num = expert_num
        
        wavelet = ['db'+str(i+1) for i in range(self.expert_num)] # Wavelet family is 'Daubechies'
        self.Expert_layers0=WaveConv1d(self.width, self.width, self.level, size, wavelet[0])
        self.Expert_layers1=WaveConv1d(self.width, self.width, self.level, size, wavelet[1])
        self.Expert_layers2=WaveConv1d(self.width, self.width, self.level, size, wavelet[2])
        self.Expert_layers3=WaveConv1d(self.width, self.width, self.level, size, wavelet[3])
        self.Expert_layers4=WaveConv1d(self.width, self.width, self.level, size, wavelet[4])
        self.Expert_layers5=WaveConv1d(self.width, self.width, self.level, size, wavelet[5])
        self.Expert_layers6=WaveConv1d(self.width, self.width, self.level, size, wavelet[6])
        self.Expert_layers7=WaveConv1d(self.width, self.width, self.level, size, wavelet[7])
        self.Expert_layers8=WaveConv1d(self.width, self.width, self.level, size, wavelet[8])
        self.Expert_layers9=WaveConv1d(self.width, self.width, self.level, size, wavelet[9])

    def forward(self, x, lambda_):
        x = lambda_[..., 0:1]*self.Expert_layers0(x) + lambda_[..., 1:2]*self.Expert_layers1(x) + \
            lambda_[..., 2:3]*self.Expert_layers2(x) + lambda_[..., 3:4]*self.Expert_layers3(x) + \
            lambda_[..., 4:5]*self.Expert_layers4(x) + lambda_[..., 5:6]*self.Expert_layers5(x) + \
            lambda_[..., 6:7]*self.Expert_layers6(x) + lambda_[..., 7:8]*self.Expert_layers7(x) + \
            lambda_[..., 8:9]*self.Expert_layers8(x) + lambda_[..., 9:10]*self.Expert_layers9(x)
        return x

    
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, input_dim, hidden_dim, space_len, expert_num, label_lifting, size, padding=0):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.hidden_dim = hidden_dim
        self.space_len = space_len
        self.padding = padding # pad the domain if input is non-periodic
        self.size = size
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.gate = nn.ModuleList()
        
        for hdim in range(self.hidden_dim):
            self.gate.append(Gate_context1d(width, width, expert_num, label_lifting, size)) 
        
        self.fc0 = nn.Conv1d(input_dim, self.width, 1) 
        self.fc1 = nn.Conv1d(self.width, self.width, 1)
        for hdim in range(self.hidden_dim):
            self.conv_layers.append(Expert_WNO(self.level, self.width, self.expert_num, self.size))
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))
        
        self.fc2 = nn.Conv1d(self.width, 128, 1)
        self.fc3 = nn.Conv1d(128, 1, 1)

    def forward(self, x, label):
        """
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)
        x = self.fc1(x)
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) # do padding, if required
        
        lambda_ = []
        label = self.get_label(label, x.shape, x.device)
        for gate_ in self.gate:
            lambda_.append(gate_( x,label ))
          
        for wib, w0, lam in zip(self.conv_layers, self.w_layers, lambda_):
            x = wib(x, lam) + w0(x)
            x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding] # remove padding, when required
        x = self.fc2(x)
        x = F.mish(x)
        x = self.fc3(x)
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[-1]
        gridx = torch.tensor(np.linspace(0, self.space_len, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
    def get_label(self, label, shape, device):
        # Adds batch and channel to the label
        batchsize, channel_size, size_x = shape
        # label = label*torch.ones(size_x, device=device)
        # label = label.repeat(batchsize, self.label_lifting, 1)
        label = label.repeat(batchsize, channel_size, 1).to(device)
        return label.float() 

# %%
""" Model configurations """
data_path = []
data_path.append(directory + 'data/Advection_1D_pde_x512_T50_N1500.mat')
data_path.append(directory + 'data/Nagumo_1D_pde_x512_T50_N1500.mat')
data_path.append(directory + 'data/Allen_Cahn_1D_pde_x512_T50_N1500_v1em4.mat')
data_path.append(directory + 'data/Burgers_1D_pde_x512_T50_N1500_pdepde.mat')
data_path.append(directory + 'data/Heat_1D_pde_x512_T50_N1500.mat')
data_path.append(directory + 'data/Wave_1D_pde_x512_T50_N1500_c2.mat')

case_len = len(data_path)
data_label = torch.arange(1, case_len+1)

ntrain = 1400
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 300
scheduler_step = 20
scheduler_gamma = 0.5

level = 4 
width = 128

T = 20
T0 = 10
step = 1
sub = 2
S = 256

# %%
""" Read data """
data = []
for path in data_path:
    print('Loading:',path)
    data.append( (MatReader(path).read_field('sol')[::sub,:,:]).permute(2,1,0) )

train_a, train_u, test_a, test_u = ( [] for i in range(4) )
for case in range(case_len):
    train_a.append( data[case][:ntrain, :T0, :] )
    train_u.append( data[case][:ntrain, T0:T0+T, :] )
    test_a.append( data[case][-ntest:, :T0, :] )
    test_u.append( data[case][-ntest:, T0:T0+T, :] )

train_loader, test_loader = [], []
for case in range(case_len):
    train_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a[case], train_u[case]), 
                                           batch_size=batch_size, shuffle=True) )
    test_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a[case], test_u[case]), 
                                          batch_size=batch_size, shuffle=False) )

# %%
""" The model definition """
model = WNO1d(width=width, level=level, input_dim=T0+1, hidden_dim=4, space_len=1, 
              expert_num=10, label_lifting=2**4, size=S).to(device) 
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    epoch_train_step = np.zeros(case_len)
    epoch_train_full = np.zeros(case_len)
    
    for i, case_loader in enumerate(train_loader):
        case_train_step = 0
        case_train_full = 0
        for xx, yy in case_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            
            for t in range(0, T, step):
                y = yy[:, t:t + step, :]
                im = model(xx,data_label[i])
                loss += myloss(im.reshape(im.shape[0], -1), y.reshape(im.shape[0], -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                xx = torch.cat((xx[:, step:, :], im), dim=1)
                
            case_train_step += loss.item()
            case_train_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_train_step[i] = case_train_step
        epoch_train_full[i] = case_train_full

    epoch_test_step = np.zeros(case_len)
    epoch_test_full = np.zeros(case_len)
    with torch.no_grad():
        for i, case_loader in enumerate(test_loader):
            case_test_step = 0
            case_test_full = 0
            for xx, yy in case_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
    
                for t in range(0, T, step):
                    y = yy[:, t:t + step, :]
                    im = model(xx,data_label[i])
                    loss += myloss(im.reshape(im.shape[0], -1), y.reshape(im.shape[0], -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    xx = torch.cat((xx[:, step:, :], im), dim=1)
    
                case_test_step += loss.item()
                case_test_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
            epoch_test_step[i] = case_test_step
            epoch_test_full[i] = case_test_full

    t2 = default_timer()
    scheduler.step()
    train_error = [val/ntrain/(T/step) for val in epoch_train_step]
    test_error = [val/ntest/(T/step) for val in epoch_test_step]
    print('Epoch-{}, Time-{:0.4f}, Train_error: PDE_1-{:0.4f}, PDE_2-{:0.4f}, PDE_3-{:0.4f}, PDE_4-{:0.4f}, PDE_5-{:0.4f}, \n \
          \t \t \t Test_error: PDE_1-{:0.4f}, PDE_2-{:0.4f}, PDE_3-{:0.4f}, PDE_4-{:0.4f}, PDE_5-{:0.4f}'
          .format(ep, t2-t1, train_error[0], train_error[1], train_error[2], train_error[3], train_error[4],
                  test_error[0], test_error[1], test_error[2], test_error[3], test_error[4] ))

# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'data/model/NCWNO_Multiphysics_1D_L4_B20_S20_L4_W128_T20_Epoch200_v1')

# %%
""" Prediction """
pred_total = []
test_error = []
with torch.no_grad():
    for i, case_loader in enumerate(test_loader):
        pred_case = []
        case_test_step = 0
        for xx, yy in case_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
    
            for t in range(0, T, step):
                y = yy[:, t:t + step, :]
                im = model(xx,data_label[i])
                loss += myloss(im.reshape(im.shape[0], -1), y.reshape(im.shape[0], -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                xx = torch.cat((xx[:, step:, ...], im), dim=1)
                
            case_test_step += loss.item()
            pred_case.append( pred.cpu().numpy() )            
            
        print('Case-{}, Case_test_step_error-{:0.4f}'.format( i, case_test_step/ntest/(T/step) ))
        pred_total.append( np.row_stack((pred_case)) )
        test_error.append( case_test_step/ntest/(T/step) )

# %%
pdes = ['Advection', 'Nagumo', 'Allen-Cahn', 'Burgers', 'Heat', 'Wave']
[print('Mean Testing Error for PDE-{} : {:0.6f}'.format(pdes[i], 100*case), '%')
         for i, case in enumerate(test_error)]

