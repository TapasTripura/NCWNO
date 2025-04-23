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
from matplotlib.gridspec import GridSpec

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
            # self.gate[hdim](torch.randn(1,width,size), torch.randn(1,label_lifting,size))
        
        self.fc0 = nn.Conv1d(input_dim, self.width, 1) # input channel is 2: (a(x), x)
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
data_path.append('data/Advection_1D_pde_x512_T50_N1500.mat')
data_path.append('data/Nagumo_1D_pde_x512_T50_N1500.mat')
data_path.append('data/Allen_Cahn_1D_pde_x512_T50_N1500_v1em4.mat')
data_path.append('data/Burgers_1D_pde_x512_T50_N1500_pdepde.mat')
data_path.append('data/Heat_1D_pde_x512_T50_N1500.mat')
data_path.append('data/Wave_1D_pde_x512_T50_N1500_c2.mat')

case_len = len(data_path)
data_label = torch.arange(1, case_len+1)

ntrain = 1400
ntest = 100

batch_size = 20

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
model = torch.load('data/model/NCWNO_Multiphysics_1D_L4_B20_S20_L4_W128_T20_Epoch200_v1',map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
""" Prediction """
pred_total = []
test_error = []
with torch.no_grad():
    for i, case_loader in enumerate(test_loader):
        t0 = default_timer()
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
        t1 = default_timer()
            
        print('Case-{}, Time-{:0.4f}, Case_test_step_error-{:0.4f}'.format( i, t1-t0, case_test_step/ntest/(T/step) ))
        pred_total.append( np.row_stack((pred_case)) )
        test_error.append( case_test_step/ntest/(T/step) )

test_u = np.array(torch.stack((test_u)))
pred_total = np.stack((pred_total))

# %%
pdes = ['Advection', 'Nagumo', 'Allen-Cahn', 'Burgers', 'Heat', 'Wave']
[print('Mean Testing Error for PDE-{} : {:0.6f}'.format(pdes[i], 100*case), '%')
         for i, case in enumerate(test_error)]

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 12

figure3 = plt.figure(figsize = (16, 12), dpi=100)
gs = GridSpec(8, 9, figure=figure3, width_ratios=[1, 1, 1, 1, 0.05, 1, 1, 1, 1],
                                    height_ratios=[1, 0.7, 0.05, 1, 0.7, 0.05, 1, 0.7])
gs.update(hspace=0.8, wspace=0.6)

sample = [12,94,98,63,36,43]
xaxis = np.linspace(0,1,S)
xaxis_burger = np.linspace(-1,1,S)

for rows in range(3):
    for case in range(2):
        cmin = np.min(test_u[case + 2*rows][sample[case + 2*rows]])
        cmax = np.max(test_u[case + 2*rows][sample[case + 2*rows]])
        
        ax1 = plt.subplot(gs[rows*2+rows, 4*case+case:4*case+2+case])
        if rows*2+rows == 3 and 4*case+case == 5:
            im = ax1.imshow(test_u[case + 2*rows][sample[case + 2*rows]], aspect='auto', origin='lower', label='True',
                                extent=[-1,1,10,T+10], cmap='turbo', vmin=cmin, vmax=cmax)
        else:
            im = ax1.imshow(test_u[case + 2*rows][sample[case + 2*rows]], aspect='auto', origin='lower', label='True',
                                    extent=[0,1,10,T+10], cmap='turbo', vmin=cmin, vmax=cmax)    
        plt.colorbar(im, ax=ax1, aspect=10, pad=0.015)
        ax1.set_ylabel('Time ($t$)', fontweight='bold', color='brown')
        ax1.set_xlabel('$x$', fontweight='bold', color='brown')
        ax1.set_title('Truth, Sample:{}'.format(sample[case + 2*rows]), color='r', fontweight='bold')
        ax1.axhline(y=14, linestyle='--', color='w', linewidth = 1)
        ax1.axhline(y=18, linestyle='--', color='w', linewidth = 1)
        ax1.axhline(y=22, linestyle='--', color='w', linewidth = 1)
        ax1.axhline(y=26, linestyle='--', color='w', linewidth = 1)
        
        ax2 = plt.subplot(gs[rows*2+rows, 4*case+2+case:4*case+4+case])
        if rows*2+rows == 3 and 4*case+case == 5:
            im = ax2.imshow(pred_total[case + 2*rows][sample[case + 2*rows]], aspect='auto', origin='lower', 
                                    extent=[-1,1,10,T+10], cmap='turbo', interpolation='Gaussian', vmin=cmin, vmax=cmax)
        else:
            im = ax2.imshow(pred_total[case + 2*rows][sample[case + 2*rows]], aspect='auto', origin='lower', 
                                    extent=[0,1,10,T+10], cmap='turbo', interpolation='Gaussian', vmin=cmin, vmax=cmax)
        plt.colorbar(im, ax=ax2, aspect=10, pad=0.015)
        ax2.set_xlabel('$x$', fontweight='bold', color='brown')
        ax2.set_title('NCWNO, sample:{}'.format(sample[case + 2*rows]), color='b', fontweight='bold')
        ax2.axhline(y=14, linestyle='--', color='w', linewidth = 1)
        ax2.axhline(y=18, linestyle='--', color='w', linewidth = 1)
        ax2.axhline(y=22, linestyle='--', color='w', linewidth = 1)
        ax2.axhline(y=26, linestyle='--', color='w', linewidth = 1)
        
        ax3 = plt.subplot(gs[rows*2+1+rows, 4*case+0+case])
        if rows*2+1+rows == 4 and 4*case+0+case > 4:
            ax3.plot(xaxis_burger, test_u[case + 2*rows][sample[case + 2*rows], 4, :], 'r', label='Truth')
            ax3.plot(xaxis_burger, pred_total[case + 2*rows][sample[case + 2*rows], 4, :], 'b--', linewidth=2, label='NCWNO')
        else:
            ax3.plot(xaxis, test_u[case + 2*rows][sample[case + 2*rows], 4, :], 'r', label='Truth')
            ax3.plot(xaxis, pred_total[case + 2*rows][sample[case + 2*rows], 4, :], 'b--', linewidth=2, label='NCWNO')
        ax3.legend(frameon=False, fontsize=16, handletextpad=0.25, ncol=2, 
                   columnspacing=0.75, bbox_to_anchor=(4,1.60))
        ax3.set_xlabel('Step=14', labelpad=0.01)
        ax3.set_ylim([cmin+0.1*cmin, cmax+0.1*cmax])
        ax3.grid(True, alpha=0.25)
        
        ax4 = plt.subplot(gs[rows*2+1+rows, 4*case+1+case])
        if rows*2+1+rows == 4 and 4*case+0+case > 4:
            ax4.plot(xaxis_burger, test_u[case + 2*rows][sample[case + 2*rows], 14, :], 'r', label='Truth')
            ax4.plot(xaxis_burger, pred_total[case + 2*rows][sample[case + 2*rows], 14, :], 'b--', linewidth=2, label='NCWNO')
        else:
            ax4.plot(xaxis, test_u[case + 2*rows][sample[case + 2*rows], 14, :], 'r', label='Truth')
            ax4.plot(xaxis, pred_total[case + 2*rows][sample[case + 2*rows], 14, :], 'b--', linewidth=2, label='NCWNO')
        ax4.set_xlabel('Step=18', labelpad=0.01)
        ax4.set_ylim([cmin+0.1*cmin, cmax+0.1*cmax])
        ax4.yaxis.set_ticklabels([])
        ax4.grid(True, alpha=0.25)
        
        ax5 = plt.subplot(gs[rows*2+1+rows, 4*case+2+case])
        if rows*2+1+rows == 4 and 4*case+0+case > 4:
            ax5.plot(xaxis_burger, test_u[case + 2*rows][sample[case + 2*rows], 16, :], 'r', label='Truth')
            ax5.plot(xaxis_burger, pred_total[case + 2*rows][sample[case + 2*rows], 16, :], 'b--', linewidth=2, label='NCWNO')
        else:
            ax5.plot(xaxis, test_u[case + 2*rows][sample[case + 2*rows], 16, :], 'r', label='Truth')
            ax5.plot(xaxis, pred_total[case + 2*rows][sample[case + 2*rows], 16, :], 'b--', linewidth=2, label='NCWNO')
        ax5.set_xlabel('Step=22', labelpad=0.01)
        ax5.set_ylim([cmin+0.1*cmin, cmax+0.1*cmax])
        ax5.yaxis.set_ticklabels([])
        ax5.grid(True, alpha=0.25)
        
        ax6 = plt.subplot(gs[rows*2+1+rows, 4*case+3+case])
        if rows*2+1+rows == 4 and 4*case+0+case > 4:
            ax6.plot(xaxis_burger, test_u[case + 2*rows][sample[case + 2*rows], 18, :], 'r', label='Truth')
            ax6.plot(xaxis_burger, pred_total[case + 2*rows][sample[case + 2*rows], 18, :], 'b--', linewidth=2, label='NCWNO')
        else:
            ax6.plot(xaxis, test_u[case + 2*rows][sample[case + 2*rows], 18, :], 'r', label='Truth')
            ax6.plot(xaxis, pred_total[case + 2*rows][sample[case + 2*rows], 18, :], 'b--', linewidth=2, label='NCWNO')
        ax6.set_xlabel('Step=26', labelpad=0.01)
        ax6.set_ylim([cmin+0.1*cmin, cmax+0.1*cmax])
        ax6.yaxis.set_ticklabels([])
        ax6.grid(True, alpha=0.25)

figure3.text(0.30, 0.34, pdes[4] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
figure3.text(0.72, 0.34, pdes[5] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
figure3.text(0.30, 0.625, pdes[2] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
figure3.text(0.72, 0.625, pdes[3] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
figure3.text(0.30, 0.91, pdes[0] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
figure3.text(0.72, 0.91, pdes[1] + ' Equation', ha='center', fontsize=20, color='indigo', fontweight='bold')
plt.show()

figure3.savefig('results/ncwno_multiphysics_1d.pdf', format='pdf', dpi=300, bbox_inches='tight')

# %%
mean = np.zeros((case_len, T))
mean_plus = np.zeros((case_len, T))
mean_minus = np.zeros((case_len, T))
test_e_store_case = np.zeros((case_len,ntest, T))

for case in range(case_len):
    for i in range(ntest):
        for j in range(T):
            test_e_store_case[case,i,j] = np.linalg.norm(test_u[case,i,j,:]-pred_total[case,i,j,:])/ \
                                            np.linalg.norm(test_u[case,i,j,:])

mean = np.mean(test_e_store_case, 1)
mean_plus = np.mean(test_e_store_case, 1) + 1.96*np.std(test_e_store_case, 1)
mean_minus = np.mean(test_e_store_case, 1) - 1.96*np.std(test_e_store_case, 1)

# %%
plt.rcParams['font.size'] = 20

figure1, axes = plt.subplots(figsize = (10, 6), dpi=100)
figure1.subplots_adjust(hspace=0.5)

for case in range(case_len):
    plt.plot(np.arange(11,T+11,1), mean[case], label=pdes[case], marker='o')
    plt.fill_between(np.arange(11,T+11,1), mean_plus[case], mean_minus[case], alpha=0.2)
plt.ylabel('L$^2$ Relative error')
plt.xlabel('Time steps')
plt.title('Simultaneously learning multiple 1D PDEs')
plt.legend(loc=2, handletextpad=0.25, labelspacing=0.25)
plt.margins(0)
plt.grid(True, alpha=0.5)
plt.ylim([0,0.1])
plt.xlim([14,30])
plt.show()

figure1.savefig('results/ncwno_multiphysics_1d_error.pdf', format='pdf', dpi=300, bbox_inches='tight')

