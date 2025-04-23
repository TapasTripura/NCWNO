"""
NCWNO for continual learning of "1D time-dependent PDEs" with "10 experts"

It requires the package "Pytorch Wavelets"
-- see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html

This code is for testing after sequentially training over all PDEs
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
""" Def: 1d Wavelet convolution layer """
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
class NCWNO1d(nn.Module):
    def __init__(self, width, level, input_dim, hidden_dim, space_len, expert_num, label_lifting, size, padding=0):
        super(NCWNO1d, self).__init__()

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
        
        # label = self.get_label(label, x.shape, x.device)
        # for wib, w0, gate_ in zip(self.conv_layers, self.w_layers, self.gate):
        #     lam = gate_(x, label)
        #     x = wib(x, lam) + w0(x)
        #     x = F.mish(x)
            
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
data_path.append('data/Allen_Cahn_1D_pde_x512_T50_N1500_v1em4.mat')
data_path.append('data/Nagumo_1D_pde_x512_T50_N1500.mat')
data_path.append('data/Wave_1D_pde_x512_T50_N1500_c2.mat')
data_path.append('data/Burgers_1D_pde_x512_T50_N1500_pdepde.mat')
data_path.append('data/Advection_1D_pde_x512_T50_N1500.mat')
data_path.append('data/Heat_1D_pde_x512_T50_N1500.mat')

case_len = len(data_path)
data_label = torch.arange(1, case_len+1)

ntrain = 1000
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
""" Load the foundation model """
model = torch.load('data/model/Foundation_1d_10exp_0', map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)
pde_no = 3

# %%
""" Prediction """
pred_continual_total = []
test_continual_error = []
with torch.no_grad():
    for learner in range(case_len):
        if learner < pde_no:
            model_test = torch.load('data/model/Foundation_1d_10exp_0', map_location=device) # Foundation model
        else:
            model_test = torch.load('data/model/Combinatorial_1d_10exp_'+str(learner), map_location=device)
        
        pred_total = []
        test_error = []
        for i, case_loader in enumerate(test_loader):
            
            pred_case = []
            case_test_step = 0
            for xx, yy in case_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
        
                for t in range(0, T, step):
                    y = yy[:, t:t + step, :]
                    im = model_test(xx,data_label[i])
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), 1)
                    xx = torch.cat((xx[:, step:, ...], im), dim=1)
                    
                case_test_step += loss.item()
                pred_case.append( pred.cpu().numpy() )
                
            print('Learner-{}, Case-{}, Case_test_step_error-{:0.4f}'.format( learner, i,
                                                                             case_test_step/ntest/(T/step) ) )
            pred_total.append( np.row_stack(( pred_case )) )
            test_error.append( case_test_step/ntest/(T/step) )
        pred_continual_total.append( np.stack((pred_total)) )
        test_continual_error.append( np.stack((test_error)) )

test_continual_error = np.array(test_continual_error) # row wise test,
test_u_1d = np.array(torch.stack((test_u)))
pred_total_1d = np.stack((pred_continual_total))

# %%
pdes = ['Allen-Cahn', 'Nagumo', 'Wave', 'Burgers', 'Advection', 'Heat']
[print('Mean Testing Error for PDE-{} : {:0.6f}'.format(pdes[i], 100*case[i]), '%')
             for i, case in enumerate(test_continual_error)]

# %%
test_e_store_case_1d = np.zeros((case_len, case_len, ntest, T))

for learner in range(case_len):
    for case in range(case_len):
        for i in range(ntest):
            for j in range(T):
                test_e_store_case_1d[learner,case,i,j] = np.linalg.norm(
                    test_u_1d[case,i,j,:]-pred_total_1d[learner,case,i,j,:]) / np.linalg.norm(test_u_1d[case,i,j,:])

# %%
test_e_store_case_1d[2:, 0] = test_e_store_case_1d[0, 0] 
test_e_store_case_1d[2:, 1] = test_e_store_case_1d[0, 1] 
test_e_store_case_1d[3:, 2] = test_e_store_case_1d[2, 2] 
test_e_store_case_1d[4:, 3] = test_e_store_case_1d[3, 3] 
test_e_store_case_1d[5:, 4] = test_e_store_case_1d[4, 4] 

accuracy_1d = 1 - test_e_store_case_1d
accuracy_1d = np.delete(accuracy_1d, (0,1), 0)

mean_1d = np.mean(accuracy_1d, 2).transpose(1,0,2)
mean_plus_1d = (np.mean(accuracy_1d, 2) + 1.96*np.std(accuracy_1d, 2)).transpose(1,0,2)
mean_minus_1d = (np.mean(accuracy_1d, 2) - 1.96*np.std(accuracy_1d, 2)).transpose(1,0,2)

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 20

pdesn = ['Allen-Cahn & Nagumo\n   &Wave', 'Burgers', 'Advection', 'Heat']
colorx = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

figure1, ax = plt.subplots(nrows=case_len, ncols=case_len-2, figsize = (16, 10), 
                           gridspec_kw={'width_ratios': [1.4, 1, 1, 1]}, dpi=100)
figure1.subplots_adjust(hspace=0.25, wspace=0.3)

index = 0
for learner in range(case_len):
    for case in range(case_len-2):
        
        ax[learner,case].plot(np.arange(T0+1,T0+T+1,1), mean_1d[learner, case], color=colorx[index])
        ax[learner,case].fill_between(np.arange(T0+1,T0+T+1,1), mean_plus_1d[learner, case], mean_minus_1d[learner, case],
                                      alpha=0.25, color=colorx[index])
        ax[learner,case].axhline( y=1, color='k', linestyle=':', linewidth=2 )
        ax[learner,case].set_xticks([11,15,20,25,30])
        if learner>2 and case==0:
            ax[learner,case].set_ylim([-1.10,1.10]) 
        elif learner>3 and case==1:
            ax[learner,case].set_ylim([-1.10,1.10]) 
        elif learner>4 and case==2:
            ax[learner,case].set_ylim([-1.10,1.10]) 
        else:
            ax[learner,case].set_ylim([0.8,1.02]) 
            ax[learner,case].set_yticks([0.8,0.9,1.0]) 
            ax[learner,case].tick_params(axis='y', colors='blue', width=2)
        ax[learner,case].grid(True, alpha=0.5)
        ax[learner,case].margins(0)
        
        if learner != case_len-1:
            ax[learner,case].xaxis.set_ticklabels([])
        if learner == 0 and case < case_len:
            ax[learner,case].set_title('Train:{}'.format(pdesn[case]), fontsize=20,
                                        pad=10, backgroundcolor='lavender', color='blue')
        if case == case_len-3:
            ax[learner,case].yaxis.set_label_position("right")
            ax[learner,case].set_ylabel('Test:\n{}'.format(pdes[learner]), rotation=270, labelpad=50,
                                        color='maroon', backgroundcolor='mistyrose')
    index += 1

figure1.text(0.43, 0.05, 'Prediction time steps', fontsize=24, color='indigo', fontweight='bold')
figure1.text(0.06, 0.38, 'Predictive accuracy', fontsize=24, color='indigo', fontweight='bold', rotation=90)
plt.show()

figure1.savefig('results/Results_1d_acc_continual.pdf', format='pdf', dpi=600, bbox_inches='tight')

# %%
# For plotting the solutions, extract only the diagonal solutions
pred_print = np.zeros((case_len, ntest, T, S))
for case in range(case_len):
    pred_print[case, ...] = pred_total_1d[case, case, ...]

# %%
plt.rcParams['font.size'] = 16

""" Plotting """ 
figure7 = plt.figure(figsize = (20, 14), dpi=100)
plt.subplots_adjust(hspace=0.30, wspace=0.50)

figure7.text(0.25,0.925,'Ground truth', fontsize=24, color='r', fontweight='bold')
figure7.text(0.13,0.92,'_'*48, fontweight='bold', fontsize=20, color='k')
figure7.text(0.68,0.925,'NCWNO', fontsize=24, color='b', fontweight='bold')
figure7.text(0.54,0.92,'_'*48, fontweight='bold', fontsize=20, color='k')

samples = [17, 41, 60]
case_index = 0
for case in range(case_len):
    index = 0
    for sample in range(ntest):
        if sample % 44 == 0: # in a row 3 samples of actual,
            plt.subplot(case_len, 6, index+1+case_index)
            if case == 3:
                plt.imshow(test_u_1d[case,samples[index]], aspect='auto', origin='lower', label='True',
                           extent=[0,1,10,30], cmap='jet', interpolation='Gaussian',
                           vmin=-0.2, vmax=0.2)
            else: 
                plt.imshow(test_u_1d[case,sample], aspect='auto', origin='lower', label='True',
                           extent=[0,1,10,30], cmap='turbo', interpolation='Gaussian')
            if case == 0:
                plt.title('IC = {}'.format(sample), color='r', fontweight='bold')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=16)
            if any(index+1+case_index == check for check in np.array([1,7,13,19,25,31])):
                plt.ylabel('1D {}\nTime (t)'.format(pdes[case]), color='brown', fontsize=20,
                           labelpad=15, fontweight='bold')
            if case == case_len-1:
                plt.xlabel('x', color='brown', fontsize=20, fontweight='bold')
            #
            #########################################
            #
            plt.subplot(case_len, 6, index+4+case_index)
            if case == 3:
                plt.imshow(pred_print[case,samples[index]], aspect='auto', origin='lower', 
                            extent=[0,1,10,30], cmap='jet', interpolation='Gaussian',
                            vmin=-0.2, vmax=0.2)
            else: 
                plt.imshow(pred_print[case,sample], aspect='auto', origin='lower', 
                            extent=[0,1,10,30], cmap='turbo', interpolation='Gaussian')
                
            if case == 0:
                plt.title('IC = {}'.format(sample), color='b', fontweight='bold')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=16)
            if case == case_len-1:
                plt.xlabel('x', color='brown', fontsize=20, fontweight='bold')
            index += 1
    case_index += case_len
plt.show()

figure7.savefig('results/ncwno_continual_1d.pdf', format='pdf', dpi=300, bbox_inches='tight')

