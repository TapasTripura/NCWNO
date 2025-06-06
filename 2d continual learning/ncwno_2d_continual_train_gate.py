"""
NCWNO for continual learning of "2D time-dependent PDEs" with "10 experts"

It requires the package "Pytorch Wavelets"
-- see https://pytorch-wavelets.readthedocs.io/en/latest/readme.html

-- It trains the foundation model. 
-- For sequential training it is suggested to train the gate network 
   in a seperate script, to avoid computational graph entanglement
"""

import os
directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
import torch
import numpy as np
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
class Expert_WNO2d(nn.Module):
    def __init__(self, level, width, expert_num, size):
        super(Expert_WNO2d, self).__init__()

        """
        The Expert Wavelet Integral Blocks

        Input Parameters
        ----------------
        level      : scalar, levels of wavelet decomposition 
        width      : scalar, kernel dimension in lifted space
        expert_num : scalar, number of local wavelet experts 
        size       : scalar, length of input 2D signal

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x)
        """

        self.level = level
        self.width = width
        self.expert_num = expert_num
        self.Expert_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        wavelet1 = ['antonini', 'antonini', 'legall', 'legall', 'near_sym_a', 'near_sym_a', 'near_sym_b', 'near_sym_b']
        wavelet2 = ['qshift_06', 'qshift_a', 'qshift_06', 'qshift_b', 'qshift_a', 'qshift_c', 'qshift_b', 'qshift_d']
        self.Expert_layers0=WaveConv2dcwt(self.width, self.width, self.level[0], size, wavelet1[0], wavelet2[0])
        self.Expert_layers1=WaveConv2dcwt(self.width, self.width, self.level[1], size, wavelet1[1], wavelet2[1])
        self.Expert_layers2=WaveConv2dcwt(self.width, self.width, self.level[2], size, wavelet1[2], wavelet2[2])
        self.Expert_layers3=WaveConv2dcwt(self.width, self.width, self.level[3], size, wavelet1[3], wavelet2[3])
        self.Expert_layers4=WaveConv2dcwt(self.width, self.width, self.level[4], size, wavelet1[4], wavelet2[4])
        self.Expert_layers5=WaveConv2dcwt(self.width, self.width, self.level[5], size, wavelet1[5], wavelet2[5])
        self.Expert_layers6=WaveConv2dcwt(self.width, self.width, self.level[6], size, wavelet1[6], wavelet2[6])
        self.Expert_layers7=WaveConv2dcwt(self.width, self.width, self.level[7], size, wavelet1[7], wavelet2[7])

    def forward(self, x, lambda_):
        x = lambda_[..., 0:1, :]*self.Expert_layers0(x) + lambda_[..., 1:2, :]*self.Expert_layers1(x) + \
            lambda_[..., 2:3, :]*self.Expert_layers2(x) + lambda_[..., 3:4, :]*self.Expert_layers3(x) + \
            lambda_[..., 4:5, :]*self.Expert_layers4(x) + lambda_[..., 5:6, :]*self.Expert_layers5(x) + \
            lambda_[..., 6:7, :]*self.Expert_layers4(x) + lambda_[..., 7:8, :]*self.Expert_layers5(x)
        return x
    
    
""" The forward operation """
class NCWNO2d(nn.Module):
    def __init__(self, width, level, input_dim, hidden_dim, space_len, label_lifting, size, expert_num=8, padding=0):
        super(NCWNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
              
        Input parameters:
        -----------------
        width     : scalar, lifting dimension of input
        level     : scalar, number of wavelet decomposition
        input_dim : scalar, number of input channels including grids
        hidden_dim: scalar, number of wavelet kernel integral blocks
        space_len : the 2D domain length
        wavelet   : string, wavelet filter
        expert_num: number of local wavelet experts
        size      : scalar, signal length
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.hidden_dim = hidden_dim
        self.space_len = space_len
        self.padding = padding # pad the domain if input is non-periodic
        self.expert_num = expert_num
        self.label_lifting = label_lifting
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.gate = nn.ModuleList()
        
        for hdim in range(self.hidden_dim):
            self.gate.append(Gate_context2dcwt(width, width, expert_num, label_lifting, size))

        self.fc0 = nn.Conv2d(input_dim, self.width, 1) # input channel is 2: (a(x), x)
        self.fc1 = nn.Conv2d(self.width, self.width, 1)
        for hdim in range(self.hidden_dim):
            self.conv_layers.append(Expert_WNO2d(self.level, self.width, self.expert_num, size))
            self.w_layers.append(nn.Conv2d(self.width, self.width, 1))

        self.fc2 = nn.Conv2d(self.width, 128, 1)
        self.fc3 = nn.Conv2d(128, 1, 1)

    def forward(self, x, label):
        """
        Input : 3-channel tensor, Initial input and location (a(x,y), x, y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        x = self.fc0(x)
        x = self.fc1(x)
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) # do padding, if required
            
        lambda_ = []
        label = self.get_label(label, x.shape, x.device)
        for gate_ in self.gate:
            lambda_.append(gate_( x,label ))
            
        for ll, (wib, w0, lam) in enumerate(zip(self.conv_layers, self.w_layers, lambda_)):
            x = wib(x, lam) + w0(x)
            if ll != self.hidden_dim - 1:
                x = F.mish(x)
        
        # label = self.get_label(label, x.shape, x.device)
        # for wib, w0, gate_ in zip(self.conv_layers, self.w_layers, self.gate):
        #     lam = gate_(x, label)
        #     x = wib(x, lam) + w0(x)
        #     x = F.mish(x)

        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding] # remove padding, when required
        x = self.fc2(x)
        x = F.mish(x)
        x = self.fc3(x)
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(np.linspace(0, self.space_len[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, self.space_len[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
    
    def get_label(self, label, shape, device):
        # Adds batch and channel to the label
        batchsize, channel_size, size_x, size_y = shape 
        # label = label*torch.ones(size_x, size_y, device=device)
        # label = label.repeat(batchsize, self.label_lifting, 1, 1)
        label = label.repeat(batchsize, channel_size, 1).to(device)
        return label.float() 
    
# %%
""" Model configurations """
data_path = []
data_path.append('data/test_NS_2D_pde_x64_x64_T50_N100.mat')
data_path.append('data/test_Allen_Cahn_2D_pde_x128_y128_T50_N100.mat')
data_path.append('data/test_Nagumo_2D_pde_x128_y128_T50_N100.mat')
data_path.append('data/test_Burgers_2D_pde_x128_y128_T50_N100.mat')
data_path.append('data/test_Advection_2D_pde_x128_y128_T50_N100.mat')
data_path.append('data/test_Heat_2D_pde_x128_y128_T50_N100.mat')

case_len = len(data_path)
data_label = torch.arange(1, case_len+1)

ntrain = 600
ntest = 100

batch_size = 20

T = 10
T0 = 10
step = 1
sub = 2
sub_ns = 1
S = 64

# %%
""" Read data """
data = []
for i, path in enumerate(data_path):
    print('Loading:',path)
    data.append( MatReader(path).read_field('sol') )

train_a, train_u, test_a, test_u = ( [] for i in range(4) )
for case in range(case_len):
    train_a.append( data[case][:ntrain, :T0, :, :] )
    train_u.append( data[case][:ntrain, T0:T0+T, :, :] )
    test_a.append( data[case][-ntest:, :T0, :, :] )
    test_u.append( data[case][-ntest:, T0:T0+T, :, :] )

train_loader, test_loader = [], []
for case in range(case_len):
    train_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a[case], train_u[case]), 
                                           batch_size=batch_size, shuffle=True) )
    test_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a[case], test_u[case]), 
                                          batch_size=batch_size, shuffle=False) )

# %%
""" The model definition """
model = torch.load('data/model/Foundation_2d_10exp_0', map_location=device)
print(count_params(model))

myloss = LpLoss(size_average=False)
pde_no = 3

# %%
""" Fixing the convolution parameters of model
    conv_layers = wavelet convolution layers 
    w_layers = skip linear convolution layers 
    fc0 = uplifting layer 0 
    fc1 = uplifting layer 1 
    fc2 = downlifting layer 0 
    fc3 = downlifting layer 2 
"""

for l in range(4):
    for p in model.conv_layers[l].parameters():
        p.requires_grad = False
        print('Wavelet_Hidden_layer-{}, parameter_requires_grad-{}'.format(l, p.requires_grad))

for l in range(4):
    for p in model.w_layers[l].parameters():
        p.requires_grad = False
        print('SKIP_Hidden_layer-{}, parameter_requires_grad-{}'.format(l, p.requires_grad))
        
for p in model.fc0.parameters():
    p.requires_grad = False
    print('FC-0: parameter_requires_grad-{}', p.requires_grad)

for p in model.fc1.parameters():
    p.requires_grad = False
    print('FC-1: parameter_requires_grad-{}', p.requires_grad)
    
for p in model.fc2.parameters():
    p.requires_grad = False
    print('FC-2: parameter_requires_grad-{}', p.requires_grad)
    
for p in model.fc3.parameters():
    p.requires_grad = False
    print('FC-3: parameter_requires_grad-{}', p.requires_grad)

# %%
""" Train Gate's for Task-4 onwards """
learning_rate = 0.001
epochs = 150
scheduler_step = 20 
scheduler_gamma = 0.5

for case in range(pde_no, case_len):
    case_loader = train_loader[ case ]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step,
                                                gamma=scheduler_gamma)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        epoch_train_step = 0
        
        for xx, yy in case_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            
            for t in range(0, T, step):
                y = yy[:, t:t + step, :]
                im = model(xx, data_label[case])
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                xx = torch.cat((xx[:, step:, :], im), dim=1)
                
            epoch_train_step += loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        with torch.no_grad():
            for j in range(case+1):
                epoch_test_step = 0
                loader = test_loader[j]
                
                for xx, yy in loader:
                    loss = 0
                    xx = xx.to(device)
                    yy = yy.to(device)
        
                    for t in range(0, T, step):
                        y = yy[:, t:t + step, :]
                        im = model(xx, data_label[j])
                        loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), 1)
                        xx = torch.cat((xx[:, step:, :], im), dim=1)
        
                    epoch_test_step += loss.item()
                
                print('Epoch-{}, PDE_learned-{}, PDE_test-{}, Train_step-{:0.4f}, Test_step-{:0.4f}'
                      .format(ep, case, j, epoch_train_step/ntrain/(T/step), epoch_test_step/ntest/(T/step) ))
            
        t2 = default_timer()
        print('Total Time-{:0.4f}'.format(t2-t1))
    
    # Sequentially save the models, so that the trained gate can be used later on,
    # note that the wavelet parameters are same in all the models
    torch.save(model, 'data/model/Combinatorial_2d_10exp_'+str(case)) 

# %%
""" Prediction """
pred_total = []
test_error = []
with torch.no_grad():
    for i, case_loader in enumerate(test_loader):
        if i < pde_no:
            model_test = torch.load('data/model/Foundation_2d_10exp_0', map_location=device) #model_0
        else:
            model_test = torch.load('data/model/Combinatorial_2d_10exp_'+str(i), map_location=device)
        pred_case = []
        case_test_step = 0
        for xx, yy in case_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[:, t:t + step, ...]
                im = model(xx, data_label[i])
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), 1)
                xx = torch.cat((xx[:, step:, ...], im), dim=1)
                
            case_test_step += loss.item()
            pred_case.append( pred.cpu().numpy() )
            
        print('Case-{},Case_test_step-{:0.4f}'.format( i, case_test_step/1/(T/step) ) )
        pred_total.append( np.row_stack(( pred_case )) )
        test_error.append( case_test_step/ntest/(T/step) )
        
# %%
pdes = ['NS', 'Allen-Cahn', 'Nagumo', 'Burgers', 'Advection', 'Heat']
[print('Mean Testing Error for PDE-{} : {:0.6f}'.format(pdes[i], 100*test_error[i]), '%') \
         for i, case in enumerate(test_error)]

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 10

""" Plotting """ 
figure7 = plt.figure(figsize = (16, 8), dpi=100)
plt.subplots_adjust(hspace=0.4, wspace=0.3)

sample = [45,83,95]
case_index = 0

for case in range(case_len):
    index = 0
    cmin = np.min(test_u[case].numpy()[sample[case],:,:,:])
    cmax = np.max(test_u[case].numpy()[sample[case],:,:,:])
    for time in range(T):
        if time % 4 == 0: # in a row 3 samples of actual,
            plt.subplot(case_len,6, index+1+case_index)
            plt.imshow(test_u[case].numpy()[sample[case],time,:,:], aspect='auto', origin='lower', label='True',
                        extent=[0,257,0,20], cmap='jet', vmin=cmin, vmax=cmax)
            plt.title('Actual, Time-{}'.format(time), color='b', fontweight='bold')
            plt.colorbar()
            if any(index+1+case_index == check for check in np.array([1,7,13,19,25])):
                plt.ylabel('{} PDE'.format(pdes[case]), color='brown', fontweight='bold')
            ###############################################,
            plt.subplot(case_len,6, index+4+case_index)
            plt.imshow(pred_total[case][sample[case],time,:,:], aspect='auto', origin='lower', 
                        extent=[0,257,0,20], cmap='jet', vmin=cmin, vmax=cmax)
            plt.title('Identified, TIme-{}'.format(time), color='r', fontweight='bold')
            plt.colorbar()
            index += 1
    case_index += 6
plt.show()
