

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets.vision import VisionDataset
from torch.optim.optimizer import Optimizer, required
import pickle
from PIL import Image
import math
from pprint import pprint as prt
# import seaborn as sns; sns.set(color_codes=True);sns.set_style("white")
import functools
import itertools
import time
from time import time as timer
import os




from meta_module import *
from resnet_meta import *
# from utils import *

os.makedirs('wIns', exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')







def argsPrinter(args,n_epochs,epoch_len,unroll,Target_DataGen,OPT,preproc,hidden_layers_zer,pixels,batch_size,cifarAB,num_workers,n_tests,eva_epoch_len,pin_memory,Target_Optimizee,**w):
    
    # task_desc: used for wIns dir name, trained .pth file name, SR database name
    _l2o_net = OPT(**args)
    if args['mode'] == 'tras':
        task_desc = f'train_scratch {_l2o_net.name} @ {Target_Optimizee().name}'
    elif args['mode'] == 'tune':
        task_desc = f'tune {_l2o_net.name} @ {Target_Optimizee().name}'
    elif args['mode'] == 'nort':
        task_desc = f'normal_train resnet18'
    elif args['mode'] in ['pure', 'srck', 'srgen', 'rec', 'tradi']:
        task_desc = f"eva({args['mode']}), {_l2o_net.name} @ {Target_Optimizee().name}"
    elif args['mode'] is None:
        task_desc = 'None'
    else:
        raise ValueError

    args['task_desc'] = task_desc 

    s=f'''

        # ============ GENERAL ============
        'pid':                  {os.getpid()}
        'task_desc':            {task_desc}




        # ============ Training from scratch ============
        'n_epochs':             {n_epochs}
        'epoch_len':            {epoch_len}
        'unroll':               {unroll}

        'batch_size':           {batch_size}
        'num_workers':          {num_workers}
        'pin_memory':           {pin_memory}


        'l2o_net.name'          {_l2o_net.name}
        'Ngf'(num_grad_feat):   {args.get('Ngf')}
        'preproc':              {preproc}
        'hidden_layers_zer':    {hidden_layers_zer}




        # ============ Evaluation ============
        'n_tests':              {n_tests}
        'eva_epoch_len':        {eva_epoch_len}

        'l2o_net.name'          {_l2o_net.name}




        # ============ Which Problem ============

        'Target_DataGen':       {Target_DataGen}
        'Target_Optimizee':     {Target_Optimizee}
        'pixels'(784/3072):     {pixels}




        # ============ SR Data Gen ============
        'want_SR':              {args.get('want_SR')}
        'n_epochs':             {args.get('n_epochs_tuneSR')}
        'epoch_len':            {args.get('epoch_len_tuneSR')}
        'num_Xyitems_SR':       {args.get('num_Xyitems_SR')}
        'SR_memlen':            {args.get('SR_memlen')}



        # ============ Tune SR ============
        'n_epochs':             {args.get('n_epochs_tuneSR')}
        'epoch_len':            {args.get('epoch_len_tuneSR')}
        'unroll':               {args.get('unroll_tuneSR')}

        'lr_meta_tuneSR':       {args.get('lr_meta_tuneSR')}
        'OPT_META_SR':          {args.get('OPT_META_SR')}


    '''
    print(s)


    return



def fit_optimizer_from_scratch(args, OPT=None, pre_tahn_scale=False, **kwargs):

    l2o_net = OPT(**args).to(DEVICE)
    # viz(l2o_net,'l2o')
    if kwargs.get('LUM'): kwargs['lum'] = kwargs['LUM'](kwargs['lum_layers'])
    if kwargs.get('lum'):
        params = itertools.chain(l2o_net.parameters(),kwargs['lum'].parameters())
    else:
        params = l2o_net.parameters()
    if pre_tahn_scale:
        params = itertools.chain([l2o_net.pre_tahn_scale,], params)

    args['meta_opt'] = optim.Adam(params, lr=0.001)
    args['l2o_net'] = l2o_net
    return train_optimizer(args, **args)







class LUM(nn.Module):
  def __init__(self, layers, pretrained_path=None, input_onehot=True):
    super().__init__()

    # layers 包含input shape 一直到输出层的neuron个数
    self.input_onehot = input_onehot
    self.param_groups = layers[0]
    self.calledBefore = False
    if self.input_onehot:
      from sklearn.preprocessing import OneHotEncoder
      enc = OneHotEncoder(sparse=False)
      X = np.arange(self.param_groups).reshape(self.param_groups,1)
      enc = enc.fit(X)
      self.onehot = enc.transform(X)  # [param_groups, param_groups] 的array


    # self.mlp = snt.nets.MLP(layers, initializers=None)
    self.mlp = MLP(layers)
    return

  def forward(self, X):
    # 输入是flat的list，输入subset
    # 输出每个元素的lum, 是flat的 tensor
    if self.input_onehot:
      mlpInput = torch.tensor(self.onehot[X], dtype=torch.float32)
    else:
      mlpInput = torch.tensor(X).view(-1,1)

    return self.mlp(mlpInput).flatten()*0.01



def normal_train(args, Target_Optimizee=None,Target_DataGen=None, root=None,batch_size=None,num_workers=None,OPT_normal_train=None,n_epochs=None,lr_normal_train=None,shuffle_cifar=None,**kwargs):

    optimizee_save_path = 'wz_saved_models/cifar_net.pth'
    
    target = Target_DataGen(training=1,**args)
    
    
    net = Target_Optimizee(**args)
    

    optimizer = OPT_normal_train(net.parameters(), lr=lr_normal_train) 
    
    
    for epoch in tqdm(range(n_epochs)):
   
   
        optimizer.zero_grad()

        loss = net(target)

        loss.backward()
        optimizer.step()

    
        if epoch%100==0:
            torch.save(net.state_dict(), optimizee_save_path)
            # wzRec(losses_1epoch,'normal-train-loss, epoch-{}'.format(iepoch))


    torch.save(net.state_dict(), optimizee_save_path)

    return
        
    
    


def train_optimizer(args, l2o_net=None, OPT=None, Target_Optimizee=None,  epoch_len=100, n_epochs=20, n_tests=20, only_want_last=True, trainL2O_want_eva_in_training=None, **kwargs):

    argsPrinter(args,**args)
    cwd = f"wz_saved_models/{args['task_desc']}.pth"
    cwd_lum = f'wz_saved_models/lum~on~{Target_Optimizee().name}.pth'

    best_l2o, best_lum, best_loss = None, None, float('inf')
    for iepoch in tqdm(range(n_epochs), 'Meta training'):

        lt1=timer()
        losses_1epoch, sr_t_n_f = run_epoch(args, epoch_len, should_train=True, **args)
        lt2=timer()
        wzRec(losses_1epoch,'train-loss, {}, epoch-{}'.format(l2o_net.name,iepoch), args['task_desc'])
        lt3=timer()
        # print(f'\ntime_1epoch={lt2-lt1}, x={(lt2-lt1)/(lt3-lt2)}')


        if (iepoch+1) % 100==0:
            save_model(l2o_net, cwd)
            if kwargs.get('lum'): save_model(lum, cwd_lum)
            if trainL2O_want_eva_in_training:
                eva_l2o_optimizer(args, **args)


        if not only_want_last and (iepoch+1) % 10==0:
            #  一共 n_tests 这么多epoch的平均sum loss
            _, sum_loss, last_loss, _ = eva_l2o_optimizer(args, **args)
            print(f'\nbest={best_loss}, sum={sum_loss}, last={last_loss}')
            if last_loss < best_loss:
                best_loss = last_loss
                best_l2o = copy.deepcopy(l2o_net.state_dict())
                if kwargs.get('lum'): best_lum = copy.deepcopy(kwargs['lum'].state_dict())
    else:
        best_loss = losses_1epoch
        best_l2o = copy.deepcopy(l2o_net.state_dict())
        if kwargs.get('lum'): best_lum = copy.deepcopy(kwargs['lum'].state_dict())

    if not only_want_last: 
        # load optimal weights
        l2o_net = OPT(**args).to(DEVICE)
        l2o_net.load_state_dict(best_l2o)
        if kwargs.get('lum'): 
            kwargs['lum'] = kwargs['LUM'](kwargs['lum_layers'])
            kwargs['lum'].load_state_dict(best_lum)


    save_model(l2o_net, cwd)
    if kwargs.get('lum'): save_model(lum, cwd_lum)
    return l2o_net, kwargs.get('lum')






def eva_l2o_traditional(args, tradi_save_name, n_tests=None, want_save_eva_loss=None, **kwargs):
    
    argsPrinter(args,**args)
    res = [run_epoch_traditional(args, **args) for _ in tqdm(range(n_tests))]
    losses_1epoch_Nrun = res


    all_losses = np.asarray(losses_1epoch_Nrun)
    np.save(f'exp_rebuttal/{tradi_save_name}.npy', all_losses)

    figDir, recDir = wzRec(all_losses, f'Eva-loss', args['task_desc'], want_save=want_save_eva_loss)
    print(f'\n=======================\nEva result figure saved to:\n< {figDir} >\n=======================\n')
    # if want_save_eva_loss:
    #     print(f'All loss/acc records saved to:\n< {recDir} >\n=======================\n')

    return figDir


def run_epoch_traditional(args, lr_tradi, eva_epoch_len, Target_DataGen=None, Target_Optimizee=None, problem_is_transfer_learn=False,OPT_TRADI=None, **kwargs):

    target = Target_DataGen(training=False,**args)
    optimizee = Target_Optimizee(**args).to(DEVICE)
    if problem_is_transfer_learn:
        assert Target_Optimizee==args['assigned_zee']==resnet18
        assert target.cifarAB =='B'
    else:
        wIni(optimizee)

    # opt_tradi = OPT_TRADI(optimizee.parameters(), lr=0.001)



    print('which_opt??', args['which_opt'])
    if args['which_opt']=='A':
        opt_tradi = optim.Adam(optimizee.parameters(), lr=0.001, betas=(0.9,0.999)  )
    elif args['which_opt']=='S':
        opt_tradi = optim.SGD(optimizee.parameters(), lr=lr_tradi,  )
    else: raise NotImplementedError








    print(f'\n\n==============\n opt_tradi is {opt_tradi} \n==============')

    losses_1epoch = []


    for i_step in tqdm(range(1, eva_epoch_len + 1), 'Eva: '):

        loss = optimizee(target)

        losses_1epoch.append(loss.data.cpu().numpy())

        loss.backward()

        opt_tradi.step()
        

    return losses_1epoch









def run_epoch(args, run_epoch_len, should_train, meta_opt=None, l2o_net=None, Target_DataGen=None, Target_Optimizee=None, unroll=1, want_sr=None, problem_is_transfer_learn=False, **kwargs):
    l1=timer()

    lum = kwargs.get('lum')
    if should_train:
        l2o_net.train()
        if lum: lum.train()
    else:
        l2o_net.eval()
        if lum: lum.eval()
        unroll = 1

    target = Target_DataGen(training=should_train,**args)
    optimizee = Target_Optimizee(**args).to(DEVICE)
    if problem_is_transfer_learn:
        assert Target_Optimizee==args['assigned_zee']==resnet18
        assert target.cifarAB =='B'
        load_model(optimizee, args['optimizee_load_cwd'], verbose=True)
    else:
        wIni(optimizee)
    # viz(optimizee,'optimizee')
    n_params, sets = l2o_net.reset(optimizee.parameters())


    hidden_states = [Variable(torch.zeros(n_params, l2o_net.hidden_layers_zer[i])).to(DEVICE) for i in range(l2o_net.n_hidden)]   # shape = [  [15910, 20] , [15910, 20]  ] = len(l2o_net.hidden_layers_zer) x [n_params, l2o_net.hidden_layers_zer[i]]

    cell_states = [Variable(torch.zeros(n_params, l2o_net.hidden_layers_zer[i])).to(DEVICE) for i in range(l2o_net.n_hidden)]

    losses_1epoch, all_losses = [], None


    if should_train:
        meta_opt.zero_grad()
        iterator = range(1, run_epoch_len + 1)
    else:
        iterator = tqdm(range(1, run_epoch_len + 1), 'Eva: ')

    
    sr_t_n_f = [] # [i_step][i_group] index to -> [grad_1nu,gfeat_1nu,update_1nu]
    if 'DM' not in l2o_net.name:
        layer_wise_nu_sele = [np.random.choice(x,10,replace=False) for x in l2o_net.layer_wise_n_neuron]
    # print(f'\n{l2o_net.layer_wise_n_neuron}\n{layer_wise_nu_sele}\n\n')
    # raise

    l2=timer()
    l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13=[],[],[],[],[],[],[],[],[],[],[]
    for i_step in iterator:
        if kwargs.get('random_scale'):
            rd_scale_bound = kwargs['random_scale']
            scaled_params = {}
            for name, p in optimizee.named_parameters():
                scaled_params[name] = p*torch.exp((2*torch.rand(p.shape, device=DEVICE)-1.)*rd_scale_bound)
            optimizee.load_state_dict(scaled_params)

        l3.append(timer())
        # cal loss
        loss = optimizee(target)
        all_losses = loss if all_losses is None else all_losses + loss

        l4.append(timer())
        losses_1epoch.append(loss.data.cpu().numpy())

        l5.append(timer())
        loss.backward(retain_graph=should_train)
        

        l6.append(timer())
        offset = 0
        result_params = {}
        hidden_states2 = [Variable(torch.zeros(n_params, l2o_net.hidden_layers_zer[i])).to(DEVICE) for i in range(l2o_net.n_hidden)]
        cell_states2 = [Variable(torch.zeros(n_params, l2o_net.hidden_layers_zer[i])).to(DEVICE) for i in range(l2o_net.n_hidden)]


        if lum: group_lr = lum(sets)
        
        sr_nf = []

        l7.append(timer())
        for i_group, (name, p) in enumerate(optimizee.named_parameters()):
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get gradients from the rest

            gradients = detach_var(p.grad.view(cur_sz, 1))   
            updates, new_hidden, new_cell = l2o_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states],i_group,float(i_step)
            )

            if want_sr and 'DM' not in l2o_net.name:
                i_nu = layer_wise_nu_sele[i_group]
                grad_1nu = gradients[i_nu,:].detach().cpu().data.numpy() # [10,1]
                update_1nu = updates[i_nu,:].detach().cpu().data.numpy() # [10,f]
                if len(l2o_net.last_g_features): # RP-model
                    gfeat_1nu = l2o_net.last_g_features[i_nu,:]
                    sr_nf.append(np.concatenate([grad_1nu,gfeat_1nu,update_1nu],axis=1))
                else: # DM-model
                    sr_nf.append(np.concatenate([grad_1nu,update_1nu],axis=1))


            updates = updates*group_lr[i_group] if lum else updates*0.01

            for j in range(len(new_hidden)):
                hidden_states2[j][offset:offset+cur_sz] = new_hidden[j]
                cell_states2[j][offset:offset+cur_sz] = new_cell[j]
            result_params[name] = p + updates.view(*p.size())
            result_params[name].retain_grad()
            
            offset += cur_sz
        l8.append(timer())

        sr_t_n_f.append(sr_nf)

        if i_step % unroll == 0:
            l9.append(timer())
            if should_train:
                meta_opt.zero_grad()
                all_losses += optimizee(target)
                all_losses.backward()
                # print('dvc?? ',gradients.device, p.device, loss.device, all_losses.device, hidden_states[0].device, hidden_states2[-1].device)
                meta_opt.step()

            l10.append(timer())
            all_losses = None
            optimizee = Target_Optimizee().to(DEVICE)
            wIni(optimizee)
            optimizee.load_state_dict(result_params)
            # updateDict_skip_running(optimizee,result_params)

            optimizee.zero_grad()

            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            l11.append(timer())
            
        else:
            l12.append(timer())
            for name, p in optimizee.named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
            l13.append(timer())

    # def mean_time_from(lxname,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13):
    #     lx = eval(lxname)
    #     ly = eval(lxname[0]+str(int(lxname[1:])+1))
    #     return np.mean(np.array(ly)-np.array(lx))
    # times = []
    # for i in [1,  3,4,5,6,7,  9,10,  12]:
    #     times.append(mean_time_from('l'+str(i),l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13))
    # print(f'\ntime is:\n  {times/np.min(times)}')


    # sr_t_n_f = np.concatenate(sr_t_n_f,axis=1)
    sr_t_n_f = np.asarray(sr_t_n_f)
    if sr_t_n_f.shape[1]!=0:
        T,nlayer,nsele,f = sr_t_n_f.shape
        sr_t_n_f = sr_t_n_f.reshape([T,nlayer*nsele,f])
    return losses_1epoch, sr_t_n_f


def eva_l2o_optimizer(args, l2o_net=None, eva_epoch_len=None, n_tests=None, want_save_eva_loss=None, **kwargs):
    
    argsPrinter(args,**args)
    res = [run_epoch(args, eva_epoch_len, should_train=False, **args) for _ in tqdm(range(n_tests))]
    losses_1epoch_Nrun, sr_r_t_n_f = zip(*res)
    # sr_t_n_f shape: [run_epoch_len, N_optimizee_param_groups, 1+l2o_net.Ngf+1]
    sr_r_t_n_f = np.asarray(sr_r_t_n_f) # [n_tests, sr_t_n_f]


    all_losses = np.asarray(losses_1epoch_Nrun)
    avg_eva_sum_loss = np.mean(np.sum(all_losses,1),0)
    avg_eva_last_loss = np.mean(all_losses[:,-1],0)

    figDir, recDir = wzRec(all_losses, f'Eva-loss', args['task_desc'], want_save=want_save_eva_loss)
    print(f'\n=======================\nEva result figure saved to:\n< {figDir} >\n=======================\n')
    if want_save_eva_loss:
        print(f'All loss/acc records saved to:\n< {recDir} >\n=======================\n')

    return figDir, avg_eva_sum_loss, sr_r_t_n_f


def wzRec(datas, ttl='', task_desc='newTask', want_save=False):

    if want_save: 
        os.makedirs(f'wIns/Recs',exist_ok=1)
        recDir = f'wIns/Recs/{task_desc}.npy'
        np.save(recDir, datas)
    else:
        recDir = None

    datas = np.asarray(datas)
    plt.close('all')
    plt.figure()
    if len(datas.shape)==1:
        min_v = min(datas)
        plt.plot(datas)
        plt.title(ttl+f', min = {min_v:5.4f}\n'+task_desc)
        plt.xlabel('step')
    elif len(datas.shape)==2:
        min_s = np.min(datas, axis=1)
        mean_min = np.mean(min_s)
        std_min = np.std(min_s)
        min_str = f'(avg={mean_min:5.4f},std={std_min:5.4f})'
        plot_ci(datas, ttl=ttl+f', min = {min_str}\n'+task_desc, xlb='step')
    else:
        raise ValueError('dim should be 1D or 2D')

    lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    figDir = f'wIns/{ttl}_{task_desc}_{lt2}.jpg'
    plt.savefig(figDir)
    plt.savefig('___wIns___.pdf',bbox_inches='tight')
    plt.show()

    return figDir, recDir




class RPOptimizer(nn.Module):
    def __init__(self, preproc=[ [6,] , [20,] ][0], hidden_layers_zer=[[7,],[20,20,]][0], grad_features='mt+gt', beta1=0.95, beta2=0.95, pre_tahn_scale=False, magic=None, **kwargs):
        super().__init__()
        self.hidden_layers_zer = hidden_layers_zer.copy()
        self.n_hidden = len(self.hidden_layers_zer)
        # self.use_preproc = bool(preproc)
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)
        
        self.grad_features = grad_features.split('+')
        self.Ngf = Ngf = len(self.grad_features)


        # build preproc MLP
        if magic is None:
            self.name = f'RP_#{Ngf}__{[preproc]+[hidden_layers_zer]}'
            self.magic = None
            print(f'\nRP: grad_features is: {grad_features}\n  Not Using Magic.\n')
        else:
            self.magic = [ [ 2, 2.5 , 3 ] , [ 2 , 3 , 4 ] ][magic]
            self.name = f'RP_magic${max(self.magic)}__{[preproc]+[hidden_layers_zer]}'
            Ngf = len(self.magic)
            print(f'\nMagic is: {self.magic} \n')
        
        self.Ngf = Ngf



        preproc = [Ngf] + preproc
        nn_list = []
        for i in range(len(preproc)-1):
            nn_list.extend([nn.Linear(preproc[i], preproc[i+1]), nn.ELU(), ])
        self.preprocNet = nn.Sequential(*nn_list)
        
        # build RNN cells
        hidden_layers_zer = [ preproc[-1] ] + hidden_layers_zer
        # self.recurs = []
        self.recurs = nn.ModuleList([])
        for i in range(len(hidden_layers_zer)-1):
            rnn = nn.LSTMCell(hidden_layers_zer[i], hidden_layers_zer[i+1])
            self.recurs.append(rnn)
            # exec('self.recurs_{}=rnn'.format(i))

        # build output proc MLP
        self.output = nn.Linear(hidden_layers_zer[-1], 1)
        if pre_tahn_scale: 
            self.pre_tahn_scale = torch.tensor(100.).to(DEVICE)
            self.pre_tahn_scale.requires_grad = True
        else: self.pre_tahn_scale = None


    def reset(self, params):
        if self.magic is None:
            self.mt = []
            self.vt = []
            self.layer_wise_n_neuron = []

            if 'mom5' in self.grad_features: self.mom5 = []
            if 'mom9' in self.grad_features: self.mom9 = []
            if 'mom99' in self.grad_features: self.mom99 = []

            n_params = 0
            sets = []
            for i, p in enumerate(params):
                N_this = np.prod(p.size())
                self.layer_wise_n_neuron.append(N_this)
                n_params += int(N_this)
                self.mt.append( torch.zeros(N_this, 1).to(DEVICE) )
                self.vt.append( torch.zeros(N_this, 1).to(DEVICE) )
                if 'mom5' in self.grad_features: self.mom5.append( torch.zeros(N_this, 1).to(DEVICE) )
                if 'mom9' in self.grad_features: self.mom9.append( torch.zeros(N_this, 1).to(DEVICE) )
                if 'mom99' in self.grad_features: self.mom99.append( torch.zeros(N_this, 1).to(DEVICE) )
                sets.append(i)
            return n_params, sets

        else:
            self.mt = []
            self.vt = [[] for _ in range(len(self.magic))]
            self.layer_wise_n_neuron = []
            n_params = 0
            sets = []
            for i, p in enumerate(params):
                N_this = np.prod(p.size())
                self.layer_wise_n_neuron.append(N_this)
                n_params += int(N_this)
                self.mt.append( torch.zeros(N_this, 1).to(DEVICE) )
                for j in range(len(self.magic)):
                    self.vt[j].append( torch.zeros(N_this, 1).to(DEVICE) )
                sets.append(i)
            return n_params, sets

    def grad2features(self, grads, idx, step):
        '''
        Adam = <feature name> mt = mt_tilde = mt_hat / <feature name> vs
               gt = gt_tilde = grads / vs
        self.mt = ms = momentum(grads, self.beta1)
        self.mt = ms = momentum(grads, self.beta1)
        
        mt_hat = normalized self.mt  
        
        
        '''
        
        if self.magic is None: # use mt,vt to process grad
            # grads shape: [cur_sz, 1]
            # step dtype is float
            self.mt[idx] = self.beta1*self.mt[idx] + (1.0-self.beta1)*grads
            self.vt[idx] = self.beta2*self.vt[idx] + (1.0-self.beta2)*grads**2
            mt_hat = self.mt[idx]/(1-torch.pow(self.beta1, step))
            vt_hat = self.vt[idx]/(1-torch.pow(self.beta2, step))
            vs = torch.sqrt(vt_hat)+1e-12
            mt_tilde = mt_hat / vs
            gt_tilde = grads / vs
            if 'mom5' in self.grad_features: # mom
                self.mom5[idx] = 0.5 * self.mom5[idx] + 0.5 * grads
            if 'mom9' in self.grad_features: # 
                self.mom9[idx] = 0.9 * self.mom9[idx] + 0.1 * grads
            if 'mom99' in self.grad_features: # 
                self.mom99[idx] = 0.99 * self.mom99[idx] + 0.01 * grads


            tensors_to_cat = []
            if 'mt' in self.grad_features:
                tensors_to_cat.append(mt_tilde)
            if 'gt' in self.grad_features:
                tensors_to_cat.append(gt_tilde)
            


            if 'AdamUp_unNorm' in self.grad_features:
                tensors_to_cat.append(self.mt[idx])
            if 'AdamUp_Norm' in self.grad_features:
                tensors_to_cat.append(mt_hat)
            if 'AdamDn_Norm' in self.grad_features:
                tensors_to_cat.append(vs)
            
            
            
            
            if 'g' in self.grad_features:
                tensors_to_cat.append(grads)
            if 'mom5' in self.grad_features:
                tensors_to_cat.append(self.mom5[idx])
            if 'mom9' in self.grad_features:
                tensors_to_cat.append(self.mom9[idx])
            if 'mom99' in self.grad_features:
                tensors_to_cat.append(self.mom99[idx])
            if 't' in self.grad_features:
                tensors_to_cat.append(torch.tensor(step/1000,device=DEVICE).repeat(*grads.shape))

            res = detach_var(torch.cat(tensors_to_cat, 1))
            return res


            # num_feat  2,      3,      4,     5
            # index    0,1      2,      3,     4
            #          orig,  +grad,  +mom5,  +t
            # adam = index_0
            # grad = index_2
            # mom5 = index_3


            # Ngf changed -> grad_features = ['mt', 'gt', 'g', 'mom5', 'mom9', 'mom99', 't']

        else:
            self.mt[idx] = self.beta1*self.mt[idx] + (1.0-self.beta1)*grads
            mt_hat = self.mt[idx]/(1-torch.pow(self.beta1, step))

            mt_tilde = []
            # vt_hat = []
            for j in range(len(self.magic)):
                self.vt[j][idx] = self.beta2*self.vt[j][idx] + (1.0-self.beta2)*torch.abs(grads)**self.magic[j]
                vt_hat = self.vt[j][idx]/(1-torch.pow(self.beta2, step))
                mt_tilde.append( self.mt[idx] / (torch.pow(vt_hat,1/self.magic[j])+1e-8 ) )
            return detach_var(torch.cat(mt_tilde, 1))



    def forward(self, grads, hidden, cell, idx, step):
        # grads shape: [cur_sz, 1]
        # hidden shape: len(self.hidden_layers_zer) x [cur_sz, self.hidden_layers_zer[i]]

        # preprocessing
        res = self.grad2features(grads, idx, step)
        self.last_g_features = res.detach().cpu().data.numpy()
        res = self.preprocNet(res)

        # forward propagation
        new_hidden, new_cell = [], []
        for ilayer in range(self.n_hidden):
            res, cel = self.recurs[ilayer](res, (hidden[ilayer], cell[ilayer]))
            new_hidden.append(res)
            new_cell.append(cel)
        if self.pre_tahn_scale:
            res = torch.tanh(self.pre_tahn_scale * self.output(res))*0.1
        else:
            res = torch.tanh(self.output(res))
        return res, new_hidden, new_cell


# =========================================
# ================ Utils ==================
# =========================================



def wIni(net):
    dic = net.state_dict()
    dic2 = {}
    std = 0.01
    for k,v in dic.items():
        dic2[k] = torch.randn(*v.shape, device=DEVICE)*std
    net.load_state_dict(dic2)

def detach_var(v):
    var = Variable(v.data, requires_grad=True).to(DEVICE)
    var.retain_grad()
    return var

def new_detached_var(v_np):
    var = Variable(v_np, requires_grad=True).to(DEVICE)
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))




def updateDict_skip_running(net,dic):
    newDic = {}
    for k,v in net.state_dict().items():
        if 'running' in k:
            newDic[k]=v
    dic.update(newDic)
    net.load_state_dict(dic)
    return




def getMLP(neurons, activation=nn.ReLU, bias=True):
    # neurons: all n+1 dims from input to output
    # len(neurons) = n+1
    # num of params layers = n
    # num of activations = n-1
    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), activation(), ])
    nn_list.append(nn.Linear(neurons[n-1], neurons[n], bias=bias))
    return nn.Sequential(*nn_list)



def load_model(net, cwd, verbose=True):
    def load_torch_file(network, cwd):
        network_dict = torch.load(cwd, map_location=lambda storage, loc: storage)
        network.load_state_dict(network_dict)
    if os.path.exists(cwd):
        load_torch_file(net, cwd)
        if verbose: print("\nLOAD success! from :", cwd)
    else:
        if verbose: print("\n\n\n  !!! FileNotFound when load_model: {}".format(cwd))

def save_model(net, cwd):  # 2020-05-20
    torch.save(net.state_dict(), cwd)
    print("\nSaved @ :", cwd)



def plot_ci(arr, vx=[], is_std=True, ttl='', xlb='',ylb='',semilogy=False, viz_un_log=False):
    arr = np.asarray(arr)
    if len(arr.shape)==1:  arr = arr.reshape(1,-1)
    rdcolor = plt.get_cmap('viridis')(np.random.rand())  # random color

    mean = np.mean(arr,axis=0)
    if is_std:
        ci = np.std(arr,axis=0)
        lowci = mean-ci*is_std
        hici = mean+ci*is_std
    else:
        lowci = np.min(arr,axis=0)
        hici = np.max(arr,axis=0)
    # plt.plot(mean, color = '#539caf')
    if viz_un_log:
        mean=np.exp(mean)
        lowci=np.exp(lowci)
        hici=np.exp(hici)
    if vx == []:
        vx_=np.arange(len(mean))
    if semilogy:
        plt.semilogy(vx_, mean, color = rdcolor)
    else:
        plt.plot(vx_, mean, color = rdcolor)
    plt.fill_between(vx_, lowci, hici, color = rdcolor, alpha = 0.4)
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    if list(vx): plt.xticks(vx)
    plt.title(ttl)
    return



