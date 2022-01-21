

from utils import *
from meta import *

from sklearn.metrics import r2_score
from Configs import *
from SR_data.SR_exp_rec import load_SR_dataset
import torch.nn.functional as F
import sys
# from w import *












np.set_printoptions(suppress=True)  # 不用科学计数法显示
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')
os.makedirs('SR_data', exist_ok=1)
if torch.cuda.is_available(): torch.cuda.set_device(bestGPU(1))


def sr_fun_tensor(input_b_tf, expr, coeff_SR, SR_memlen, Ngf, expr_Nvar, coeff_SRNorm, **w):
    # the SR function predicts the LSTM output, which is NOT multiplied by 0.01
    # lum is never used in sr study.
    # any input/output re-scale must be done simultaneously in TWO places:
        # 1. the sr_prep2, just before outputing results.
        # 2. this function, just before outputing results.
    # I/O is a group (batch) 
        # input shape:      [batch_size, N_tf]
        # output shape:     [batch_size, 1]

    # coeff_SR: [ivar, it, order_i, (k1_order_i, k2_order_i, alpha_order_i) ]
    # coeff_SR shape: [expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len ]
    # coeff_SRNorm:     [expr_Nvar,]


    # btf: [batch_size, SR_memlen, Ngf]
    # output tensor is 2-D, shape is [batch_size, 1]

    btf = input_b_tf.reshape(-1, SR_memlen, Ngf)
    assert Ngf==expr_Nvar
    btf /= coeff_SRNorm




    def expm(x,a):
        return torch.sign(x) * abs(x)**a
    def relu(inX):
        return F.relu(inX)
    def greater(x,y):
        return (x>y).float()
    def mult(x,y):
        return x*y
    def sin(x):
        return torch.sin(x)
    def cos(x):
        return torch.sin(x)
    def sign(x):
        return torch.sign(x)
    def plus(x,y):
        return x+y
    def square(x):
        return x**2
    def cube(x):
        return x**3
    def tanh(x):
        return torch.tanh(x)
    def exp(x):
        return torch.exp(x)
    def pow(x,y):
        return torch.sign(x)*torch.pow(abs(x), y)
    def Abs(x):
        return abs(x)
    def re(x):
        return torch.real(x)
    def div(x,y):
        return x/y
    def erfc(x):
        return torch.erfc(x)
    def sqrtm(x):
        return torch.sqrt(abs(x))
    def logm(x):
        return torch.log(abs(x) + 1e-7)
    def sinh(x):
        return torch.sinh(x)
    def asinh(x):
        return torch.asinh(x)




    try:

        mt_0 = btf[:,0,0]
        mt_1 = btf[:,1,0]
        mt_2 = btf[:,2,0]
        mt_3 = btf[:,3,0]
        mt_4 = btf[:,4,0]
        mt_5 = btf[:,5,0]

    except IndexError:
        pass


    gt_0 = btf[:,0,1] 
    gt_1 = btf[:,1,1]
    gt_2 = btf[:,2,1]
    gt_3 = btf[:,3,1]
    gt_4 = btf[:,4,1]
    gt_5 = btf[:,5,1]



    g_0 = btf[:,0,2]    
    g_1 = btf[:,1,2]
    g_2 = btf[:,2,2]
    g_3 = btf[:,3,2]
    g_4 = btf[:,4,2]
    g_5 = btf[:,5,2]

    mom5_0 = btf[:,0,3]    
    mom5_1 = btf[:,1,3]
    mom5_2 = btf[:,2,3]
    mom5_3 = btf[:,3,3]
    mom5_4 = btf[:,4,3]
    mom5_5 = btf[:,5,3]


    expr = eval(expr)

    b1 = expr.reshape(-1,1)


    return b1


class History_Collector:
    def __init__(self, SR_memlen, Ngroup_zee):
        self.SR_memlen = SR_memlen
        self.Ngroup_zee = Ngroup_zee
        self.past_mems = []
    def __call__(self, gfeat, i_step, i_group):
        # gfeat shape:      [nNeu_thisLayer, Ngf]
        # output shape:     [nNeu_thisLayer, N_tf = SR_memlen * Ngf]

        assert i_step>=1
        if i_step ==1: # i_step is in range(1, run_epoch_len + 1)
            self.past_mems.append(None)
            self.past_mems[i_group] = [new_detached_var(torch.zeros(*gfeat.shape,device=DEVICE)) for _ in range(self.SR_memlen)]

        # del self.past_mems[i_group][0]
        # self.past_mems[i_group].append(gfeat)
        del self.past_mems[i_group][-1]
        self.past_mems[i_group].insert(0,gfeat)

        res = torch.cat(self.past_mems[i_group], dim=1)


        # # inspect stats
        # # MNIST problem: std:
        # #     mt ~ 0.2
        # #     gt ~ 1.0
        # #     g/mom ~ 0.0005
        # if i_step>self.SR_memlen:
        #     for i_group in range(len(self.past_mems)):
        #         ntf = torch.stack(self.past_mems[i_group], dim=1)  # shape = [nNeu_thisLayer, SR_memlen, Ngf]
        #         calStat_thisLayer(ntf)
        #     raise



        return res



def calStat_thisLayer(ntf):
    # input shape: [nNeu_thisLayer, SR_memlen, Ngf]
    F = ntf.shape[-1]
    mv = np.zeros((F,2))
    for f in range(F):
        this = ntf[:,:,f].reshape(-1)
        mv[f] = [torch.mean(this), torch.std(this)]
    print(f'mean & std at this layer (feat x m/std):\n{mv}')
    return mv
    










def eva_sr_trainingPerform(args, eva_epoch_len=None, OPT=None, n_tests=None, want_save_eva_loss=None, Target_Optimizee=None, **kwargs):
    print('TODO: add no grad to eva')
    
    args['l2oShell_4SR_featPrep'] = OPT(**args).to(DEVICE)
    args['Ngf'] = args['l2oShell_4SR_featPrep'].Ngf
    
    res = [run_epoch_sr(args, eva_epoch_len, should_train=False, **args) for _ in range(n_tests)]

    losses, coeffs = zip(*res)

    np.save('coeffs_fake_eva.npy',coeffs)

    all_losses = np.asarray(losses)
    avg_eva_sum_loss = np.mean(np.sum(all_losses,1),0)
    avg_eva_last_loss = np.mean(all_losses[:,-1],0)

    wzRec(all_losses, 'Eva-SR-loss', Target_Optimizee.name, want_save=want_save_eva_loss)

    return


def fine_tune_SR(args, coeff_SR=None, OPT=None, epoch_len_tuneSR=None, n_epochs_tuneSR=20, srFineTune_want_eva_in_train=None, OPT_META_SR=None, lr_meta_tuneSR=None, **kwargs):

    # this function will not build new SR coeffs (in-place tuning)
    # but really DOES initialize a new meta-opt
    coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
    print(f'current coeff_SR:\n{coeff_SR_list}\n{coeff_SR_list.tolist()}')


    args['opt_meta_SR'] = OPT_META_SR(itertools.chain.from_iterable([[coeff_SR,]]), lr=lr_meta_tuneSR, momentum=0.9)
    args['l2oShell_4SR_featPrep'] = OPT(**args).to(DEVICE)
    args['Ngf'] = args['l2oShell_4SR_featPrep'].Ngf
    coeffs_allEP = []
    

    # for iepoch in tqdm(range(n_epochs_tuneSR), 'Fine tuning SR'):
    for iepoch in range(n_epochs_tuneSR):

        losses_1epoch, coeffs = run_epoch_sr(args, epoch_len_tuneSR, should_train=True, **args)
        coeffs_allEP.append(coeffs)

        wzRec(losses_1epoch,'tuneSR-train-loss, epoch-{}'.format(iepoch))

        coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
        print(f'current coeff_SR:\n{coeff_SR_list}\n{coeff_SR_list.tolist()}')
        np.save('coeff_SR.npy',coeff_SR_list)
        np.save('coeffs_allEP.npy',coeffs_allEP)


        if (iepoch) % 100==0:
            if srFineTune_want_eva_in_train:
                eva_sr_trainingPerform(args, **args)


    coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
    print(f'final coeff_SR:\n{coeff_SR_list}\n{coeff_SR_list.tolist()}')
    eva_sr_trainingPerform(args, **args)
    return coeff_SR, coeff_SR_list




def run_epoch_sr(args, run_epoch_len, should_train, rec_SR = True, coeff_SR=None, opt_meta_SR=None, l2oShell_4SR_featPrep=None, Target_DataGen=None, Target_Optimizee=None, unroll_tuneSR=None, SR_memlen=None, Ngroup_zee=None,expr=None,Ngf=None, **kwargs):

    if not should_train:
        unroll_tuneSR = 1
    if bool(rec_SR):
        args['rec_SR'] = []
        coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
        args['rec_SR'].append(coeff_SR_list) 
        np.save('coeffs_train_1EP.npy',args['rec_SR'])


    target = Target_DataGen(training=should_train,**args)
    if type(target) is Cifar_f:
        round1 = 391
    else:
        raise NotImplementedError
    
    optimizee = Target_Optimizee(**args).to(DEVICE)
    wIni(optimizee)

    l2oShell_4SR_featPrep.reset(optimizee.parameters())

    losses_1epoch, all_losses = [], None
    if should_train:
        opt_meta_SR.zero_grad()
        iterator = range(1, run_epoch_len + 1)
    else:
        iterator = tqdm(range(1, run_epoch_len + 1), 'Eva: ')
    iterator = tqdm(range(1, run_epoch_len + 1))

    history_collector = History_Collector(SR_memlen, Ngroup_zee)
    for i_step in iterator:


        loss = optimizee(target)
        all_losses = loss if all_losses is None else all_losses + loss

        losses_1epoch.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)
        
        offset = 0
        result_params = {}

        for i_group, (name, p) in enumerate(optimizee.named_parameters()):
            cur_sz = int(np.prod(p.size()))

            gradients = detach_var(p.grad.view(cur_sz, 1))
            sr_features_cur = l2oShell_4SR_featPrep.grad2features(gradients, i_group, i_step)
            sr_inputs = history_collector(sr_features_cur, i_step, i_group)
            updates = sr_fun_tensor(sr_inputs, **args)

            updates = updates*0.01

            result_params[name] = p + updates.view(*p.size())
            result_params[name].retain_grad()

            offset += cur_sz


        if i_step % unroll_tuneSR == 0:
            if should_train:


                opt_meta_SR.zero_grad()
                all_losses += optimizee(target)
                all_losses.backward()
                opt_meta_SR.step()



                if bool(rec_SR): 
                    coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
                    args['rec_SR'].append(coeff_SR_list) 
                    np.save('coeffs_train_1EP.npy',args['rec_SR'])

                coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
                print(f'current coeff_SR:\n{coeff_SR_list}\n{coeff_SR_list.tolist()}')

        
            all_losses = None
            optimizee = Target_Optimizee().to(DEVICE)
            wIni(optimizee)
            optimizee.load_state_dict(result_params)
            # updateDict_skip_running(optimizee,result_params)

            optimizee.zero_grad()
        else:
            for name, p in optimizee.named_parameters():
                rsetattr(optimizee, name, result_params[name])

        if i_step%round1==0:
            acc = optimizee.cal_test_acc(target.testloater)

    return losses_1epoch, args['rec_SR']




class SRDataset(torch.utils.data.Dataset):
    def __init__(self, which_SR_dataset):
        import numpy as np
        # data = np.load(SR_dataset_fname)
        Xy_train, Xy_test, Xy_fit, feature_names, l2o_num_grad_features, N_pre = load_SR_dataset(which_SR_dataset)
        self.X = torch.tensor(Xy_fit, device=DEVICE)
        self.N = len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        return x[:-1], x[-1:]

    def __len__(self):
        return self.N





def fit_SR(args, lr_fitSR=None,expr=None,n_epochs_fit_SR=20,OPT_fitSR=None, l2o_net=None, which_SR_dataset=None,SR_memlen=None,Ngf=None,coeff_SR=None, **kwargs):
    # need to un-comment 'step-0-prep' in order to have correct l2o_net name

    coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
    print(f'finished, coeff_SR = \n\t {coeff_SR_list}\n{coeff_SR_list.tolist()}')


    opt_fitSR = OPT_fitSR(itertools.chain.from_iterable([[coeff_SR,]]), lr=lr_fitSR)

    mse = nn.MSELoss()

    dataset_train = SRDataset(which_SR_dataset)
    srIter = get_infinite_iter(dataset_train, shuffle=True,sampler=None, **args)


    eva_r2s = []
    for iepoch in range(n_epochs_fit_SR):

        l2o_input_slices, l2o_output = srIter.sample() # parallel working.

        sr_pred = sr_fun_tensor(l2o_input_slices,expr,coeff_SR,SR_memlen,Ngf)

        loss = mse(sr_pred, l2o_output)


        opt_fitSR.zero_grad()
        loss.backward()
        opt_fitSR.step()




        if (iepoch) % 100==0:
            eva_r2 = evaSR_R2(args, **args)
            coeff_disp = [f'{x.detach().cpu().data.item():.4f}' for x in coeff_SR]

            # print(f'current coeff_SR: {coeff_disp}')
            eva_r2s.append(eva_r2)
            progress_bar(iepoch, n_epochs_fit_SR, f'loss={loss} , eva_r2={eva_r2} , coeff_SR={coeff_disp}')

    wzRec(eva_r2s,'fit SR, eva_r2')

    coeff_SR_list = coeff_SR.detach().cpu().data.numpy().round(5)
    print(f'finished, coeff_SR = \n\t {coeff_SR_list}\n{coeff_SR_list.tolist()}')
    np.save('wz_saved_models/fitted SR ~ of ~ {l2o_net.name}.npy', coeff_SR_list)
    return coeff_SR, coeff_SR_list



@torch.no_grad()
def evaSR_R2(args,expr=None,coeff_SR=None, which_SR_dataset=None,SR_memlen=None,Ngf=None, **w):
    # dataset_test = np.load(which_SR_dataset)
    Xy_train, Xy_test, Xy_fit, feature_names, l2o_num_grad_features, N_pre = load_SR_dataset(which_SR_dataset)
    x = torch.tensor(Xy_test[:,:-1], device=DEVICE)
    y_true = Xy_test[:,-1]
    pred_tensor = sr_fun_tensor(x, expr, coeff_SR, SR_memlen, Ngf)
    pred = pred_tensor.detach().cpu().data.numpy()
    r2 = r2_score(y_true, pred)
    
    print(f'\n\nR2 = {r2:.5f}\n\n')

    return r2





def viz_grad_seq(sr_r_t_n_f):

    # [run, time, neuron/layer, feat]
    n_neuron = sr_r_t_n_f.shape[1]
    r,t,n,f = sr_r_t_n_f.shape
    layerIidx = [np.random.choice(n,1), (0,1,-2,-1), list(range(n)) ] [-1] 

    to_viz_r_t_n_f = sr_r_t_n_f[:,:,layerIidx,:]
    grad_rt_n = to_viz_r_t_n_f[:,:,:,0].reshape([r*t,len(layerIidx)])
    update_rt_n = to_viz_r_t_n_f[:,:,:,-1].reshape([r*t,len(layerIidx)])
    gfeat_rt_n_f = to_viz_r_t_n_f[:,:,:,1:-1].reshape([r*t,len(layerIidx),f-2])
    if f==2:
        print('viz for DM-opt')
    elif f>2:
        print('viz for RP-opt')
    else: raise ValueError
    plt.close('all')
    for ilayer in range(n):
        plt.subplot(2,n//2,ilayer+1)
        plt.plot(grad_rt_n[:,ilayer], '-x', label=f'grad @ layer {ilayer}') # len(layerIidx) 条数据
        plt.plot(update_rt_n[:,ilayer], '-o', label=f'update @ layer {ilayer}')
        def plot_feat():
            for i_f in range(f-2):
                feati_rt_n = gfeat_rt_n_f[:,:,i_f]
                plt.plot(feati_rt_n[:,ilayer], '-1', label=f'the {i_f}-th gfeat @ layer {ilayer}')
        # if f>2:  plot_feat()
        plt.legend()



def sr_prep1(args, SR_memlen, num_Xyitems_SR, eva_epoch_len=None, **kwargs):

    Ngroup_zee = args['Ngroup_zee'] = len(list(args['Target_Optimizee']().parameters()))

    tSample_1epoch = eva_epoch_len//SR_memlen-2 
    assert tSample_1epoch>0, f'eva_epoch_len={eva_epoch_len}, should > SR_memlen*2 = {SR_memlen}*2 = {SR_memlen*2}'

    what_we_get_1epoch = tSample_1epoch*Ngroup_zee

    epochs_needed = int( num_Xyitems_SR / what_we_get_1epoch /10 * 1.6 )+1
    args['n_tests'] = epochs_needed
    args['tSample_1epoch'] = tSample_1epoch
    args['available_Interval'] = eva_epoch_len - tSample_1epoch*SR_memlen-2 # -2 is for safety
    args['SR_memlen'] = SR_memlen
    args['num_Xyitems_SR'] = num_Xyitems_SR
    print(f'\n\n----------\nSR prep:\n\tepochs_needed = {epochs_needed}\n\ttSample_1epoch = {tSample_1epoch}\n\tSR_memlen = {SR_memlen}\n\teva_epoch_len = {eva_epoch_len} > tSample_1epoch*SR_memlen = {tSample_1epoch}*{SR_memlen} = {tSample_1epoch*SR_memlen}\n\tNgroup_zee = {Ngroup_zee}\n\nTarget_DataGen = {Target_DataGen}\nTarget_Optimizee = {Target_Optimizee}\n----------\n')
    return


def sr_prep2(sr_r_t_n_f, args, available_Interval=None, tSample_1epoch=None, SR_memlen=None, Ngroup_zee=None, num_Xyitems_SR=None, **kwargs):

    # returned SR_Xy_rn_tf shape:
    #     [N_sample , SR_memlen * Ngf+1 ] ; the last column is label

    # sr_gfu shape: [eva_epoch_len, Ngroup_zee, 1+l2o_net.Ngf+1]

    # sr_r_t_n_f shape: 
    #     [which run (r) , which time step (t) , which neuron (n) , grad/feature/update (f) ]
    #     shape: [n_tests, eva_epoch_len, Ngroup_zee, 1+l2o_net.Ngf+1]

    r,t,n,f = sr_r_t_n_f.shape
    f_true = f-2 if f>2 else f-1 # RP or DM
    SR_XY_rtn_f = []
    tSelect_r_n = np.random.rand(r,n,tSample_1epoch+1)
    def interv_to_int(tSelect_r_n):
        # guarantee the randomness in sampling
        sum_ = np.sum(tSelect_r_n, axis=2)
        tSelect_r_n = tSelect_r_n/sum_[:,:,None]*available_Interval
        tSelect_r_n = tSelect_r_n.astype(int)
        return tSelect_r_n
    tSelect_r_n = interv_to_int(tSelect_r_n) # shape is [r,n,tSample_1epoch+1], every [_r, _n] dimension, when sumed up along axis 'tSample_1epoch+1', is no more than available_Interval

     # the [_r, _n] position means the starting t for this sample of sequence

    rn_t_f = [] # when t&f are viewed together, this one means to be collection of t_f sequences (along t axis)
    for _r in range(r):
        for _n in range(n):
                tSelect = tSelect_r_n[_r,_n] # a 1-D array of length tSample_1epoch+1, meaning the 'interval', rather than starging point.
                for ith_selection in range(tSample_1epoch):
                    starting_point = sum(tSelect[:ith_selection+1]) + SR_memlen*ith_selection
                    cur_item_t_f = sr_r_t_n_f[_r,starting_point:starting_point+SR_memlen,_n,:] # is an 2-D array
                    if cur_item_t_f[SR_memlen//4*3-1,0] != 0.0 :
                        # only append items that grad is not 0 at this ramdomly selected location
                        rn_t_f.append(cur_item_t_f)
    # after for loop, rn_t_f is a 3-D, indexed by [rn, t, f] (somehow transposed)
    rn_t_f = np.asarray(rn_t_f)
    def extract_features_labels(rn_t_f):
        if f>2: # RP model, remove original grad feature
            rn_t_f = rn_t_f[:,:,1:]
        features_rn_t_f = rn_t_f[:,:,:-1] # 3-D, remove update column
        _, N_pre, _ = features_rn_t_f.shape
        rev_idx = np.arange(N_pre)[::-1]
        features_rn_t_f_rev = features_rn_t_f[:,rev_idx,:]
        feat_reversed = features_rn_t_f_rev.reshape([-1, SR_memlen*f_true]) # 2-D array
        labels_rn_1 = rn_t_f[:,-1,-1].reshape([-1,1]) # 1-D array, size = |rn|, reshaped to column
        SR_Xy_rn_tf = np.concatenate([feat_reversed,labels_rn_1],axis=1)
        return SR_Xy_rn_tf
    SR_Xy_rn_tf = extract_features_labels(rn_t_f)
    
    # at this point, we have a 2-D array, which is exactly the expected shape
    np.random.shuffle(SR_Xy_rn_tf)

    desc = f'#samples={SR_Xy_rn_tf.shape[0]}, #memlen={SR_memlen}, #feat={f_true}'
    print(f"\n=============\nPost proc: randomly select time sub-sequences and re-formulate into Xy-samples for SR.\n\t Input shape:\t\t sr_r_t_n_f.shape = {sr_r_t_n_f.shape}\n\t Input samples:\t r*(tSample_1epoch*Ngroup_zee) = {r}*{tSample_1epoch}*{Ngroup_zee} = {r*tSample_1epoch*Ngroup_zee}\n\t SR feature+label dim:\t\t SR_memlen*L2O_features = {SR_memlen}*{f_true}={SR_memlen*f_true}\n\t which L2O:\t\t \t{args['l2o_net'].name}\n\t feature dim is: [1(grad) + {f-2} + 1(update)]\n\t Obtained SR_Xy shape:\t\t {SR_Xy_rn_tf.shape}\n\t However, desired n_sample is:\t\t {num_Xyitems_SR}\n desc:\n\t {desc}\n=============\n\n")
    return SR_Xy_rn_tf, desc




def comma_paren(expr, loc):
    # given string expr, return the position of mathing ')' and the ','
    # assume expr has fun(A,B) structure after loc
    substr = expr[loc:]
    comma_ = 0
    cnt_left = 0
    seen = False
    for i, s in enumerate(expr[loc:]):
        if s==',':
            comma_ = i
        elif s=='(':
            cnt_left += 1
            seen = True
        elif s==')':
            cnt_left -= 1

        if seen and cnt_left==0:
            paren_ = i
            break
    return loc+comma_ , loc+paren_




def find_ith(s,sub,ith):
    i = 0
    loc = s.find(sub)
    # for i in range(ith):
    while i<ith and loc!=-1:
        loc = s.find(sub,loc+1)
        i+=1
    return loc

def find_random_subs(s,sub):
    # given a string s and a sub string, if sub is a substring of s, then return the location of a randomly selected sub in s (if multiple sub in s)
    # if sub not present, return -1
    cnt = s.count(sub)
    if cnt==0:   # not present
        return -1
    elif cnt==1:
        loc = s.find(sub)
        return loc
    else:
        ith = np.random.choice(cnt)
        loc = find_ith(s,sub,ith)
        return loc




# expr.find('expm')
# expr.count('expm')

# 'abcd abcd'.replace('a','x',1)

# 'aabc abc'.count('ab')


# mutate_expr(args,**args)
def mutate_expr(args,expr=None,coeff_SR=None,**w):
    p = np.random.rand()

    if 'expm' in expr:
        probs = [0.3,1.1]
        if p<probs[0]:
            add_term(args,**args)
        elif p<probs[1]:
            expr = mutExpm(expr)
    else:
        add_term(args,**args)
    return




def mutExpm(expr):
    loc, comma, paren = find_expm_a(expr)

    before = expr[:loc]
    after = expr[paren+1:]
    part1 = expr[loc:comma+1]
    part3 = expr[paren:paren+1]  # ')'
    a = float(expr[comma+1:paren])

    def is_single(s, start, end):
        # judge if input is completely wraped in '()'
        i1,i2 = start,end
        while i1>=0:
            if s[i1]==' ': i1-=1
            elif s[i1]!='(': return False
            else: break
        while i2<len(s):
            if s[i2]==' ': i2+=1
            elif s[i2]!=')': return False
            else: break
        return True    

    single = is_single(expr, loc-1, paren+1)
    if single:
        expr = before # -wz
        return 





def find_expm_a(expr):
    # given an string expr, find the location and value of the 'a' in x**a.
    # if multiple results, return an randomly selected one
    # eg: expr = "tanh(tanh( coeff_SR[0] + sinh(expm( coeff_SR[1] * mt_0, 3))))"  then ›› a=3
    assert 'expm' in expr
    loc = find_random_subs(expr,'expm')
    comma, paren = comma_paren(expr, loc)

    # full_expm = expr[loc:paren+1]
    # before = expr[:loc]
    # part1 = expr[loc:comma+1]
    # a = float(expr[comma+1:paren])
    # part3 = expr[paren:paren+1]  # ')'
    # after = expr[paren+1:]

    # print(expr)
    # print(before)
    # print(part1)
    # print(a)
    # print(part3)
    # print(after)

    # raise
    return loc, comma, paren












which_DataGen = 0
Target_DataGen = [MNISTLoss_f, Cifar_half_f][which_DataGen]
AB=1
cifar_root  = ['datasets/cifar10-A', 'datasets/cifar10-B'][AB]

Target_Optimizee = [MLP_MNIST,MLP_MNIST2,resnet18][0]




SERVER = 2 if torch.cuda.is_available() else 0
OPT = RPOptimizer


args = {
        # Training

        'n_epochs':  [1,  2,   100][SERVER],
        'epoch_len': [5, 200,  800,][SERVER],
        'unroll':    [2,  1,   20,][SERVER],
        'random_scale': 0.,
        # 'Target_DataGen': Target_DataGen,
        'only_want_last': 1,
        'want_save_eva_loss': False,
        # Model
        'OPT': OPT,
        'preproc': [6,],
        'hidden_layers_zer': [7,],
        # 'Ngf': 2,
        'beta1': 0.95,
        'beta2': 0.95,
        'pre_tahn_scale': None,
        'LUM': [LUM, None][1],

        # 'lum_layers': [len(list(optimizee_train().parameters())), 20, 1],





        # l2oShell_4SR_featPrep
        'grad_features': 'mt+gt+g+mom5',







        # SR stage 0: Generate SR database
        'want_sr': 1,
        'eva_epoch_len': [80,  900,   500][SERVER],
        'n_tests':       [2,    2,     10][SERVER], # will be overwritten if want_sr
        # 'Target_Optimizee': Target_Optimizee,

        # 'expr': "tanh(tanh( coeff_SR[0] + sinh(cube( coeff_SR[1] * mt_0))))",
        # 'coeff_SR': torch.tensor([0.024803324, -1.9923366],device=DEVICE),

        'OPT_fitSR':   [optim.Adam, optim.SGD][1],
        'lr_fitSR': 0.001,
        'n_epochs_fit_SR': int(1e4),

        # SR: stage 3: fine tune SR
        'which_SR_dataset':  [None, 'RP_s','RP_s_i','RP','DM'][1],
        'lr_meta_tuneSR': 0.01,
        'OPT_META_SR':          [optim.Adam, optim.SGD][1],
        'epoch_len_tuneSR':     [5,   200,  800,][SERVER],
        'unroll_tuneSR':      20,
        'n_epochs_tuneSR':      [2 ,  100,   20][SERVER],
        'srFineTune_want_eva_in_train': 0,





        # train expr
        'expr_vname_list':  ['mt', 'gt', 'g', 'mom5', ],
        'expr_Nvar':        4,
        'expr_memlen':      3,
        'expr_maxOrd':      3,
        'expr_group_len':   3,  # k1,k2,alpha











        # MLP_MNIST optimizee
        'pixels':  [28*28,  3*32*32][which_DataGen],

        # cifar
        'num_workers': 2,
        'batch_size':   128,
        'cifarAB': 'A',


        # other static
        'META_OPT_LR':0.001,
        'pin_memory': False,
        'mode': None,
        'rec_SR': 1,

        
        }



def plot_sin():

    N=10
    L = N



    def trm(x,n):
        return (-1)**((n-1)//2) * 1/np.math.factorial(n)*x**n
    x = np.linspace(-L,L,100)
    ords = [2*i+1 for i in range(N)]
    sinx = sum([trm(x,i) for i in ords])
    plot(x,sinx)
    plt.title(f'highest order = {N}')




# def idx_k(coeff_SR, expr_group_len):
#     # output the index (1-D) for the k1 in coeff_SR
#     _, _, o3 = coeff_SR.shape
#     return np.arange(o3//3, )



def genExpVTO_split(vname, t, io, expr_group_len):
    # indecies: var,t,io,twin
    # order = io+1
    # coeff_SR: [ivar, it, order_i, (k1_order_i, k2_order_i, alpha_order_i) ]
    # coeff_SR shape: [expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len ]

  
    want_twin = 1
    v2i = {'mt':0, 'gt':1, 'g':2, 'mom5': 3}
    

    if want_twin:
        iv = v2i[vname]

        thisv = f'{vname}_{t}'
        res1 = f'coeff_SR[{iv},{t},{io},0] * expm({thisv}, {io+1}+coeff_SR[{iv},{t},{io},2])'
        res2 = f'coeff_SR[{iv},{t},{io},1] * expm({thisv}, {io+1}-coeff_SR[{iv},{t},{io},2])'

        return res1+ ' + ' +res2


def genExpVTO(vname, t, io, expr_group_len):
    # indecies: var,t,io,twin
    # order = io+1
    # coeff_SR: [ivar, it, order_i, (k1_order_i, k2_order_i, alpha_order_i) ]
    # coeff_SR shape: [expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len ]

  
    want_twin = 1
    v2i = {'mt':0, 'gt':1, 'g':2, 'mom5': 3}
    

    if want_twin:
        iv = v2i[vname]

        thisv = f'{vname}_{t}'
        # res1 = f'coeff_SR[{iv},{t},{io},0] * expm({thisv}, {io+1})'
        res1 = f'coeff_SR[{iv},{t},{io},0] * {thisv}'

        return res1



def genExp(expr_vname_list, expr_memlen, expr_maxOrd, expr_group_len, **w):
    res = []
    expr_vname_list = ['mt', 'gt', 'g', 'mom5', ]
    expr_vname_list = ['mt', 'gt', ]
    expr_vname_list = ['mt', 'g', ]


    expr_memlen = 2
    expr_maxOrd = 1



    for vname in expr_vname_list:
        for t in range(expr_memlen):
            for io in range(expr_maxOrd):
                res.append(genExpVTO(vname,t,io, expr_group_len))

    # res = '\ \n+'.join(res)
    # print(res); raise
    # print(res)
    # raise
    res = '+'.join(res)
    return res






def do_with_expr(args, coeff_SR_dic=None,**w):
    # coeff_SR: [ivar, it, order_i, (k1_order_i, k2_order_i, alpha_order_i) ]
    # coeff_SR shape: [expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len ]
    # norm is to devide, denorm is to multiply (var's std)

    def coeff_SR_ini(coeff_SR_dic, expr_Nvar,expr_memlen,expr_maxOrd,expr_group_len, **w):
        # std: MNIST task:
        #     mt ~ 0.2
        #     gt ~ 1.0
        #     g/mom ~ 0.0005
 



        coeff_SRNorm = np.array([0.2, 1, 0.0005, 0.0005]) * 1

        k_scalar_ini_abs = 0.001
        alpha_ini = 0.05
        assert expr_group_len==3
        coeff_SR = np.zeros((expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len))

        k_ini = -k_scalar_ini_abs**(np.arange(expr_maxOrd)+1).reshape((expr_maxOrd,1)).repeat(2,axis=1)
        coeff_SR[...,0:2] = k_ini
        coeff_SR[...,2] = alpha_ini

        if coeff_SR_dic is not None:
            for k,v in coeff_SR_dic.items():
                coeff_SR[k] = v


        return coeff_SR, coeff_SRNorm


    coeff_SR, coeff_SRNorm = coeff_SR_ini(coeff_SR_dic, **args)



    args['expr'] = genExp(**args)
    # print(genExp(**args))
    # raise

    args['coeff_SR'] = torch.tensor(coeff_SR, device = DEVICE, dtype=torch.float32)
    args['coeff_SR'].requires_grad = True
    args['coeff_SRNorm'] = torch.tensor(coeff_SRNorm, device=DEVICE, dtype=torch.float32, requires_grad=False)


    return







coeffs = np.load('coeffs_train_1EP.npy') # [expr_Nvar, expr_memlen, expr_maxOrd, expr_group_len ]

print(coeffs.shape)
# raise















if __name__ == '__main__':





    # =====================================================
    # ============== stage 2.5: Fine tune SR ==============

    print('\n\n         321321       \n\n')




    # do_with_pretrained(0, args)
    do_with_problem(args, *problem_comb['r18'], **args)


    args['Ngroup_zee'] = len(list(args['Target_Optimizee']().parameters()))
    args['SR_memlen'] = 2
    args['num_Xyitems_SR'] = 580

    args['n_epochs_tuneSR'] = 1
    args['epoch_len_tuneSR'] = 400
    args['unroll_tuneSR'] = 10
    args['lr_meta_tuneSR'] = 0.005



    coeff_SR_dic = {
                    (0,0,2,0): -1.98 * 0.2**3,
                    }

    do_with_expr(args, coeff_SR_dic, **args)
   



    fine_tune_SR(args, **args)
    
    args['expr']


    # ===================================================
    # =========== stage 3: Eva SR performance ===========



    # print('\n\n         45tyj       \n\n')

    # coeff_SR_dic = {

    #                 (0,0,2,0): -1.98 * 0.2**3 ,

    #                 }




    # args['n_tests']=2
    # args['eva_epoch_len']=8

    # do_with_expr(args, coeff_SR_dic, **args)
    # eva_sr_trainingPerform(args, **args)

















