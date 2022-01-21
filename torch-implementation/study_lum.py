# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from utils import *
from meta import *
from torch.optim import functional as FO
from utils import __WELL_TRAINED__



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')



class wSGD(Optimizer):
    def __init__(self, params, lr=1.0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, **kwargs):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(wSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    # @torch.no_grad()
    def step(self, named_params, lumv):

        # self.param_groups[0].keys() have 6 keys, including 'params' 
        # self.param_groups[0]['params'] len=num param group list
        # self.param_groups[0]['params'][i]  are tensors
        # for CNN, the shape above is [out_channel, in_channel, kernal_sz1, kernal_sz2]
        
        result_params = {}
        assert len(self.param_groups)==1

        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        
        for i, (name,p) in enumerate(named_params):
            # print(name)
            # print( self.param_groups[0]['params'][1].grad)
            if p.grad is None:
                continue
            d_p = p.grad
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
            lr = group['lr']*lumv[i] if lumv is not None else group['lr']
            # p.add_(d_p, alpha=-lr)
            # p = torch.add(p, -lr*d_p)

            # pnew = p - lr*detach_var(d_p)
            pnew = p - lr*detach_var(d_p)
            pnew.retain_grad()
            result_params[name] = pnew


        return result_params


class wAdam(Optimizer):
    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(wAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    # @torch.no_grad()
    def step(self, named_params, lumv):

        names = []
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_sums = []
        max_exp_avg_sqs = []
        state_steps = []
        group = self.param_groups[0]
        for name, p in named_params:
            if p.grad is not None:
                names.append(name)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

        beta1, beta2 = group['betas']

        if lumv is not None:
            lrs = lumv * group['lr'] # [lv * group['lr'] for it in lumv]

        else:
            lrs = [group['lr'] for _ in range(len(params_with_grad))]

        result_params = wz_FO_adam(names, params_with_grad,
               grads,
               exp_avgs,
               exp_avg_sqs,
               max_exp_avg_sqs,
               state_steps,
               group['amsgrad'],
               beta1,
               beta2,
               lrs,
               group['weight_decay'],
               group['eps']
               )
        return result_params












def train_lumv_scratch(args, LOWER_OPT=None, OPT_META_LUM=None, Target_Optimizee=None, lumv_ini=None, n_epochs=20, lr_meta=None, studyLum_want_eva_in_training=None, **kwargs):

    optimizee = Target_Optimizee(**args).to(DEVICE)

    # -- !!! DO NOT FEED **args INTO THIS !!! --
    lower_opt = LOWER_OPT(optimizee.parameters())

    N = len(list(optimizee.parameters()))
    # lumv = [torch.tensor(lumv_ini,requires_grad=True) for _ in range(N)]

    lumv = torch.tensor(np.ones(N, dtype=np.float32)*lumv_ini,requires_grad=True, device=DEVICE)
    # lumv = torch.tensor(np.ones(N)*lumv_ini,requires_grad=False)
    # aa=itertools.chain.from_iterable([[lumv]],)
    # for x in aa: print('????,',x)

    # lumv_cat = torch.cat([torch.unsqueeze(x,0) for x in lumv]).to(DEVICE)
    # lumv_cat.requires_grad=True
    meta_opt = OPT_META_LUM(itertools.chain.from_iterable([[lumv,]]), lr=lr_meta)
    # lumv_cat.requires_grad=True



    for iepoch in tqdm(range(n_epochs), 'Meta training LUM'):

        losses_1epoch = run_epoch_lum(args, lower_opt, lumv, meta_opt, **args)

        wzRec(losses_1epoch,'train-lumv-loss, epoch-{}'.format(iepoch))


        if (iepoch+1) % 100==0:
            print(f'current lum:\n     {[x.cpu().data for x in lumv]}')
            if studyLum_want_eva_in_training:
                eva_l2o_optimizer(args, **args)

    lumv_list = [x.cpu().data for x in lumv]
    lumv_list = lumv.cpu().data.numpy().tolist()
    print(f'finished, lumv = \n\t {lumv_list}')
    return lumv, lumv_list



def run_epoch_lum(args, lower_opt,lumv,meta_opt,Target_DataGen, Target_Optimizee, epoch_len, should_train=False,  unroll=1, **kwargs):

    if not should_train:
        unroll = 1

    target = Target_DataGen(**args)
    optimizee = Target_Optimizee(**args).to(DEVICE)
    # viz(optimizee,'optimizee')


    # n_params = 0
    # sets = []
    # for i, p in enumerate(params):
    #     N_this = np.prod(p.size())
    #     n_params += int(N_this)
    #     sets.append(i)



    losses_1epoch, all_losses = [], None

    if should_train:
        meta_opt.zero_grad()
        iterator = range(1, epoch_len + 1)
    else:
        iterator = tqdm(range(1, epoch_len + 1), 'Eva: ')


    for iteration in iterator:

        # cal loss
        loss = optimizee(target)
        all_losses = loss if all_losses is None else all_losses + loss

        losses_1epoch.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)
        result_params = lower_opt.step(optimizee.named_parameters(), lumv)


        if iteration % unroll == 0:

            if should_train:
                meta_opt.zero_grad()
                all_losses += optimizee(target)
                all_losses.backward()
                meta_opt.step()

            all_losses = None
            optimizee = Target_Optimizee().to(DEVICE)
            # optimizee.load_state_dict(result_params)
            updateDict_skip_running(optimizee,result_params)
                    
            optimizee.zero_grad()
        else:
            for name, p in optimizee.named_parameters():
                rsetattr(optimizee, name, result_params[name])

    return losses_1epoch





r = resnet18()
# o=optim.Adam(r.parameters(), -1)




# set_seed(12395)

# optimizee_train = MLP_MNIST
OPT = [torch.optim.SGD, torch.optim.Adam, RPOptimizer][0]

SERVER = 0
N_groups = 10

which_DataGen = 1
Target_DataGen = [MNISTLoss, Cifar_half][which_DataGen]

args = {
        # system
        'OPT': OPT,
        'LOWER_OPT': [wSGD, wAdam][1],
        'OPT_META_LUM': [optim.SGD, optim.Adam][1],
        'lumv_ini': 0.01,
        # 'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        # 'classesA': ('plane', 'bird', 'deer', 'frog', 'ship', ),
        # 'classesB': ('car', 'cat', 'dog', 'horse', 'truck'),
        'batch_size': 4,
        'num_workers': 2,


        'layers_in_blocks': {'cnn': ([3,])},



        'should_train': 1,
        'lr_meta': 2.,

        'lr_normal_train': 0.001,
        'n_epochs': [1,100][SERVER],
        'epoch_len': [5,200][SERVER],
        'unroll': [2,20][SERVER],

        'n_tests': [2,10][SERVER],
        'eva_epoch_len': 800,



        # optimizee
        'Target_DataGen': Target_DataGen,



        # new
        'studyLum_want_eva_in_training': 0,
        'want_save_eva_loss': False,

        }





AB=1
root = ['datasets/cifar10-A', 'datasets/cifar10-B'][AB]
args.update({
    'root': root, 'training': 1, 'shuffle_cifar': 1,
    })



# print(f'\nargs:\n{args}')



# =========== Training ===========
# viz_dataset(1)

# normal_train(args, root, **args)
args['Target_Optimizee']= [MLP_MNIST,MLP_MNIST2,resnet18][-1]

train_lumv_scratch(args, **args)


# =========== Evaluation ===========

# # == load model ==
# l2o_net = OPT(**args).to(DEVICE)
# cwd = WELL_TRAINED[0]

# load_model(l2o_net, cwd)
# if args['LUM']:
#     lum = LUM(args['lum_layers'])
# cwd = WELL_TRAINED[0]

#     load_model(lum, cwd)
#     args['lum'] = lum


# # == Eva model ==
# optimizee_test = MLP_MNIST2
# dic2 = {'l2o_net':l2o_net, 
#         'Target_Optimizee':optimizee_test,
#         }; args.update(dic2)
# eva_l2o_optimizer(args, **args)



