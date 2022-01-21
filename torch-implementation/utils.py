# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline


import torch
import sys
from meta import *



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__WELL_TRAINED__ = [



    ['wz_saved_models/! RP_[ #2, [6], [7]]~on~[30, 20]-mar22.pth', 
    {'Ngf': 2,
     'preproc':[6,],
     'hidden_layers_zer': [7,]
     },RPOptimizer,
     '#0, the original rp only wish LSTM cut-to-small'
     ],






    # ====== 2 ======
    ['wz_saved_models/! RP_[ #4, [9], [12]]~on~MLP_MNIST__len800-mar24.pth',
    {'Ngf': 4,
     'preproc':[9,],
     'hidden_layers_zer': [12,]
     },RPOptimizer,
     'used for SR for mom5,adam',
     ],





    ['wz_saved_models/! RP_[ #5, [10], [12]]~on~MLP_MNIST-mar23.pth',
    {'Ngf': 5,
     'preproc':[10,],
     'hidden_layers_zer': [12,]
     },RPOptimizer,
     '#3, time is a feature of RP input, not sure worse or better.',
     ],



    ['wz_saved_models/! RP_#2__[[6], [7]]~on~MLP_MNIST_long_epoch_back_to_short-mar24.pth',
    {'Ngf': 2,
     'preproc':[6,],
     'hidden_layers_zer': [7,]
     },RPOptimizer,
     '',
     ],


    ['wz_saved_models/RP_#2__[[6], [7]]~on~SMALL_CNN.pth', 
    {'Ngf': 2,
     'preproc':[6,],
     'hidden_layers_zer': [7,]
     },RPOptimizer,
     '#0, the original rp only wish LSTM cut-to-small',
     ],



    # ====== 1 ======
    ['wz_saved_models/! RP_#2__[[6], [7]]~on~MLP_MNIST__len800-mar24.pth', 
    {'Ngf': 2,
     'preproc':[6,],
     'hidden_layers_zer': [7,]
     },RPOptimizer,
     '#1, the variant from original rp, only trained over longer epoch (200->800)',
     ],






    # ====== -1 (ResNet) ======
    ['/wz_saved_models/! ResNet-cifar-A.pth',resnet18,
    'This is the resnet-18 trained on cifar-A using Adam(0.001) for 10w epoches'],

    ]



def do_with_pretrained(idx, args, is_zee=False):
    # cwd = __WELL_TRAINED__[idx][0]
    # config = __WELL_TRAINED__[idx][1]
    # if idx is None:
    #     print('\n\n\ndid not load pretrained!!!\n\n\n')
    #     return

    if not is_zee: # load optimizer

        if type(idx) is tuple:
            print(f'loading default L2O')
            RP_config = idx
            Ngf,preproc,hidden_layers_zer,grad_features = RP_config            

            # cwd = get_newest_file("wz_saved_models")

            cwd = 'xxxxx'
            

            config = {'Ngf':Ngf,'preproc':preproc,'hidden_layers_zer':hidden_layers_zer,'grad_features':grad_features,}
            OPT = RPOptimizer
        else:
            cwd, config, OPT, desc = __WELL_TRAINED__[idx]


        args.update(config); args['OPT'] = OPT
        l2o_net = args['OPT'](**args).to(DEVICE)
        load_model(l2o_net, cwd)
        args['l2o_net'] = l2o_net


    else: # load optimizee
        print('\n\n !!!! loading pre_zee, the problem is resnet18 !!!!')
        cwd, assigned_zee, desc = __WELL_TRAINED__[idx]

        args['problem_is_transfer_learn'] = True
        args['optimizee_load_cwd'] = cwd
        args['assigned_zee'] = assigned_zee
        args['cifarAB'] = 'B'

    return


def do_with_problem(args, Target_DataGen, Target_Optimizee, n_epochs=None, epoch_len=None, unroll=None,eva_epoch_len=None,n_tests=None,META_OPT_LR=None, **other_configs):


    args['Target_DataGen'] = Target_DataGen
    if Target_DataGen in [MNISTLoss_f, MNISTLoss]:
        args['pixels'] = 28*28
    elif Target_DataGen in [Cifar_half, Cifar_half_f]:
        args['pixels'] = 3*32*32

    args['Target_Optimizee'] = Target_Optimizee
    if args.get('l2o_net'):
        choose_meta_opt = [optim.Adam, ]; which_meta_opt = 0
        OPT_META_L2O = choose_meta_opt[which_meta_opt]
        args['meta_opt'] = OPT_META_L2O(args['l2o_net'].parameters(), lr=META_OPT_LR)

    else:
        print('\n\nL2O model not initialized before, planing to train L2O model from scratch...\n Should NOT see this if: \n\t fine-tune L2O \n\t evaluation\n\t SR Gen Data\n OK to see if:\n\t Train L2O from scratch \n\t Fitting SR\n\t fine-tune SR \n\t Evaluate SR performance \n\t normal Train resnet \n\n')
    if n_epochs is not None: args['n_epochs'] = n_epochs
    if epoch_len is not None: args['epoch_len'] = epoch_len
    if unroll is not None: args['unroll'] = unroll

    if eva_epoch_len is not None: args['eva_epoch_len'] = eva_epoch_len
    if n_tests is not None: args['n_tests'] = n_tests

    args.update(other_configs)
    return














# ============================================
# ================ Problems ==================
# ============================================



def get_infinite_iter(dataset, batch_size=1,num_workers=2,shuffle=True,sampler=None, pin_memory=None, **args):
    if sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(list(range(len(dataset))))
    
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler = sampler,
            # shuffle=True,
            num_workers=num_workers,
            persistent_workers = True,            
            pin_memory=pin_memory,
            # collate_fn=dataset.collate_fn,
        )
    
    dataloaderIter = dataloader.__iter__()
   
    class InfIter(dataloaderIter.__class__):
        def __init__(self,loader):
            super().__init__(loader)
            self.loader = loader
        def __next__(self):
            try:
                return super().__next__()
            except StopIteration:
                self._reset(self.loader)
                return super().__next__()
        def sample(self):
            data_batch, label_batch = self.__next__()
            return data_batch.to(DEVICE), label_batch.to(DEVICE)

    infIter = InfIter(dataloader)
    return infIter



class MNISTLoss:
    def __init__(self, training=True, **kwargs):
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (1.0,))])
        dataset = datasets.MNIST(
            './datasets', train=True, download=False,
            transform=trans)
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            print('\nnew sample!!!\n')
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
            print('\nnew sample ,  DONE. !!!\n')
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        # print(batch.shape)
        # raise
        return batch


class Cifar_half:
    def __init__(self, cifarAB='A', training=True, shuffle_cifar=True,num_workers=2,batch_size=128,**kwargs):

        cifar_roots  = {'A': 'datasets/cifar10-A',
                       'B': 'datasets/cifar10-B'}
        root = cifar_roots[cifarAB]
        self.cifarAB = cifarAB

        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10_halfLabel(root=root, train=bool(training), transform=transform)
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=bool(shuffle_cifar), num_workers=num_workers)

        self.batches = []
        self.cur_batch = 0


    def sample(self):
        if self.cur_batch >= len(self.batches):
            print('\nnew sample!!!\n')
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
            print('\nnew sample ,  DONE. !!!\n')
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class MNISTLoss_f:
    def __init__(self, training=True, num_workers=2, batch_size=None, **kwargs):
        trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (1.0,))])

        dataset = datasets.MNIST(
            './datasets',  train=True, download=True,
            transform=trans)


        indices = list(range(len(dataset))) # len = 60000
        np.random.RandomState(10).shuffle(indices)
        if training:
            indices = indices[:len(indices) // 2]
        else:
            indices = indices[len(indices) // 2:]

        self.infIter = get_infinite_iter(dataset,batch_size=batch_size,num_workers=num_workers,sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), **kwargs)

    def sample(self):



        samp = self.infIter.sample()
        # print(f'''\n\n\n{(
        #     )}\n\n\n''');
        # print(samp[0].shape)
        # from w import viz_imgs
        # plt.imshow(samp[0][33,0,...])
        # print(samp[0][33,0,...].shape)
        # viz_imgs(samp[0][:10,...])
        # print(samp[1])

        # raise
        return samp


class Cifar_half_f:
    def __init__(self, cifarAB='A', training=True, shuffle_cifar=True,num_workers=2,batch_size=None,**kwargs):

        cifar_roots  = {'A': 'datasets/cifar10-A',
                       'B': 'datasets/cifar10-B'}
        root = cifar_roots[cifarAB]
        self.cifarAB = cifarAB

        transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10_halfLabel(root=root, train=bool(training), transform=transform)

        self.infIter = get_infinite_iter(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=True, **kwargs)


    def sample(self):
        return self.infIter.sample()








class Cifar_f:
    def __init__(self, cifarAB='A', training=True, shuffle_cifar=True,num_workers=2,batch_size=None,pin_memory=None, **kwargs):


        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./datasets', train=True, download=True, transform=transform_train)

        self.infIter = get_infinite_iter(trainset,batch_size=batch_size,num_workers=num_workers,shuffle=True, **kwargs)



        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./datasets', train=False, download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


    def sample(self):
        return self.infIter.sample()



class MLP_MNIST(MetaModule):
    name = 'MLP_MNIST'
    def __init__(self, pixels = [28*28, 3*32*32][0], hidden_layers_zee=[[30,20,], [50,20]][0], activation=[nn.Sigmoid, nn.ReLU][1], **kwargs):
        super().__init__()
        self.hidden_layers_zee = hidden_layers_zee
        self.pixels = pixels
        
        self.layers = {}
        for i in range(len(hidden_layers_zee)):
            self.layers[f'mat_{i}'] = MetaLinear(pixels, hidden_layers_zee[i]).float()
            pixels = hidden_layers_zee[i]

        self.layers['final_mat'] = MetaLinear(pixels, 10).float()
        self.layers = nn.ModuleDict(self.layers)

        # print(f'optimizee 111 .device={self.layers.device}')


        self.activation = activation()
        self.loss = nn.NLLLoss()

    # def all_named_parameters(self):
    #     return [(k, v) for k, v in self.named_parameters()]


    def forward(self, loss):
        inp, out = loss.sample()
        # print(inp.shape)
        inp = Variable(inp.view(inp.size()[0], self.pixels)).to(DEVICE)
        out = Variable(out).to(DEVICE)

        cur_layer = 0
        while f'mat_{cur_layer}' in self.layers:
            inp = self.activation(self.layers[f'mat_{cur_layer}'](inp))
            cur_layer += 1

        inp = F.log_softmax(self.layers['final_mat'](inp), dim=1)
        l = self.loss(inp, out)
        return l


class MLP_MNIST2(MLP_MNIST):
    name = 'MLP_MNIST2'
    def __init__(self, *args, **kwargs):
        super().__init__(hidden_layers_zee=[50,20,20,12,], *args, **kwargs)

class SMALL_CNN(MetaModule):
    name = 'SMALL_CNN'
    # def __init__(self, pixels = [28*28, 3*32*32][1], hidden_layers_zee=[('conv', 3,6,3), ('conv', 6, 16, 3), ('fc', 16*5*5, 120), ('fc', 120, 84), ('fc', 84, 5),], activation=[nn.Sigmoid, nn.ReLU][1], **kwargs):
    def __init__(self, pixels = [28*28, 3*32*32][1], hidden_layers_zee=[('conv', 3,6,3), ('conv', 6, 12, 3), ('fc', 12*6*6, 5),], activation=[nn.Sigmoid, nn.ReLU][1], **kwargs):
        # used for cifar-AB, where output has 5 labels

        super().__init__()
        self.hidden_layers_zee = hidden_layers_zee
        self.N_layers = len(hidden_layers_zee)
        self.activation = activation()

        self.layers = {}
        for il in range(len(hidden_layers_zee)):
            if hidden_layers_zee[il][0]=='conv':
                self.layers[f'conv_{il}'] = MetaConv2d(*hidden_layers_zee[il][1:]).float()
            elif hidden_layers_zee[il][0]=='fc':
                self.layers[f'fc_{il}'] = MetaLinear(*hidden_layers_zee[il][1:]).float()
            else:
                raise NotImplementedError
        self.layers = nn.ModuleDict(self.layers)
        self.pool = nn.MaxPool2d(2, 2)
        self.loss = nn.NLLLoss()

    def forward(self, loss):
        x, label = loss.sample()

        x = Variable(x).to(DEVICE)
        label = Variable(label).to(DEVICE)


        for il in range(self.N_layers):
            if self.hidden_layers_zee[il][0]=='conv':
                x = self.pool(self.activation(self.layers[f'conv_{il}'](x)))
            elif self.hidden_layers_zee[il][0]=='fc':
                if self.hidden_layers_zee[il-1][0]=='conv':
                    # print(x.shape)
                    x = x.view(-1, self.hidden_layers_zee[il][1])
                x = self.layers[f'fc_{il}'](x)
                if il<self.N_layers-1:
                    x = self.activation(x)


        pred = F.log_softmax(x, dim=1)
        l = self.loss(pred, label)

        return l









# in_channels, out_channels, kernel_size, stride=1,
class CNN_official_tutorial(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





class DMOptimizer(nn.Module):
    def __init__(self, preproc=False, hidden_layers_zer=[20,20], preproc_factor=10.0,**w):
        super().__init__()
        # assert len(hidden_layers_zer)==2
        hidden_layers_zer=[20,20]
        preproc = False
        self.hidden_layers_zer = hidden_layers_zer.copy()
        self.n_hidden = len(self.hidden_layers_zer)
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_layers_zer[0]) # gf (input_size, hidden_size, bias=True)
        else:
            self.recurs = nn.LSTMCell(1, hidden_layers_zer[0])
        self.recurs2 = nn.LSTMCell(hidden_layers_zer[0], hidden_layers_zer[1])
        self.output = nn.Linear(hidden_layers_zer[1], 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)
        self.name = 'DM_{}'.format(hidden_layers_zer)

    def reset(self, params):
        self.last_g_features = []
        n_params = 0
        sets = []
        for i,p in enumerate(params):
            n_params += int(np.prod(p.size()))
            sets.append(i)
        return n_params, sets

        
    def forward(self, grads, hidden, cell, *args):
        # grads shape: [cur_sz, 1]
        # hidden shape: len(self.hidden_layers_zer) x [cur_sz, self.hidden_layers_zer[i]]
        if self.preproc:
            grads = grads.data
            inp2 = torch.zeros(grads.size()[0], 2).to(DEVICE)
            keep_grads = (torch.abs(grads) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (torch.log(torch.abs(grads[kfit_optimizereep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = torch.sign(grads[keep_grads]).squeeze()
            
            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * grads[~keep_grads]).squeeze()
            grads = Variable(inp2).to(DEVICE)
        hidden0, cell0 = self.recurs(grads, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
# hidden_sz -> hidden_layers_zer










# =========================================
# ================ Utils ==================
# =========================================


def plot(*x,**w):
    close=0
    if close:
        plt.close('all')
        plt.plot(*x,**w)
        plt.show()
    else:
        plt.figure(998)
        plt.plot(*x,**w)


        


last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
    TOTAL_BAR_LENGTH = 65.
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    # L.append('  Step: %s' % format_time(step_time))
    # L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f




def get_newest_file(file_dir='.', filter='file'):
    # os.getcwd()
    if file_dir=='.': file_dir = os.path.dirname(__file__)

    files=os.listdir(file_dir)
    files.sort(key=lambda fn: os.path.getmtime(os.path.join(file_dir,fn)) if not os.path.isdir(os.path.join(file_dir,fn)) else 0)

    newest = files[0]
    # prt(files)
    for x in files:
        if filter == 'file':
            print(os.path.isdir(x),x)
            if not os.path.isdir(x):
                newest = x
        elif filter == 'dir':
            if os.path.isdir(x):
                newest = x
        elif filter != '':

            print(x[-len(filter):]==filter, x)
            if x[-len(filter):]==filter:
                newest = x
        else:  # filter==''
            newest = x

    # newest = files[-1]
    return os.path.join(file_dir,newest)





def viz(net, ttl='', print_device=False):
    viz_ = []
    for name, p in net.named_parameters():
        if print_device:
            viz_.append((name, p.device, list(p.size())))
        else:
            viz_.append((name, list(p.size())))
    # print(f'\nparams of: {ttl}\nlength = {len(viz_)}\n{viz_}')
    print(f'\nparams of: {ttl}\nN_groups = {len(viz_)}')
    prt(viz_)
    return



def imshowcifar(img,labels=''):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # print(np.transpose(npimg, (1, 2, 0)).shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    print(f'labels are: < {labels} >')

def viz_dataset(AB):
    batch_size = 4
    root = ['datasets/cifar10-A', 'datasets/cifar10-B'][AB]
    classes= ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classesA= ('plane', 'bird', 'deer', 'frog', 'ship', )
    classesB= ('car', 'cat', 'dog', 'horse', 'truck')

    trainset = CIFAR10_halfLabel(root=root, train=False, transform= transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()


    print(images.shape)
    print(labels)
    
    lbstr = ' '.join('%5s' % [classesA,classesB][AB][labels[j]] for j in range(batch_size))
    imshowcifar(torchvision.utils.make_grid(images), lbstr)

# def set_seed(seed):
#     import random
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)


class CIFAR10_halfLabel(VisionDataset):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
    ) -> None:
        super(CIFAR10_halfLabel, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        self.data: Any = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry[b'data'])
                self.targets.extend(entry[b'labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]//2
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.data)




def wz_FO_adam(names, params, # FO is for some 'functional' in torch
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    amsgrad,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    result_params = {}
    for i, param in enumerate(params):
        name = names[i]
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr[i] / bias_correction1

        # param.addcdiv_(exp_avg, denom, value=-step_size)
        pnew = param - step_size*detach_var(exp_avg/denom)
        pnew.retain_grad()
        result_params[name] = pnew
    return result_params



            
            


def bestGPU(gpu_verbose=True, is_bestMem=True, **w):

    import GPUtil
    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil*100
        load = gpu.load*100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem if is_bestMem else bestLoad
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')
    # print(type(best))

    return int(best)


