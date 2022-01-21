# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from utils import *
from meta import *
# from utils import __WELL_TRAINED__

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')
SERVER = 2
RP_config = [
              (2,   [6,],   [7,]   ),
              (4,   [9,],   [12,]  ),
              (5,   [10,],  [12,]  ),
              (5,   [9,],   [12,], 'mt+g+mom5+mom9+mom99'  ),

             ][-1]

optimizee_train = [MLP_MNIST,MLP_MNIST2,SMALL_CNN,resnet18][0]

which_DataGen=0
Target_DataGen = [MNISTLoss_f, Cifar_half_f][which_DataGen]
# Target_DataGen = [MNISTLoss, Cifar_half][which_DataGen]

AB=0
cifar_root  = ['datasets/cifar10-A', 'datasets/cifar10-B'][AB]


OPT = RPOptimizer

args = {
        # Training

        'n_epochs':       [2,   3,   200][SERVER],
        'epoch_len':      [5,  200,  800][SERVER],
        'unroll':         [2,   2,    20][SERVER],

        'n_tests':        [2,   2,    3 ][SERVER],
        'eva_epoch_len':  [8,  800,  400][SERVER],

        'random_scale': None,
        'OPT': OPT,
        'only_want_last': 1,
        'want_sr': 0,
        'trainL2O_want_eva_in_training': 0,
        'want_save_eva_loss': False,


        'pin_memory': True,
        'num_workers': 8,



        # task
        'Target_DataGen': Target_DataGen,


        # Model (below 3 will be overwritten if do_with_pretrained)
        'preproc': RP_config[1],
        'hidden_layers_zer': RP_config[2],

        'beta1': 0.95,
        'beta2': 0.95,
        'pre_tahn_scale': 0,
        'LUM': [LUM, None][1],
        # 'lum_layers': [len(list(optimizee_train().parameters())), 20, 1],
        

        # MLP_MNIST optimizee
        'pixels':  [28*28, 3*32*32][which_DataGen],

        # cifar
        # 'training':1,
        'cifarAB': 'A',
        'batch_size':128,

        # resnet adaptation
        'lr_normal_train': 0.001,
        'OPT_normal_train': optim.Adam,       

        }



problem_comb = {'mni': [MNISTLoss_f, MLP_MNIST],
                'mni2': [MNISTLoss_f, MLP_MNIST2],
                'cnn': [Cifar_half_f, SMALL_CNN],
                'r18h': [Cifar_half_f, resnet18],
                'r18': [Cifar_f, resnet18],
                'r50': [Cifar_f, resnet50],
                }



