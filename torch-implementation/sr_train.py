# feature sequence is already reversed in time in extract_features_labels() in stage023.py

import numpy as np
from pysr import pysr, best, get_hof, best_callable
from matplotlib import pyplot as plt
import itertools

from SR_data.SR_exp_rec import *



print('\n\n\n\n\n             ijuhbjj           \n\n\n\n\n')


mode = ['SR for L2O', 
        'SC: mom5', 
        'sc: Adam'
        ][2]

train_size = 10000

print(f'\n\nmode is:  {mode}\n\n')




if mode=='SR for L2O':

    which_SR_dataset = [None, 'RP_s','RP_s_i','RP','DM'][1]

    # feature_names = ['mt', 'gt', 'g', 'mom5', 'mom9', 'mom99'   ]

    SR_Xy, Xy_test, Xy_fit, feature_names, Nfeat, N_pre = load_SR_dataset(which_SR_dataset, train_size, feature_names=feature_names)



    y = SR_Xy[:,-1]
    X = SR_Xy[:,:-1]

    variable_names = list(itertools.chain(*[ [ f'{v}_{t}' for i, v in enumerate(feature_names)] for t in range(N_pre) ]))
    print('variable_names: ' , variable_names)
    print('Variable STD:\n',np.std(SR_Xy,axis=0))

    equations = pysr(X, y, 
            niterations = 30,  # default = 100
            ncyclesperiteration = 9999, # default=300, increases diversity
            populations = 8,    # increases diversity, maybe better if > procs=4
            binary_operators=["plus", "mult", "pow", "greater", "div"],
            # binary_operators=["plus", "mult", "pow"],
            # extra_sympy_mappings={ },
            unary_operators=["relu", "exp", "tanh", "sinh", "asinh", "erfc", "square", "cube", "abs", "sign", "logm", "sqrtm" ],

            # maxsize = 2000,  # default=20
            maxdepth = 9999,  # default=None
            # batching = False,  #   batchSize = 50,  # make evolution faster for large datasets
            variable_names = variable_names, 
            ) 


    bsym = best(equations)
    print('best:   ',bsym)



else:    # sanity checks
    
    SR_Xy, Xy_test, Xy_fit, feature_names, Nfeat, N_pre = load_SR_dataset('RP_s_i', train_size)

    print('SR_Xy loaded shape: ',SR_Xy.shape)

    assert Nfeat==4
    # Xall dims: [N_sample, N_pre(reversed in ), ]
    Xall = SR_Xy[:,:-1].reshape([-1,N_pre,Nfeat])  # [Nsample, N_pre, Nfeat]
    Xgrad = Xall[:,:,2]  # [Nsample, N_pre]
    Xadam = Xall[:,:,0]  # [Nsample, N_pre]
    Xmom5 = Xall[:,:,3]  # [Nsample, N_pre]



    if mode=='SC: mom5':
        y = Xmom5[:,0] # already reversed; 0 is the most recent
        variable_names = np.asarray(list(itertools.chain(*[[ f'g_{t}' for t in range(N_pre) ]])))
        X = Xgrad

    elif mode=='sc: Adam':
        y = Xadam[:,0] # already reversed; 0 is the most recent
        variable_names = np.asarray(list(itertools.chain(*[[ f'g_{t}' for t in range(N_pre) ]])))
        X = Xgrad


    Nsamp, N_pre = X.shape
    
    x_select = np.arange(5)
    X = X[:,x_select]
    variable_names = variable_names[x_select]



    print('X.shape=',X.shape)
    print('len(variable_names)=',len(variable_names))
    print('variable_names: ' , variable_names)
    print('Variable STD:\n',np.std(SR_Xy,axis=0))

    print('\n\n\n\n\n\n\n   4rfg    \n\n\n\n')

    equations = pysr(X, y, 
            niterations = 10,  # default = 100
            ncyclesperiteration = 3000, # default=300, increases diversity; should set to a large number in order to discover more variable - it is important
            populations = 8,    # increases diversity, maybe better if > procs=4
            # binary_operators=["plus", "mult", "pow", "greater", "div"],
            binary_operators=["plus", "mult"],
            # extra_sympy_mappings={ },
            unary_operators=["relu", "exp", "tanh", "sinh", "asinh", "erfc", "square", "cube", "abs", "sign", "logm", "sqrtm" ],
            # unary_operators=[],

            # maxsize = 9000,  # default=20
            maxdepth = 8000,  # default=None, should set to a large number in order to discover more variable - it is important
            # batching = False,  #   batchSize = 50,  # make evolution faster for large datasets
            variable_names = variable_names, 
            ) 


    bsym = best(equations)
    print('best:   ',bsym)



