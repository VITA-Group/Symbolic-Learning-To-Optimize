import numpy as np
from pysr import pysr, best, get_hof
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sympy import *
import time
import sympy
from matplotlib import pyplot as plt
import scipy
import sympy.printing as printing


from SR_data.SR_exp_rec import *
import itertools

from srUtils import *


train_size = 10000


which_SR_dataset = [None, 'RP_s','RP_s_i','RP','DM'][0]

feature_names = ['mt','g','mom5','mom9','mom99']


Xy_train, Xy_test, _, feature_names, Nfeat, N_pre = load_SR_dataset(which_SR_dataset, train_size, feature_names=feature_names)

variable_names = list(itertools.chain(*[ [ f'{v}_{N_pre-1-t}' for i, v in enumerate(feature_names)] for t in range(N_pre) ]))



print('Xy_train loaded shape: ',Xy_train.shape)
print('Xy_test loaded shape: ',Xy_test.shape)

print('variable_names: ' , variable_names)







# ==============================================
# ================= Train Data =================
# ==============================================




y_true = Xy_train[:,-1]
SR_Xy = Xy_train








# def evaR2(SR_Xy, ttl):
if Nfeat==1: # DM
    for t_ in range(N_pre):
        exec(f'g_{t_} = SR_Xy[:,{N_pre-1-t_}]')
else: # RP
    Nsample, _ = SR_Xy.shape
    SR_Xy_resp = SR_Xy[:,:-1].reshape(Nsample, N_pre, -1)
    for t_ in range(N_pre):
        try:
            exec(f'mt_{t_} = SR_Xy_resp[:,{N_pre-1-t_},0]')
            # exec(f'gv_{t_} = SR_Xy_resp[:,{N_pre-1-t_},1]')
            exec(f'g_{t_} = SR_Xy_resp[:,{N_pre-1-t_},1]')
            exec(f'mom5_{t_} = SR_Xy_resp[:,{N_pre-1-t_},2]')
            exec(f'mom9_{t_} = SR_Xy_resp[:,{N_pre-1-t_},3]')
            exec(f'mom99_{t_} = SR_Xy_resp[:,{N_pre-1-t_},4]')
            # exec(f't_{t_} = SR_Xy_resp[:,{N_pre-1-t_},4]')
        except IndexError:
            continue

# # print(g_1.shape)
# # print('!!')
# # print(dir())
# # print(locals().keys())
# # print(g_1.shape)
# raise
    







y_pred1 =  sinh((((mom9_18 + -0.83260727) * ((sinh(mt_1) + (g_0 * 1.377225)) + (mom99_15 * 3.8932905))) + ((mom9_7 + 0.02196215) + mom5_4)) + (mom5_1 + mom9_6))


r1 = r2_score(y_true, y_pred1)
print(f'\n\t Train set -> r1 = {r1}\n\n')






# y_pred2 = sinh(sinh(sinh(sinh(mv_0))))

# r2 = r2_score(y_true, y_pred2)
# print(ttl, 'r2:', r2)













# =============================================
# ================= Test Data =================
# =============================================



y_true = Xy_test[:,-1]
SR_Xy = Xy_test







# def evaR2(SR_Xy, ttl):
if Nfeat==1: # DM
    for t_ in range(N_pre):
        exec(f'g_{t_} = SR_Xy[:,{N_pre-1-t_}]')
else: # RP
    Nsample, _ = SR_Xy.shape
    SR_Xy_resp = SR_Xy[:,:-1].reshape(Nsample, N_pre, -1)
    for t_ in range(N_pre):
        try:
            exec(f'mt_{t_} = SR_Xy_resp[:,{N_pre-1-t_},0]')
            # exec(f'gv_{t_} = SR_Xy_resp[:,{N_pre-1-t_},1]')
            exec(f'g_{t_} = SR_Xy_resp[:,{N_pre-1-t_},1]')
            exec(f'mom5_{t_} = SR_Xy_resp[:,{N_pre-1-t_},2]')
            exec(f'mom9_{t_} = SR_Xy_resp[:,{N_pre-1-t_},3]')
            exec(f'mom99_{t_} = SR_Xy_resp[:,{N_pre-1-t_},4]')
            # exec(f't_{t_} = SR_Xy_resp[:,{N_pre-1-t_},4]')
        except IndexError:
            continue





y_pred1 =  sinh((((mom9_18 + -0.83260727) * ((sinh(mt_1) + (g_0 * 1.377225)) + (mom99_15 * 3.8932905))) + ((mom9_7 + 0.02196215) + mom5_4)) + (mom5_1 + mom9_6))


r1 = r2_score(y_true, y_pred1)
print(f'\n\t test -> r1 = {r1}\n\n')



# y_pred2 = sinh(sinh(sinh(sinh(mv_0))))

# r2 = r2_score(y_true, y_pred2)
# print(ttl, 'r2:', r2)


