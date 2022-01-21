

import numpy as np
from pysr import pysr, best, get_hof, best_callable
from matplotlib import pyplot as plt


def me(y):
    return np.sqrt(np.dot(y,y)/len(y))


def remove_nan(X, y):
  SR_Xy = np.concatenate([X, y.reshape(-1,1)],axis=1)
  print('before remove-nan, shape is: ',SR_Xy.shape)
  N_samples, n_features = SR_Xy.shape
  res = []
  for sp in SR_Xy:
    has_nan = 0
    for i_feat in range(n_features):
      if sp[i_feat]!=sp[i_feat]:
        has_nan=1
    if not has_nan:
      res.append(sp)
  res = np.asarray(res)
  print('AFTER remove-nan, shape is: ',res.shape)
  X = res[:,:-1]
  y = res[:,-1].reshape(-1)
  return X, y



# ----- loading, train/test split -----
SR_Xy_orig = np.load('SR_Xy.npy')
# SR_Xy = np.load('SR_Xy_layer2.npy')

# SR_Xy = np.load('subXy.npy')
print('SR_Xy original loaded shape: ',SR_Xy_orig.shape)
N = len(SR_Xy_orig)
N_pre = 20
train_test_split = int(N*0.85)
# train_test_split = N
SR_Xy = SR_Xy_orig[:train_test_split]


# ----- normalize t -----
# SR_Xy[:,N_pre] /= 1e1  # i_layer
SR_Xy[:,N_pre+3] /= 1e4  # t
# SR_Xy[:,-1] *= 10   # y_true
# SR_Xy[:,-2] /= 1e2  # k
# SR_Xy *= 30




# ----- assign abck -----
a = SR_Xy[:,-5].flatten()
b = SR_Xy[:,-4].flatten()
c = SR_Xy[:,-3].flatten()
k = SR_Xy[:,-2].flatten()



# # ----- exp: 随意指定目标SR -----
# x05 = SR_Xy[:,:11]
# my_weight=np.linspace(1,0,11).reshape(1,-1)
# mulf = (-1)**np.arange(11)
# # my_weight = my_weight*mulf
# my_weight = mulf
# my_weight[0]=2
# print(my_weight)



# ----- assign y -----
# y = np.sum(x05*my_weight,axis=1)
# y = x05[:,0]**2*3 + x05[:,2]*10 + np.exp(x05[:,4])

y = SR_Xy[:,-1]
# y = SR_Xy[:,0]*(-0.01)
# y = a
# y = np.log(-a*k)
print('y.shape',y.shape)
print('y samples: ', y)


# ----- assign X -----
# selectx = np.arange(N_pre+4).tolist()
selectx = [0,1,2,3,4,] + [N_pre+1,N_pre+2,N_pre+3]
# selectx = np.arange(N_pre).tolist() + [N_pre+1,N_pre+2,N_pre+3,]

# selectx = [N_pre+1,N_pre+2,]
# selectx = [N_pre+1,N_pre+2,N_pre+3,]
# selectx = [N_pre+1, N_pre+3, ]

X = SR_Xy[:,selectx]
print('X shape', X.shape)
print('X samples,',X)
variable_names = np.array(['g{}'.format(i) for i in range(20)]+['ilayer', 'mean_g', 'mean_g2', 't'])
variable_names = variable_names[selectx]
print('selected var-names: ', variable_names )


# ----- remove nan -----
X, y = remove_nan(X, y)
print('Variable STD:\n',np.std(SR_Xy,axis=0))

# raise






# ----- begin SR -----

equations = pysr(X, y, 
        niterations = 800,  # default = 100
        ncyclesperiteration = 900, # default=300, increases diversity
        populations = 8,    # increases diversity, maybe better if > procs=4
        # binary_operators=["plus", "mult", "pow", "greater", "div"],
        binary_operators=["plus", "mult", "pow", "greater"],
        # extra_sympy_mappings={ },
        # unary_operators=["relu", "exp", "tanh", "sinh", "asinh", "erfc", "square", "cube", "abs", "sign", "logm", "sqrtm"],
        unary_operators=["relu", "exp", "tanh", "sinh", "asinh", "erfc", "square", "cube", "abs", "sign", "sqrtm"],

        # constraints={'pow': (2, 2), },
        maxsize = 9999,  # default=20
        maxdepth = None,  # default=None
        batching = False,#   batchSize = 50,  # make evolution faster for large datasets
        variable_names = variable_names, 
        warmupMaxsize = 0,  # default=0, maxsize increases every warmupMaxsize
        ) 







bsym = best(equations)

print('best:   ',bsym)





