
# feature sequence is already reversed in time in extract_features_labels() in stage023.py

import numpy as np
from pysr import pysr, best, get_hof, best_callable
from matplotlib import pyplot as plt
import itertools


def relu(inX):
    return np.maximum(0.,inX)
def greater(x,y):
    return np.greater(x,y).astype('float')
    # return np.maximum(x,y)
def mult(x,y):
    return x*y
def sin(x):
    return np.sin(x)
def cos(x):
    return np.sin(x)
def sign(x):
    return np.sign(x)
def plus(x,y):
    return x+y
def square(x):
    return x**2
def cube(x):
    return x**3
def tanh(x):
    return np.tanh(x)
def exp(x):
    return np.exp(x)
def pow(x,y):
    return np.sign(x)*np.power(abs(x), y)
def Abs(x):
    return abs(x)
def re(x):
    return np.real(x)
def div(x,y):
    return x/y
def erfc(x):
    return scipy.special.erfc(x)
def sqrtm(x):
    return np.sqrt(np.abs(x))
def logm(x):
    return np.log(np.abs(x) + 1e-8)
def sinh(x):
    return np.sinh(x)
def asinh(x):
    return np.arcsinh(x)





def srTrain(X, y=None, variable_names=None):
    if y is not None:
        y = np.asarray(y).reshape(-1)
        X = np.asarray(X)
    else:
        X = np.asarray(X)
        X, y = X[:,:-1], X[:,-1]
    print(f'\n\n===\n\t shapes:\n\t\t{X.shape}\n\t\t{y.shape}\n===')


    if variable_names is None:

        variable_names = list(itertools.chain(*[ [ f'x_{t}' ] for t in range(len(X[0])) ]))

    print('variable_names: ' , variable_names)
    # raise
    print('Variable STD:\n',np.std(X,axis=0))

    equations = pysr(X, y, 
            niterations = 300,  # default = 100
            ncyclesperiteration = 900, # default=300, increases diversity
            populations = 8,    # increases diversity, maybe better if > procs=4
            binary_operators=["plus", "mult", "pow", "greater", "div"],
            # binary_operators=["plus", "mult", "pow"],
            # extra_sympy_mappings={ },
            unary_operators=["relu", "exp", "tanh", "sinh", "asinh", "erfc", "square", "cube", "abs", "sign", "logm", "sqrtm" ],
            # unary_operators=[ "square", "cube", "abs", "sin", "cos" ],

            maxsize = 200,  # default=20
            # maxdepth = 1000,  # default=None
            # batching = False,  #   batchSize = 50,  # make evolution faster for large datasets
            variable_names = variable_names, 
            ) 

    bsym = best(equations)
    print('best:   ',bsym)
    return
