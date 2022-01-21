# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# from utils import *
# from meta import *
# from utils import __WELL_TRAINED__

# from train_rp import *
import argparse

# =========== Supported modes ===========
    # purely eva and save fig
    # SR check if converge
    # SR gen data AND save npy



# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'DEVICE={DEVICE}')

parser = argparse.ArgumentParser(description='new args to assign')

parser.add_argument('--eva_epoch_len', '-l', default=200)
parser.add_argument('--ww','--num_Xyitems_SR', '--s','-sn', type=int, default=200)






newArgs = parser.parse_args()


if __name__ == '__main__':
    
    print(newArgs)
    m={}
    m.update(**vars(newArgs))
    print(m.keys())
    print(m)

    import numpy as np
    from pysr import pysr, best

    # Dataset
    X = 2*np.random.randn(100, 5)
    y = 2*np.cos(X[:, 3]) + X[:, 0]**2 - 2

    # Learn equations
    equations = pysr(X, y, niterations=5,
        binary_operators=["plus", "mult"],
        unary_operators=[
          "cos", "exp", "sin", #Pre-defined library of operators (see https://pysr.readthedocs.io/en/latest/docs/operators/)
          "inv(x) = 1/x"]) # Define your own operator! (Julia syntax)

    ...# (you can use ctl-c to exit early)

    print(best(equations))

