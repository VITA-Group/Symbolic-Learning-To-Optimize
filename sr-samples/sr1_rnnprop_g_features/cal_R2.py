

import numpy as np
from pysr import pysr, best, get_hof
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
# from sympy import *
import time
import sympy
from matplotlib import pyplot as plt
import scipy
import sympy.printing as printing







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








def surf_3d(im,ttl,ax_,ay_,):
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np
    """
    Parameters
    ----------
    im : m - n matrix
    Returns: None
    plot and show the 3d surface of input matrix
    -------
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X = np.arange(len(im[0]))
    Y = np.arange(len(im))
    X, Y = np.meshgrid(X, Y)
    # R = im2
    Z = im
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1e-4,1e-4)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(ttl)
    plt.xlabel('g1')
    plt.ylabel('t')
    ax.xticks=ax_
    ax.yticks=ay_
    plt.show()
    lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())   
    plt.savefig("./___{}.jpg".format(lt2))
    plt.close('all')

def sr_expression(x,y):

    g4=5.e-3
    t=x
    g1=y
    res = g4 + Abs(t - 1.864943)**exp(greater(0.026726894, t))*sign(t - 1.864943) + sign(greater(t, 0.80258375*Abs(g1) + 0.0172415) - 0.60028476)
    # res = g4 + Abs(t - 1.864943)**exp(greater(0.026726894, t))*sign(t - 1.864943) + sign(greater(t, 0.80258375*Abs(g1) + 0.0172415) - 0.60028476)

    return res
# ax = np.linspace(0,0.2,num=20)
# ay = np.linspace(-0.01,0.002,num=20)
# arr = np.zeros([len(ax),len(ay)])
# for ix,x in enumerate(ax):
#     for iy,y in enumerate(ay):
#         arr[ix,iy]=sr_expression(x,y)
# surf_3d(arr,'SR for A*K, grad+t',ax,ay)



# below are for calculating R2-score and more
def hasnan(arr):
    arr = arr.copy().reshape(-1)
    for x in arr:
        if x!=x:
            return True
    return False
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




















































def sr_fun_1229(g0,g1,g2,g3,g4,mean_g,mean_g2,t):
  # 输入输出都是flatten的array
  t /= 1e4
  # print('yo')
  res =  mult(pow(relu(mult(g0, -0.008472766)), plus(asinh(plus(t, erfc(0.37580237))), plus(relu(plus(plus(plus(tanh(plus(g4, plus(tanh(tanh(tanh(tanh(tanh(tanh(relu(mult(plus(g2, plus(plus(g1, g3), 0.020662013)), 3.0930731)))))))), square(g0)))), mean_g2), abs(g0)), mean_g2)), t))), 1.320138)
  
  
  
  

  return res
s=0.03
g_m = np.random.randn(6)*s
m2 = np.random.randn()*s**2
t=30/1e4
inputs=np.append(g_m,[m2,t])
y_rec = []
for t in range(800):
    inputs[-1]=t/1e4*30
    inputs[-3:-1] = abs(inputs[-3:-1])
    y=sr_fun_1229(*inputs)
    y_rec.append(y)


# print(inputs)
# print(y_rec[:10])
# plt.close('all')
# plt.plot(y_rec)

# raise








if __name__ == '__main__':
    

    # ----- loading, train/test split -----
    SR_Xy_orig = np.load('SR_Xy.npy')
    # SR_Xy = np.load('SR_Xy_layer2.npy')

    # SR_Xy = np.load('subXy.npy')
    print('SR_Xy original loaded shape: ',SR_Xy_orig.shape)
    N = len(SR_Xy_orig)
    N_pre = 20
    train_test_split = int(N*0.85)

    def evaluate(SR_Xy, ttl):
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


        # ----- assign y -----
        # y = np.sum(x05*my_weight,axis=1)
        y_true = SR_Xy[:,-1]
        # y = SR_Xy[:,0]*(-0.01)
        # y_true = a
        # y_true = np.log(-a*k)


        SR_Xy, y_true = remove_nan(SR_Xy, y_true)
        print('y_true.shape',y_true.shape)







        for i_ in range(20):
            exec('g{} = SR_Xy[:,{}]'.format(i_,i_))
        g0 = SR_Xy[:,0]
        g1 = SR_Xy[:,1]
        g2 = SR_Xy[:,2]
        g3 = SR_Xy[:,3]
        g4 = SR_Xy[:,4]
        g5 = SR_Xy[:,5]
        g6 = SR_Xy[:,6]
        g7 = SR_Xy[:,7]
        g8 = SR_Xy[:,8]
        g9 = SR_Xy[:,9]
        g10 = SR_Xy[:,10]
        g11 = SR_Xy[:,11]
        g12 = SR_Xy[:,12]
        g13 = SR_Xy[:,13]
        g14 = SR_Xy[:,14]
        g15 = SR_Xy[:,15]
        g16 = SR_Xy[:,16]
        g17 = SR_Xy[:,17]
        g18 = SR_Xy[:,18]
        g19 = SR_Xy[:,19]




        ilayer = SR_Xy[:,N_pre]
        mean_g = SR_Xy[:,N_pre+1]
        mean_g2 = SR_Xy[:,N_pre+2]
        t=SR_Xy[:,-6]
        





        # y_pred1 = (m05 - 0.30453208)*relu(grad)
        # y_pred2 = mult(pow(greater(0.28059223, square(plus(tanh(sin(grad)), m05))), t), mult(plus(mult(abs(sin(sin(sin(sin(tanh(mult(m05, greater(0.14005204, m05)))))))), plus(mult(relu(abs(mult(t, -0.036141187))), -3.3275316), cos(t))), -0.2728164), grad))

        y_pred1 = mult(cube(plus(mult(sign(cube(cube(mult(g0, 0.012507909)))), 3.7903202), pow(asinh(plus(plus(pow(g14, 0.3136837), plus(pow(plus(plus(asinh(asinh(plus(tanh(pow(plus(pow(g12, sign(sign(greater(sqrtm(g19), g9)))), asinh(plus(tanh(cube(pow(cube(plus(g5, g10)), abs(g3)))), g15))), 1.9742187)), sinh(sinh(pow(mult(g6, pow(mean_g, 2.9580286)), abs(plus(abs(plus(erfc(square(plus(square(plus(exp(g16), g11)), abs(g15)))), g1)), g6)))))))), g11), pow(plus(tanh(sinh(pow(sinh(sinh(plus(asinh(tanh(plus(plus(plus(mult(plus(plus(plus(g7, g17), g15), g3), 8.598821), g11), g16), g16))), pow(plus(mult(g9, 0.41136304), g2), 0.20316619)))), 1.8023529))), tanh(tanh(plus(pow(g19, 0.41457632), mult(plus(plus(g2, tanh(g19)), g15), -2.4503496))))), 1.1864426)), 3.176606), tanh(tanh(pow(plus(mult(plus(plus(g1, plus(g1, g8)), g4), 2.278607), g16), 0.109452926))))), 0.23052847)), plus(mult(plus(mult(tanh(sinh(mult(mean_g, 17.297403))), 1.1104674), 1.1961644), 1.0720572), plus(mean_g, g12))))), -2.884877e-5)
        
        y_pred2 =mult(cube(plus(mult(sign(cube(cube(mult(g0, 0.012507909)))), 3.7903202), pow(asinh(plus(plus(tanh(pow(plus(plus(g14, g4), g11), 0.3136837)), plus(pow(plus(plus(tanh(tanh(plus(pow(plus(tanh(pow(g12, plus(plus(greater(g14, g19), g1), plus(greater(g11, abs(g7)), 0.43955708)))), asinh(cube(pow(cube(mult(plus(g5, g10), 0.31567344)), abs(g2))))), 1.7999914), -0.23245786))), sinh(sinh(sinh(pow(mult(g6, square(mean_g)), plus(pow(abs(plus(erfc(plus(square(square(plus(square(exp(g16)), plus(g10, g13)))), g9)), mult(g1, 1.7952574))), 0.90774804), abs(g10))))))), plus(tanh(plus(cube(pow(plus(sinh(sinh(plus(tanh(tanh(plus(plus(tanh(plus(tanh(mult(plus(plus(plus(g7, g17), g15), g3), 10.189377)), g16)), g1), g11))), pow(plus(mult(g9, 0.4680273), g2), 0.23052664)))), g2), 2.1302805)), sqrtm(relu(g19)))), mult(pow(g19, 0.37028554), 0.42734128))), plus(plus(g5, 3.1130025), g19)), tanh(pow(plus(plus(mult(plus(g1, g1), 1.7044061), relu(g9)), g16), 0.14071077)))), 0.2660637)), plus(sinh(tanh(sinh(plus(mult(mean_g, 17.243078), g5)))), 1.1961644)))), -2.884877e-5)
        
        

        # y_pred3 = mult(mult(square(plus(plus(tanh(plus(mult(plus(pow(mean_g, plus(plus(mean_g2, -0.0703944), mean_g2)), -0.81851226), 0.93844116), plus(mean_g, pow(plus(mean_g, plus(plus(mult(plus(mean_g, -0.05388384), 4.249717), mean_g), mult(mean_g, 1.6308507))), 0.7676956)))), mean_g), mean_g)), square(plus(plus(plus(mean_g, mean_g), mean_g), 1.3118293))), -0.016616961)




        r1 = r2_score(y_true, y_pred1)
        r2 = r2_score(y_true, y_pred2)
        # r3 = r2_score(y_true, y_pred3)


        print(ttl, 'r1:', r1)
        print(ttl, 'r2:', r2)
        # print(ttl, 'r3:', r3)

        print(ttl, 'mse1:',mse(y_true, y_pred1))

    SR_Xy = SR_Xy_orig[:train_test_split]
    evaluate(np.array(SR_Xy), 'train -> ')
    SR_Xy = SR_Xy_orig[train_test_split:]
    evaluate(np.array(SR_Xy), 'test -> ')




