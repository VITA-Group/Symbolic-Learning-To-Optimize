

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



# str_expr = ' Abs(Abs(t*(g4 + Abs(g1)**re(t)*Abs(sign(g1)) - 0.21847524))**0.6314086*sign(t*(g4 + Abs(g1)**re(t)*Abs(sign(g1)) - 0.21847524)) - 0.34767756)'

# expr = sympy.sympify(str_expr)
# print(expr)




# ------ 简化表达式 -----
# below are ONLY FOR sympy:
# def pow(x,y):
#     return Pow(x,y)
# def plus(x,y):
#     return Add(x,y)
# def mult(x,y):
#     return Mul(x,y)
# def square(x):
#     return x**2
# def relu(x):
#     return None
# def greater(x,y):
#     return None
# def cube(x):
#     return x**3
# def div(x,y):
#     return x/y
# def sqrtm(x):
#     return sqrt(x)
# def logm(x):
#     return log(x)
# def relu(x):
#     return  tanh(x)**2





t = sympy.symbols('t')
for ig in range(80):
    exec("g{} = sympy.symbols('g{}')".format(ig,ig))
mean_g = sympy.symbols('mean_g')
mean_g2 = sympy.symbols('mean_g2')




# expr=simplify(Add(Add(Pow(Add(Add(g2, g3), Mul(Add(Add(g4, Add(g6, tanh(g7))), g5), 0.5451829)), 1.0916176), Mul(g1, 0.9318088)), g0)) #  标准范例

# expr =  erfc(plus(square(plus(plus(plus(plus(plus(plus(plus(pow(square(plus(g0, 0.031478696)), 0.4400791), 1.5804387), mult(plus(t, t), plus(abs(g0), 0.6307337))), mean_g2), mean_g2), mult(g4, plus(mult(plus(g1, g4), 2.3371828), 0.5028161))), t), plus(mult(mean_g, -1.7847556), mean_g2))), pow(pow(plus(g2, plus(g0, plus(g0, plus(mean_g2, mean_g2)))), sinh(plus(g3, 0.24738483))), 0.6236241)))



def orig_expr_2_shortened_latex(expr):
    # 输入不是字符串，而是sympy表达式


    # print('begin simplify...')
    # mm = simplify(expr)
    # print(mm)
    # print(printing.latex(mm))


    latex_print = printing.latex(expr)
    
    
    def _shorten(mystr):
        def find_nums(mystr):
            import re
            return re.findall(r"\d+\.\d*",mystr)
        def replace_with(mystr, num1, num2):
            list_ = mystr.split(num1)
            return num2.join(list_)
        def replace_all_num(mystr, num1_list, num2_list):
            for i in range(len(num1_list)):
                mystr = replace_with(mystr, num1_list[i], num2_list[i])
            return mystr
        def cut_short(num1_list):
            num2_list = []
            for num1 in num1_list:
                num2_list.append(num1[:4])
            return num2_list
        print('\n\norig:\n\n',mystr)
        num1_list = find_nums(mystr)
        num2_list = cut_short(num1_list)
        mystr_short = replace_all_num(mystr, num1_list, num2_list)
        print('\n\ncutted short:\n\n',mystr_short)
        print('\n\n')
        return mystr_short
    
    shortened_latex = _shorten(latex_print)
    return shortened_latex 


# orig_expr_2_shortened_latex(expr)
# raise












t=np.linspace(0,0.95,100)
y=-95.2351875*(t - 0.68077416)**4*(t - 0.18431336)**4 - 0.0021136854074292
# plt.close('all')
# plt.plot(t,y)
# raise







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

        y_pred1 = mult(cube(plus(mult(sign(cube(cube(mult(g0, 0.009727086)))), 3.7162495), pow(asinh(plus(plus(asinh(pow(plus(plus(plus(g14, g4), g11), relu(g8)), sqrtm(sqrtm(g4)))), plus(pow(plus(plus(tanh(sinh(tanh(plus(pow(plus(tanh(pow(g12, abs(sinh(plus(plus(pow(plus(greater(g14, g19), g0), 0.8133476), plus(sinh(plus(plus(greater(g11, plus(greater(-0.017733395, g10), plus(greater(-0.013830963, g15), -0.020104267))), g1), plus(g18, 0.28218478))), g14)), g15))))), asinh(asinh(cube(pow(cube(mult(plus(g5, g10), 0.22884895)), abs(g2)))))), 2.774574), -0.28017968)))), sinh(sinh(sinh(pow(mult(g6, square(mean_g)), plus(pow(abs(plus(erfc(plus(square(square(plus(square(exp(g16)), plus(g10, g13)))), g9)), mult(g1, plus(plus(g8, 1.8794711), g19)))), 0.8866711), abs(g10))))))), plus(tanh(plus(plus(cube(pow(sinh(sinh(plus(tanh(tanh(plus(tanh(plus(asinh(tanh(plus(mult(plus(pow(plus(plus(g7, g17), g15), 0.9425374), g3), 9.934433), 0.064821236))), sinh(abs(g16)))), g7))), pow(plus(mult(g9, 0.45747387), g2), 0.23265901)))), 9.578903)), -0.17729722), sqrtm(relu(g19)))), mult(pow(g19, 0.2925597), 0.358058))), plus(pow(plus(mult(g19, 0.8707853), g10), plus(plus(g12, g11), 0.21399178)), 3.102611)), pow(plus(plus(g1, g1), plus(relu(g9), g16)), 0.14836852))), 0.33483222)), plus(sinh(tanh(sinh(plus(plus(plus(plus(mult(mean_g, 19.89673), abs(g19)), mean_g), abs(g19)), g5)))), 1.2230495)))), -2.8221632e-5)





        y_pred2 = mult(pow(relu(mult(g0, -0.008472766)), plus(asinh(plus(t, erfc(0.37580237))), plus(relu(plus(plus(plus(tanh(plus(g4, plus(tanh(tanh(tanh(tanh(tanh(tanh(relu(mult(plus(g2, plus(plus(g1, g3), 0.020662013)), 3.0930731)))))))), square(g0)))), mean_g2), abs(g0)), mean_g2)), t))), 1.320138)




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




