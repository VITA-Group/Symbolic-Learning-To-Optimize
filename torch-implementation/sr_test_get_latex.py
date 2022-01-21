

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





def pow(x,y):
    return Pow(x,y)
def plus(x,y):
    return Add(x,y)
def mult(x,y):
    return Mul(x,y)
def square(x):
    return x**2
def relu(x):
    return None
def greater(x,y):
    return None
def cube(x):
    return x**3
def div(x,y):
    return x/y
def sqrtm(x):
    return sqrt(x)
def logm(x):
    return log(x)
def relu(x):
    return  tanh(x)**2




name_list = ['mt', 'gt', 'g', 'mom5', 'mom9', 'mom99'   ]

N_pre = 100

for name in name_list:
    for i in range(N_pre):
        exec(f"{name}_{i} = sympy.symbols('{name}_{i}')")
        # exec(f"{name}{i} = sympy.symbols('{name}{i}')")




def orig_expr_2_shortened_latex(expr):

    

    print('begin simplify...')
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











# expr = sinh((((mom9_18 + -0.83260727) * ((sinh(mt_1) + (g_0 * 1.377225)) + (mom99_15 * 3.8932905))) + ((mom9_7 + 0.02196215) + mom5_4)) + (mom5_1 + mom9_6))


expr = tanh(tanh(0.024803324 + sinh(cube(-1.9923366 * mt_0))))
    # -0.77700216*mv_0





print('...finished')
orig_expr_2_shortened_latex(expr)






















