# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# from utils import *
# from meta import *
# from utils import __WELL_TRAINED__

from Configs import *
from stage023 import *
import argparse






print(get_newest_file(filter='py'))

# raise

# python l2o_evaluation.py   -m srck   -p mni   -l 300  -d 1
# python l2o_evaluation.py  -m srgen   -p mni  -l 300  -s 50000  -d 3



# =========== Supported modes ===========

# --- train traditional  ---
# python l2o_evaluation.py  -m tradi     -p mni  -n 2    -l 1000  -d 0




# --- purely eva and save fig ---

# python l2o_evaluation.py  -m pure  -pr 0  -p mni  -n 10  -l 300  -d 0





# --- SR check if converge ---

# python l2o_evaluation.py  -m srck  -pr 1  -p mni   -l 300  -d 1
# python l2o_evaluation.py  -m srck  -pr 2  -p mni   -l 300  -d 1




# --- Record loss/acc results in npy file ---

# python l2o_evaluation.py  -m rec  -pr 1  -p mni  -n 20  -l 300  -d 0
# python l2o_evaluation.py  -m rec  -pr 2  -p mni  -n 20  -l 300  -d 0





# --- SR gen data AND save npy ---

# python l2o_evaluation.py  -m srgen  -pr 1  -p mni  -l 300  -s 2000  -d 1









DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')

parser = argparse.ArgumentParser(description='new args to assign')

parser.add_argument('--mode', '-m', type=str, default='pure', choices = ['pure', 'srck', 'srgen', 'rec', 'tradi'])
parser.add_argument('--pre_zer', '-pr', type=int, default=-100, help="which pretrained optimizer")
parser.add_argument('--do_with_pre_zee', '-dope', action='store_true')
parser.add_argument('--pre_zee', '-pe', type=int, default=-1, help="which pretrained optimizee")
parser.add_argument('--which_problem', '-p', type=str, default='mni', help="which problem", choices = ['mni', 'cnn', 'res'])
parser.add_argument('--n_tests', '-n', type=int, default=20)
parser.add_argument('--eva_epoch_len', '-l', type=int, default=200)
parser.add_argument('--num_Xyitems_SR', '-s', type=int, default=200)
parser.add_argument('--SR_memlen', '-mem', type=int, default=20)
parser.add_argument('--device', '-d', type=int, default=0, help="which GPU")
parser.add_argument('--META_OPT_LR', '-lr', type=float, default=0.001)
parser.add_argument('--batch_size', '-b', type=int, default=256)
parser.add_argument('--num_workers', '-w', type=int, default=8)
parser.add_argument('--pin_memory', type=bool, default=True, help="only used for MNISTLoss_f and Cifar_half_f")
parser.add_argument('--want_savel2o_evaluation_loss', type=bool, default=False, help="save losses/acc at every step, every epochs (N_run x eva_epoch_len)")

parser.add_argument('--iexp', type=int, default=0)
parser.add_argument('--which_opt', type=str, default='A')
# parser.add_argument('--magic', '-ma', type=int, default=None, help="which magic")
parser.add_argument('--lr_tradi', type=float, default=0.001)






newArgs = parser.parse_args()




if __name__ == '__main__':


    args.update(**vars(newArgs))
    mode = newArgs.mode
    which_problem = newArgs.which_problem
    # # os.environ["CUDA_VISIBLE_DEVICES"] = "2" # no effect ...
    if torch.cuda.is_available(): torch.cuda.set_device(args['device'])
    args.pop('Target_DataGen',None); args.pop('Target_Optimizee',None);
    problem_comb = {'mni': [MNISTLoss_f, MLP_MNIST],
                    'cnn': [Cifar_half_f, SMALL_CNN],
                    'res': [Cifar_half_f, resnet18],
                    }

    if newArgs.pre_zer==-100:
        pre_zer = RP_config
    else:
        pre_zer = newArgs.pre_zer



    if mode in ['srck', 'pure', 'rec']:

        # =========== Evaluation ===========
        do_with_pretrained(pre_zer, args)
        if newArgs.do_with_pre_zee: do_with_pretrained(newArgs.pre_zee, args, is_zee=True)
        do_with_problem(args, *problem_comb[which_problem], **args)
        if mode == 'rec': args['want_savel2o_evaluation_loss'] = True
        if mode == 'srck': args['n_tests']=2

        eva_l2o_optimizer(args, **args)












    elif mode=='tradi':

        # =========== Evaluation ===========

        if newArgs.do_with_pre_zee: do_with_pretrained(newArgs.pre_zee, args, is_zee=True)
        do_with_problem(args, *problem_comb[which_problem], **args)
        args['want_savel2o_evaluation_loss'] = True
        # args['OPT_TRADI'] = [optim.SGD][0]

        args['tradi_save_name'] = f"{args['which_problem']} $ {args['which_opt']} $ {args['lr_tradi']}"
        eva_l2o_traditional(args, **args)




    elif mode == 'srgen':

        # =========== stage 0: Prep <Always Keep Uncomment> ===========
        do_with_pretrained(pre_zer, args)
        if newArgs.do_with_pre_zee: do_with_pretrained(newArgs.pre_zee, args, is_zee=True)
        do_with_problem(args, *problem_comb[which_problem], **args)
        sr_prep1(args, **args)

        # =========== stage 0: Gen Data ===========
        _, _, sr_r_t_n_f = eva_l2o_optimizer(args, **args)


        # =========== stage 0: Save npy ===========
        SR_Xy, desc = sr_prep2(sr_r_t_n_f, args, **args)
        fname_tosave = f"SR_data/SR_Xy ~ {args['task_desc']} ~ {desc}.npy"
        np.save(fname_tosave, SR_Xy)
        print(f'\n=========\n\n  Database saved @:  {fname_tosave}\n=========')




    else:
        raise NotImplementedError





    # =========== Training L2O from scratch ===========
    do_with_problem(args, 
                    MNISTLoss_f, 
                    MLP_MNIST, 
                    optim.Adam,
                    n_epochs=200,
                    epoch_len=200,
                    unroll=20,
                    batch_size = 4096,
                    num_workers = 1,
                    )
    argsPrinter(args,**args)
    fit_optimizer_from_scratch(args, **args)



    # =========== Fine tuning ===========
    
    do_with_pretrained(1, args)
    do_with_problem(args, 
                    MNISTLoss_f, 
                    MLP_MNIST, 
                    optim.Adam,
                    n_epochs=200,
                    epoch_len=200,
                    unroll=20,
                    )
    train_optimizer(args, **args)






    # =========== Normal train ResNet ===========

    args['n_epochs']= int(1e5)
    args['Target_Optimizee']= resnet18
    normal_train(args,**args)
























