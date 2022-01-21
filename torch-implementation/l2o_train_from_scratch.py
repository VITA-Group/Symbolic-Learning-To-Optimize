

from Configs import *
import argparse




# python l2o_train_from_scratch.py  -m tras  -p mni   -d 0



# # =========== Supported modes ===========

# # --- train L2O from scratch ---

# python l2o_train_from_scratch.py  -m tras   -p mni  -n 200  -l 200   -r 20   -b 128   -lr 0.001  -d 0

# python l2o_train_from_scratch.py  -m tras   -f 4   -p mni  -n 200  -l 200   -r 20   -b 128   -lr 0.001  -d 0

# python l2o_train_from_scratch.py  -m tras   -p mni  -ma 1   -n 200  -l 200   -r 20   -b 128   -lr 0.001  -d 2

# python l2o_train_from_scratch.py  -m tras   -opt 1   -p mni     -n 600  -l 200   -r 20   -b 128   -lr 0.001  -d 2


# python l2o_train_from_scratch.py  -m tras   -p cnn  -n 200  -l 200   -r 20   -b 512   -lr 0.001  -d 0
# python l2o_train_from_scratch.py  -m tras   -dope  -pe -1   -p res   -n 200  -l 1000   -r 20   -b 64   -lr 0.001  -d 0



# # --- fine tune L2O ---

# python l2o_train_from_scratch.py  -m tune   -pr 1   -p cnn   -n 200  -l 1000   -r 20   -b 64   -lr 0.001  -d 0
# python l2o_train_from_scratch.py  -m tune   -pr 1   -dope -pe -1 -p res  -n 200  -l 1000   -r 20   -b 64   -lr 0.001  -d 0





# # --- Normal train resnet18 from scratch ---

# python l2o_train_from_scratch.py  -m nort


# python l2o_train_from_scratch.py  -m tras   -p mni  -ma 0   -n 200  -l 200   -r 20   -b 128   -lr 0.001  -d 0


# python l2o_train_from_scratch.py  -m tras   -p mni  -ma 2   -n 200  -l 200   -r 20   -b 128   -lr 0.001  -d 2




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')

parser = argparse.ArgumentParser(description='new args to assign')


parser.add_argument('--mode', '-m', type=str, default='tras', choices = ['tras', 'tune', 'nort'], help='whether to train L2O from scratch or fine tune L2O; not related to train_scratch/tune optimizee')

parser.add_argument('--pre_zer', '-pr', type=int, default=0, help="which pretrained optimizer")

parser.add_argument('--do_with_pre_zee', '-dope', action='store_true')
parser.add_argument('--pre_zee', '-pe', type=int, default=-1, help="which pretrained optimizee")

parser.add_argument('--which_problem', '-p', type=str, default='mni', help="which problem", choices = ['mni', 'cnn', 'res'])


parser.add_argument('--n_epochs', '-n', type=int, default=200)
parser.add_argument('--epoch_len', '-l', type=int, default=200)
parser.add_argument('--unroll', '-r', type=int, default=20)
parser.add_argument('--batch_size', '-b', type=int, default=256)
parser.add_argument('--num_workers', '-w', type=int, default=8)
parser.add_argument('--META_OPT_LR', '-lr', type=float, default=0.001)

parser.add_argument('--device', '-d', type=int, default=0, help="which GPU")
parser.add_argument('--pin_memory', type=bool, default=True, help="only used for MNISTLoss_f and Cifar_half_f")
parser.add_argument('--magic', '-ma', type=int, default=None, help="which magic")
parser.add_argument('--which_OPT', '-opt', type=int, default=0, help="rp/dm")
parser.add_argument('--grad_features', '-f', type=str, default='mt+g+mom5+mom9+mom99', help="grad_features for RP")







newArgs = parser.parse_args()


if __name__ == '__main__':
    
    args['OPT'] = [RPOptimizer,DMOptimizer] [newArgs.which_OPT]

    args.update(**vars(newArgs))
    mode = newArgs.mode
    which_problem = newArgs.which_problem
    if torch.cuda.is_available(): torch.cuda.set_device(args['device'])


    if mode == 'tras':
        # =========== Training L2O from scratch ===========
        if newArgs.do_with_pre_zee: do_with_pretrained(newArgs.pre_zee, args, is_zee=True)
        args.pop('Target_DataGen',None); args.pop('Target_Optimizee',None)
        do_with_problem(args, *problem_comb[which_problem], **args)
        fit_optimizer_from_scratch(args, **args)


    elif mode == 'tune':
        # =========== Fine tuning ===========
        do_with_pretrained(newArgs.pre_zer, args)
        if newArgs.do_with_pre_zee: do_with_pretrained(newArgs.pre_zee, args, is_zee=True)
        args.pop('Target_DataGen',None); args.pop('Target_Optimizee',None)
        do_with_problem(args, *problem_comb[which_problem], **args)

        train_optimizer(args, **args)


    elif mode == 'nort':
        # =========== Normal train ResNet ===========
        args['n_epochs']= int(1e5)
        args['Target_Optimizee'] = resnet18
        args['OPT_normal_train'] = optim.Adam
        args['lr_normal_train'] = 0.001

        normal_train(args,**args)


    else:
        raise NotImplementedError





