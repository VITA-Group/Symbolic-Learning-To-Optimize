


from stage023 import *
import argparse

# =========== Supported modes ===========
    # SR fitting
    # SR fine tune
    # SR eva

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE={DEVICE}')


parser = argparse.ArgumentParser(description='new args to assign')

parser.add_argument('--device', '-d', type=int, default=0, help="which GPU")



newArgs = parser.parse_args()




if __name__ == '__main__':

    # =========== stage 0: Prep <Always Keep Uncomment> ===========

    do_with_pretrained(0, args)
    do_with_problem(args, 
                    MNISTLoss_f, 
                    MLP_MNIST, 
                    eva_epoch_len=[250, 300][1],
                    n_tests=3,
                    )
    args['Ngroup_zee'] = len(list(args['Target_Optimizee']().parameters()))
    sr_prep1(args, 
             SR_memlen = 20   ,
             num_Xyitems_SR = 580  ,
             **args)




    # =========== stage 2: fit SR ===========


    coeff_SR_numpy = np.array([[0.4,0.5],[0,0.1]])
    args['coeff_SR_ini'] = torch.tensor(coeff_SR_numpy, device = DEVICE)
    coeff_SR, coeff_SR_list = fit_SR(args, **args)


    # =========== stage 2: test R2-score ===========
    args['coeff_SR'] = torch.tensor(np.load('wz_saved_models/fitted SR ~ of ~ {l2o_net.name}.npy'), device = DEVICE)
    evaSR_R2(args, **args)




    # =========== stage 3: Fine tune SR ===========

    args['coeff_SR'] = torch.tensor(np.load('wz_saved_models/fitted SR ~ of ~ {l2o_net.name}.npy'), device = DEVICE)
    fine_tune_SR(args, **args)


    # =========== stage 3: Eva SR performance ===========
    eva_sr_trainingPerform(args, **args)







