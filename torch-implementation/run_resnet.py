
# inhereted codes from the below repo, for resnet; for part of neurips exp
# https://github.com/kuangliu/pytorch-cifar





from utils import *
from torch.optim.lr_scheduler import LambdaLR
import numpy.linalg as la
import os
import resnet_meta

os.makedirs('wz_saved_models', exist_ok=1)

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.cuda.set_device(bestGPU(1))

# if torch.device('cuda' if torch.cuda.is_available():
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True



def use_model(net_desc='r18',cwd='No pretrained cwd'):
    print('==> Building model..')

    print(f'\n\nUsing: {net_desc}\n\n')

    if net_desc=='r18':
        
        net = resnet_meta.resnet18()

    elif net_desc=='r50':
        net = resnet_meta.resnet50()

    net = net.to(DEVICE)



    load_model(net,cwd)

    return net







def getArgs(net_desc='r', cwd='No pretrained cwd'):
    args = {
            'num_workers':      2,
            'resume':           0,
            'lr':               0.1,
            'cwd':              '',
            'n_epochs':         200,
            'pin_memory':       True,
            'criterion':        nn.CrossEntropyLoss(),


            }

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./datasets', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    testset = torchvision.datasets.CIFAR10(
        root='./datasets', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args['num_workers'], pin_memory=args['pin_memory'])

    args['trainloader'] = trainloader
    args['testloader'] = testloader
    args['net'] = use_model(net_desc, cwd)

    return args




def trainNN_zoo(num_workers=None,resume=None,lr=None,cwd='',n_epochs=None,trainloader=None,testloader=None,net=None,criterion=None,**w):

    best_acc = 0

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    def train_1epoch():
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        if CUDA:
            # iterator = enumerate(trainloader) 
            iterator = enumerate(tqdm(trainloader))
        else:
            iterator = enumerate(tqdm(trainloader))


        for batch_idx, (inputs, targets) in iterator:

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = net._forward_impl(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            avg_loss = train_loss/(batch_idx+1)

            # if CUDA: progress_bar(batch_idx, len(trainloader), f'Loss: {avg_loss:.3f} | Train Acc: {acc:>5.2f} | lr: {optimizer.param_groups[0]["lr"]:.3f}')

        return acc, avg_loss

    rec = []
    for epoch in range(n_epochs):
        print(f'\n\t Epoch: < {epoch}/{n_epochs} >')
        train_acc, train_loss = train_1epoch()
        # test()
        test_acc, best_acc = eva_net(net, testloader, trainloader, best_acc, criterion)
        rec.append([train_acc, train_loss, test_acc, best_acc])
        scheduler.step()
    rec = np.array(rec)
    wzRec(rec[:,0], ttl='train_acc', want_save=True)
    wzRec(rec[:,1], ttl='train_loss', want_save=True)
    wzRec(rec[:,2], ttl='test_acc', want_save=True)
    wzRec(rec[:,3], ttl='best_acc', want_save=True)
    return






@torch.no_grad()
def eva_net(net,testloader,trainloader,best_acc=101.,criterion=None,**w):
    # nonlocal best_acc


    # testloader = trainloader


    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net._forward_impl(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        avg_loss = test_loss/(batch_idx+1)
        
        # progress_bar(batch_idx, len(testloader), f'Loss: {avg_loss:.3f} | Test Acc: {acc:>5.2f}')
    desc = f'\n\n  Final test acc is:  {acc:.2f}\n  avg_loss is: {avg_loss} \n'
    print(desc)
    net.test_acc = acc
    if acc > best_acc:
        save_model(net, f'./wz_saved_models/{net._get_name()}.pth')
        best_acc = acc
        net.best_acc = best_acc

    return acc, best_acc







# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # ||||||||||||||||| main function |||||||||||||||||
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print('\n\n\n    4567654     \n\n\n')

__TRAINED__ = [
    
                'wz_saved_models/meta_resnet18 ~ $93.45 ~ April6.pth',
                'wz_saved_models/___xh_resnet18 ~ $95.57 ~ April7.pth',


                ]



if __name__ == '__main__':



    # ===========================================
    # ============= train a resnet ==============

    args = getArgs('r50','wz_saved_models/cResNet.pth')
    trainNN_zoo(**args)




    # ================================================
    # ============= Evaluate resnet/etc ==============

    args = getArgs('r', __TRAINED__[0])
    test_acc, _ = eva_net(**args)

