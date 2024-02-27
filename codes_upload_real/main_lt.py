import argparse
from dataset import DatasetGenerator
from models import *
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import random
from utils import *
from config import *
import sys

import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('darkgrid')
plt.switch_backend('agg')
plt.figure(figsize=(20, 20), dpi=600)

# 限制0号设备的显存的使用量为0.5，就是半张卡那么多，比如12G卡，设置0.5就是6G。
#torch.cuda.set_per_process_memory_fraction(0.4, 0)
#torch.cuda.empty_cache()
# 计算一下总内存有多少。
#total_memory = torch.cuda.get_device_properties(0).total_memory
# 使用0.499的显存:
#tmp_tensor = torch.empty(int(total_memory * 0.39), dtype=torch.int8, device='cuda')

# 清空该显存：
#del tmp_tensor
#torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='Robust loss for learning with noisy labels')
parser.add_argument('--root', type=str, default="database/", help='the data root')
parser.add_argument('--gpus', type=str, default='0')
# learning settings
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=0, help='the number of worker for loading data')
parser.add_argument('--grad_bound', type=float, default=5., help='the gradient norm bound')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--dataset', type=str, default="CIFAR100", metavar='DATA', help='Dataset name (default: CIFAR10)')
parser.add_argument('--noise_type', type=str, default='asymmetric', help='the noise type: clean, symmetric, pairflip, asymmetric')
parser.add_argument('--noise_rate', type=float, default=0.4, help='the noise rate')
parser.add_argument('--is_student', type=int, default=1, help='if use the student loss')
parser.add_argument('--loss', type=str, default='JS', help='the loss functions: CE, FL, GCE')

args = parser.parse_args()

label = args.loss

if args.is_student:
    args.is_student = True
    label = label + '+LT'
else:
    args.is_student = False
    label = label

if args.noise_rate == 0.0:
    args.noise_type = 'clean'

if args.noise_type == 'asymmetric':
    asymm = True
else:
    asymm = False


#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
print('We are using', device)


if device == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)


seed = args.seed
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

print(args)

def evaluate(loader, model, ep):
    model.eval()
    correct = 0.
    total = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if args.is_student == True:
            z, tfeat, tcl, tmean, tra, trl = model(x, None, ep)
        else:
            z, tfeat = model(x)
        probs = F.softmax(z, dim=1)
        pred = torch.argmax(probs, 1)
        total += y.size(0)
        correct += (pred==y).sum().item()

    acc = float(correct) / float(total)
    return acc

def calculate_loss(criterion, out, y, ep=None):
    if args.is_student:
        if args.dataset != 'MNIST':
            out = F.normalize(out, dim=1)
    if args.loss == 'NPCL':
        loss = criterion(out, y, ep)
    else:
        loss = criterion(out, y)
    return loss


data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               asym=args.noise_type=='asymmetric',
                               dataset_type=args.dataset,
                               noise_rate=args.noise_rate
                               )

data_loader = data_loader.getDataLoader()
train_loader = data_loader['train_dataset']
test_loader = data_loader['test_dataset']



if args.is_student == True:
    al, la = get_params_lt(args.dataset, label, args)
else:
    al, la = 0., 0.

if args.dataset == 'MNIST':
    in_channels = 1
    num_classes = 10
    weight_decay = 1e-3
    lr = 0.01
    epochs = 50
elif args.dataset == 'CIFAR10':
    in_channels = 3
    num_classes = 10
    weight_decay = 1e-4
    lr = 0.01
    epochs = 120
elif args.dataset == 'CIFAR100':
    in_channels = 3
    num_classes = 100
    weight_decay = 1e-5
    lr = 0.1
    epochs = 200
    lamb = 10 if asymm else 4
else:
    raise ValueError('Invalid value {}'.format(args.dataset))


criterion = get_loss_config(args.dataset, train_loader, num_classes=num_classes, loss=args.loss, args=args)

norm = None
print(label)

if args.dataset != 'CIFAR100':
    model = CNN(all_epo=epochs, type=args.dataset, if_student=args.is_student, al=al, la=la, device=device).to(device)
else:
    model = ResNet34(all_epo=epochs, num_classes=100, if_student=args.is_student, al=al, la=la, device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

test_accs = []
for ep in range(epochs):
    model.train()
    total_loss = 0.
    for batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        optimizer.zero_grad()

        if args.is_student == True:
            logit, feat, center_loss, means, ralpha, rlambda = model(batch_x, batch_y, ep)
            loss = calculate_loss(criterion, logit, batch_y, ep) + center_loss
        else:
            logit, feat = model(batch_x)
            ralpha, rlambda = 0, 0
            loss = calculate_loss(criterion, logit, batch_y, ep)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        total_loss += loss.item()
    scheduler.step()
    test_acc = evaluate(test_loader, model, ep)

    log('Iter {}/{}: loss={:.4f}, test_acc={:.4f}, lr={:.4f}, ralpha={:.4f}, rlambda={:.4f}'.format(ep, epochs, total_loss, test_acc,
                                                                         scheduler.get_last_lr()[0], ralpha, rlambda))
    test_accs.append(test_acc)

    sys.stdout.flush()

    save_path = 'result/seed'+ str(args.seed) + '/' + str(args.loss) + '/' + str(args.dataset) + '/' + str(args.noise_type)+ '/' + str(args.noise_rate) + '/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.savetxt(save_path + 'student' + str(args.is_student) + '.txt', np.array(test_accs))