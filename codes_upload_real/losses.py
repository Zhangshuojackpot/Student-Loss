import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



eps = 1e-7

class Bootstrapping_Soft(nn.Module):
    def __init__(self, num_classes=10, beta=0.95):
        super(Bootstrapping_Soft, self).__init__()
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, y_pred, labels):
        pred = F.softmax(y_pred, dim=1)
        #pred /= torch.sum(pred, dim=-1, keepdim=True)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        mix_labels = (self.beta * label_one_hot + (1. - self.beta) * pred)
        loss = (-1 * torch.sum(mix_labels * torch.log(pred), dim=-1))
        return loss.mean()


class Bootstrapping_Hard(nn.Module):
    def __init__(self, num_classes=10, beta=0.8):
        super(Bootstrapping_Hard, self).__init__()
        self.num_classes = num_classes
        self.beta = beta

    def forward(self, y_pred, labels):
        pred = F.softmax(y_pred, dim=1)
        #pred /= torch.sum(pred, dim=-1, keepdim=True)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        pred_one_hot = F.one_hot(torch.argmax(pred, 1), self.num_classes).float().to(pred.device)
        pred_one_hot = torch.clamp(pred_one_hot, min=1e-4, max=1.0)
        mix_labels = (self.beta * label_one_hot + (1. - self.beta) * pred_one_hot)
        loss = (-1 * torch.sum(mix_labels * torch.log(pred), dim=-1))
        return loss.mean()

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=1, b=1):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss


class RCELoss(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * loss.mean()

class NRCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NRCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        norm = 1 / 4 * (self.num_classes - 1)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * norm * rce.mean()


class NCELoss(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = -1 * torch.sum(label_one_hot * pred, dim=1) / (-pred.sum(dim=1))
        return self.scale * loss.mean()

class MAELoss(nn.Module):
    def __init__(self, num_classes=10, scale=2.0):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * loss.mean()

class NMAE(nn.Module):
    def __init__(self, num_classes=10, scale=1.0):
        super(NMAE, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        norm = 1 / (self.num_classes - 1)
        loss = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * norm * loss.mean()

class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class NGCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7, scale=1.0):
        super(NGCELoss, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        loss = numerators / denominators
        return self.scale * loss.mean()




class NCEandRCE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)

class NCEandMAE(nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCELoss(num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes=10, ln_neg=1):
        super(NLNL, self).__init__()
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).cuda().random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).cuda()
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0.5, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, num_classes=10, alpha=None, size_average=True, scale=1.0):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.rce = RCELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(gamma=gamma, num_classes=num_classes, scale=alpha)
        self.mae = MAELoss(num_classes=num_classes, scale=beta)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


class JensenShannonDivergenceWeightedScaled(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights = weights

        self.scale = -1.0 / ((1.0 - self.weights[0]) * np.log((1.0 - self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001

    def custom_kl_div(self, prediction, target):
        output_pos = target * (target.clamp(min=1e-7).log() - prediction)
        zeros = torch.zeros_like(output_pos)
        output = torch.where(target > 0, output_pos, zeros)
        output = torch.sum(output, dim=1)
        return output.mean()


    def forward(self, pred, labels):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1))
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float()
        distribs = [labels] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()

        jsw = sum([w * self.custom_kl_div(mean_distrib_log, d) for w, d in zip(self.weights, distribs)])
        return self.scale * jsw


class CustomLoss(torch.nn.Module):
    def __init__(self, class_num, param=0.01):
        super(CustomLoss, self).__init__()
        self.class_num = class_num
        self.th = 1/self.class_num
        self.param = param
        self.l = [-100000000000000.]
        for i in range(self.class_num-1):
            self.l.append(100000000000000.)

    def th_gather_nd(self, x, coords):
        x = x.contiguous().cpu()
        coords = coords.cpu()
        inds = coords.mv(torch.LongTensor(x.stride()))
        x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
        return x_gather.to('cuda')

    def forward(self, y_pred, y_true2):

        y_pred = F.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0)

        self.batch_size = y_pred.size()[0]
        self.cp_label = y_true2[0:self.batch_size]
        y_true = y_true2[:self.batch_size]

        self.cp_label_onehot = F.one_hot(self.cp_label, num_classes=self.class_num)                    #shape = (batch_size, class_nums)
        self.y_true_onehot = F.one_hot(y_true, num_classes=self.class_num)                             #shape = (batch_size, class_nums)
        self.predict_label = torch.reshape(torch.argmax(y_pred, dim=1), [self.batch_size, 1])                   #shape = (batch_size, 1)

        y_true = torch.reshape(y_true, [self.batch_size, 1])
        self.cp_label = torch.reshape(self.cp_label, [self.batch_size, 1])

        NL_score = self.NL(y_pred, self.cp_label)
        PL_score = self.PL(y_pred, self.predict_label)
        score = NL_score + self.param * PL_score
        #max = tf.constant([5],dtype='float32')
        #out = tf.where(score < 5, x=score, y=)

        return score

    def NL(self, y_pred, cp_label):
        # build a index to gather the socre from the socre matrix
        index = torch.linspace(0, self.batch_size-1, self.batch_size).unsqueeze(dim=-1).to(torch.int64).to('cuda')                  #shape = (batch_size, 1)
        index = torch.concat((index, cp_label), dim=1)                                            # shape = (batch_size, 2)
        py = self.th_gather_nd(y_pred, index).unsqueeze(dim=0)                                                       # shape = (1,batch_size)

        # calculate the NL cross_entropy between the cp_label and the predict score
        cp_label_onehot = F.one_hot(cp_label, num_classes=self.class_num).squeeze()     #shape = (batch_size, class_nums)



        cross_entropy = cp_label_onehot * torch.log(torch.clamp(1-y_pred, 1e-7, 1.))                                #shape = (batch_size, class_nums)
        cross_entropy = torch.sum(cross_entropy, dim=1)                                    #shape = (batch_size, 1)

        weight = -1. + py                                                                      #shape = (1, batch_size)
        out = torch.matmul(weight, torch.reshape(cross_entropy, [self.batch_size, 1]))                        #shape = (1,1)
        #out = -torch.sum(cross_entropy,dim=-1)
        return torch.reshape(out, [1]) / float(self.batch_size)                                                            #shape = (1, )

    def PL(self, y_pred, pred_label):
        # select the predict score that satisfied the th
        one = torch.ones_like(y_pred)
        zero = torch.zeros_like(y_pred)
        label = torch.where(y_pred < self.th, zero, one)                   #shape = (batch_size, class_num)
        label = torch.sum(label, dim=-1)                               #shape = (batch_size)

        one = torch.ones_like(label)
        zero = torch.zeros_like(label)
        label = torch.where(label < 2, one, zero)
        D = y_pred * torch.reshape(label, [self.batch_size, 1])                #shape = (batch_size, class_num)

        # calculate the PL
        num = self.batch_size
        index = torch.reshape(torch.linspace(0, num - 1, num), [num, 1]).to(torch.int64).to('cuda')                                      # shape = (n, 1)
        D_label = torch.reshape(torch.argmax(D, dim=1), [num, 1])                                       # shape = (n, 1)
        index = torch.concat((index, D_label), dim=1)                         # shape = (n, 2)
        py = self.th_gather_nd(D, index).unsqueeze(dim=0)                                         # shape = (1, n)
        py = 1 + torch.square(py)                                         # shape = (1, n)
        py = torch.reshape(py, [1,num])                                        # shape = (n)
        weight = torch.prod(py)
        #weight = 1

        one_hot = self.y_true_onehot * torch.reshape(pred_label, [self.batch_size, 1])
        cross_entropy = one_hot * torch.log(y_pred)            # shape = (batch_size, class_nums)
        cross_entropy = torch.sum(cross_entropy, dim=1)                # shape = (batch_size, 1)

        out = -weight * cross_entropy
        out = torch.sum(out, dim=0)

        return out


class NPCLLoss(torch.nn.Module):
    def __init__(self, class_num, Lrate, Nratio):
        super(NPCLLoss, self).__init__()

        self.class_num = class_num
        self.Lrate = Lrate
        self.Nratio = Nratio

    def HardHingeLoss(self, logit, groundTruth):
        Nc = logit.data.size()
        y_onehot = torch.FloatTensor(len(groundTruth), Nc[1]).cuda()

        y_onehot.zero_()
        y_onehot.scatter_(1, groundTruth.data.view(len(groundTruth), 1), 1.0)
        y = torch.autograd.Variable(y_onehot).cuda()
        t = logit * y
        L1 = torch.sum(t, dim=1)

        M, idx = logit.topk(2, 1, True, True)

        f1 = torch.eq(idx[:, 0], groundTruth).float()
        u = M[:, 0] * (1 - f1) + M[:, 1] * f1

        L = torch.clamp(1.0 - L1 + u, min=0)

        return L

    def logsumexp(self, inputs, dim=None, keepdim=False):
        return (inputs - F.log_softmax(inputs, dim)).mean(dim, keepdim=keepdim)

    def SoftHingeLoss(self, logit, groundTruth):
        Nc = logit.data.size()
        y_onehot = torch.FloatTensor(len(groundTruth), Nc[1]).cuda()

        y_onehot.zero_()
        y_onehot.scatter_(1, groundTruth.data.view(len(groundTruth), 1), 1.0)

        y = torch.autograd.Variable(y_onehot).cuda()
        t = logit * y
        L1 = torch.sum(t, dim=1)
        M, idx = logit.topk(2, 1, True, True)

        f1 = torch.eq(idx[:, 0], groundTruth).float()

        u = self.logsumexp(logit, dim=1) * (1 - f1) + M[:, 1] * f1

        L = torch.clamp(1.0 - L1 + u, min=0)

        return L

    def forward(self, y_1, t, ep):
        ###
        #  y_1 : prediction logit
        #  t   : target
        # Lrate:  true/false  at the initiliztion phase (first a few epochs) set false to train with an upperbound ;
        #                     at the working phase , set true to traing with NPCL.
        # Nratio:  noise ratio , set to zero for the clean case(it becomes CL when setting to zero)

        ###
        y_1 = F.softmax(y_1, dim=1)
        y_1 = torch.clamp(y_1, min=1e-7, max=1.0)

        loss_1 = self.SoftHingeLoss(y_1, t)
        ind_1_sorted = torch.argsort(loss_1.data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        epsilon = self.Nratio

        if self.Lrate <= ep:

            Ls = torch.cumsum(loss_1_sorted, dim=0)
            B = torch.arange(start=0, end=-len(loss_1_sorted), step=-1)
            B = torch.autograd.Variable(B).cuda()
            _, pred1 = torch.max(y_1.data, 1)
            E = (pred1 != t.data).sum()
            C = (1 - epsilon) ** 2 * float(len(loss_1_sorted)) + (1 - epsilon) * E
            B = C + B
            mask = (Ls <= B.float()).int()
            num_selected = int(sum(mask))
            Upbound = float(Ls.data[num_selected - 1] <= (C.float() - num_selected))  # footnate in the paper
            num_selected = int(min(round(num_selected + Upbound), len(loss_1_sorted)))

            ind_1_update = ind_1_sorted[:num_selected]

            loss_1_update = self.SoftHingeLoss(y_1[ind_1_update], t[ind_1_update])

        else:
            loss_1_update = self.SoftHingeLoss(y_1, t)

        return torch.mean(loss_1_update)

class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

