
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LTLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, all_epoch, alpha=0.1, lambda_=0.01):
        super(LTLoss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.all_epoch = all_epoch
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.raw_freedom_degrees = nn.Parameter(torch.randn(num_classes, 1))
        nn.init.xavier_uniform_(self.means)
        nn.init.xavier_uniform_(self.raw_freedom_degrees)

        print('alpha={}, lambda_={}'.format(self.alpha, self.lambda_))

    def forward(self, feat, labels=None):
        feat = F.normalize(feat, dim=1)
        means = F.normalize(self.means, dim=1)
        freedom_degrees = self.raw_freedom_degrees.squeeze() ** 2 + 0.1
        batch_size = feat.size()[0]
        feat_len = torch.tensor(feat.size()[1], dtype=torch.float32).to(self.device)

        XY = torch.matmul(feat, torch.transpose(means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(means, 0, 1) ** 2, dim=0, keepdim=True)
        real_neg_sqr_dist = XX - 2.0 * XY + YY

        if labels is None:
            sqr_dist = real_neg_sqr_dist

        else:
            labels_reshped = labels.view(labels.size()[0], -1)

            if self.device == 'cuda':
                ALPHA = torch.zeros(batch_size, self.num_classes).to(self.device).scatter_(1, labels_reshped,
                                                                                           self.alpha)
                K = ALPHA + torch.ones([batch_size, self.num_classes]).to(self.device)
            else:
                ALPHA = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_reshped, self.alpha)
                K = ALPHA + torch.ones([batch_size, self.num_classes])

            sqr_dist = torch.multiply(K, real_neg_sqr_dist)

        fd = torch.tile(torch.unsqueeze(freedom_degrees, dim=0), (batch_size, 1))

        exp = (fd + torch.log(feat_len)) / 2.0

        neg_sqr_dist = 1. + sqr_dist / fd
        all_neg_sqr_dist = torch.pow(neg_sqr_dist + 1e-8, -exp)

        c_up = torch.exp(torch.lgamma(torch.clamp(exp, 1e-8, np.inf)))
        c_down = torch.exp(torch.lgamma(torch.clamp(fd / 2., 1e-8, np.inf))) * torch.pow(fd, torch.log(feat_len) / 2.)

        logit = c_up / c_down * all_neg_sqr_dist

        logit = F.normalize(logit, dim=1)

        if labels is None:
            psudo_labels = torch.argmax(logit, dim=1)
            means_batch = torch.index_select(means, dim=0, index=psudo_labels)
            center_loss = self.lambda_ * (torch.sum((feat - means_batch) ** 2) / 2) * (1. / batch_size)  # center_loss
        else:
            means_batch = torch.index_select(means, dim=0, index=labels)
            center_loss = self.lambda_ * (torch.sum((feat - means_batch) ** 2) / 2) * (1. / batch_size)  # center_loss
        return logit, center_loss, means, self.alpha, self.lambda_
