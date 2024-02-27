from losses import *

def get_bound(args):

    if args.loss == 'JNPL':
        return 3.
    else:
        return args.grad_bound

MNIST_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10, q=0.01),
    "SCE": SCELoss(num_classes=10, a=0.01, b=1),
    "DMI": DMILoss(num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=100, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=100, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=100, num_classes=10),
    "JNPL": CustomLoss(class_num=10, param=0.1),
}

CIFAR10_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=10),
    "GCE": GCELoss(num_classes=10, q=0.01),
    "SCE": SCELoss(num_classes=10, a=0.1, b=1),
    "DMI": DMILoss(num_classes=10),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=10),
    "NGCE": NGCELoss(num_classes=10),
    "NCE": NCELoss(num_classes=10),
    "NFL+RCE": NFLandRCE(alpha=1, beta=1, num_classes=10, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=1, beta=1, num_classes=10),
    "NCEandRCE": NCEandRCE(alpha=1, beta=1, num_classes=10),
    "JNPL": CustomLoss(class_num=10, param=0.01)
}

CIFAR100_CONFIG = {
    "CE": nn.CrossEntropyLoss(),
    "FL": FocalLoss(gamma=0.5),
    "MAE": MAELoss(num_classes=100),
    "GCE": GCELoss(num_classes=100, q=0.001),
    "SCE": SCELoss(num_classes=100, a=6, b=0.1),
    "DMI": DMILoss(num_classes=100),
    "NFL": NormalizedFocalLoss(gamma=0.5, num_classes=100),
    "NGCE": NGCELoss(num_classes=100),
    "NCE": NCELoss(num_classes=100),
    "NFL+RCE": NFLandRCE(alpha=10, beta=1, num_classes=100, gamma=0.5),
    "NCEandMAE": NCEandMAE(alpha=10, beta=1, num_classes=100),
    "NCEandRCE": NCEandRCE(alpha=10, beta=1, num_classes=100),
    "JNPL": CustomLoss(class_num=100, param=0.01)
}

def get_loss_config(dataset, train_loader, num_classes, args, loss):
    if loss == 'GCE' and not args.is_student:
        return GCELoss(num_classes=num_classes)
    if loss == 'JNPL' and args.is_student:
        return CustomLoss(class_num=num_classes, param=1)
    if loss == 'JS' and args.is_student:
        return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.1, 0.9])
    if loss == 'NCEandRCE' and args.is_student:
        return NCEandRCE(alpha=1, beta=10, num_classes=num_classes)

    if dataset == 'MNIST':
        if loss == 'JS':
            if args.noise_rate == 0.:
                return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.1, 0.9])
            elif args.noise_type == 'symmetric':
                if args.noise_rate == 0.2:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.7, 0.3])
                elif args.noise_rate == 0.4:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.7, 0.3])
                elif args.noise_rate == 0.6:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.9, 0.1])
                elif args.noise_rate == 0.8:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.9, 0.1])
                else:
                    raise ValueError('Not Implemented')
            elif args.noise_type == 'asymmetric':
                return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.3, 0.7])
            else:
                raise ValueError('Not Implemented')
        elif loss in MNIST_CONFIG:
            return MNIST_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')

    if dataset == 'CIFAR10':
        if loss == 'JS':
            if args.noise_rate == 0.:
                return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.1, 0.9])
            elif args.noise_type == 'symmetric':
                if args.noise_rate == 0.2:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.7, 0.3])
                elif args.noise_rate == 0.4:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.7, 0.3])
                elif args.noise_rate == 0.6:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.9, 0.1])
                elif args.noise_rate == 0.8:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.9, 0.1])
                else:
                    raise ValueError('Not Implemented')
            elif args.noise_type == 'asymmetric':
                return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.3, 0.7])
            else:
                raise ValueError('Not Implemented')
        elif loss in CIFAR10_CONFIG:
            return CIFAR10_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')

    if dataset == 'CIFAR100':

        if loss == 'JS':
            if args.noise_rate == 0.:
                return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.1, 0.9])
            elif args.noise_type == 'symmetric':
                if args.noise_rate == 0.2:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.1, 0.9])
                elif args.noise_rate == 0.4:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.3, 0.7])
                elif args.noise_rate == 0.6:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.5, 0.5])
                elif args.noise_rate == 0.8:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.3, 0.7])
                else:
                    raise ValueError('Not Implemented')
            elif args.noise_type == 'asymmetric':
                if args.noise_rate == 0.2:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.5, 0.5])
                elif args.noise_rate == 0.3:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.5, 0.5])
                elif args.noise_rate == 0.4:
                    return JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=[0.5, 0.5])
                else:
                    raise ValueError('Not Implemented')
            else:
                raise ValueError('Not Implemented')
        elif loss in CIFAR100_CONFIG:
            return CIFAR100_CONFIG[loss]
        else:
            raise ValueError('Not Implemented')


def get_mnist_exp_criterions_and_names(num_classes):
    return list(MNIST_CONFIG.keys()), list(MNIST_CONFIG.values())

def get_cifar10_exp_criterions_and_names(num_classes, train_loader=None):
    return list(CIFAR10_CONFIG.keys()), list(CIFAR10_CONFIG.values())

def get_cifar100_exp_criterions_and_names(num_classes, train_loader):
    return list(CIFAR100_CONFIG.keys()), list(CIFAR100_CONFIG.values())


def get_params_lt(dataset, label, arg):
    if '+LT' in label and '+SR' not in label:
        if dataset == 'MNIST':
            if (arg.noise_type == 'symmetric' and arg.noise_rate == 0.8):
                return (0.1, 1e-2)
            else:
                return (0.3, 5e-2)
        elif dataset == 'CIFAR10':
            if (arg.noise_type == 'asymmetric' and arg.loss == 'JNPL'):
                return (0.01, 5e-2)

            if (arg.noise_type == 'symmetric' and arg.noise_rate == 0.8):
                return (0.01, 1e-3)
            else:
                return (0.1, 5e-2)
        elif dataset == 'CIFAR100':
            if (arg.noise_type == 'asymmetric' and arg.loss == 'JNPL'):
                return (0.01, 5e-2)

            if (arg.noise_type == 'asymmetric' and arg.loss == 'CE'):
                return (0.01, 5e-3)

            if (arg.noise_type == 'asymmetric' and arg.loss == 'GCE'):
                return (0.01, 5e-3)

            if (arg.noise_type == 'asymmetric' and arg.loss == 'SCE'):
                return (0.01, 5e-3)

            if (arg.noise_type == 'symmetric' and arg.noise_rate == 0.6) or (
                    arg.noise_type == 'symmetric' and arg.noise_rate == 0.8):
                return (0.01, 1e-3)
            else:
                return (0.05, 5e-2)
    else:
        return 0, 0

