class AttackParameters:
    def __init__(self, dataset='ImageNet'):
        if dataset == 'ImageNet' or dataset == 'Caltech256' or dataset == 'Flowers102':
            self.w_mean = 1e0
            self.w_var = 1e1
            self.w_tv = 1e-3
            self.b_std = 0.003
            self.b_loc = 0.003
            self.fcl_w_times = 1e-5
            self.w_mean_diff = 1e6
            self.w_var_diff = 2e4
            self.w_tv_diff = 1e-6
        elif dataset == 'CIFAR100':
            self.w_mean = 1e0
            self.w_var = 1e1
            self.w_tv = 1e-3
            self.b_std = 0.003
            self.b_loc = 0.003
            self.fcl_w_times = 1e-5
            self.w_mean_diff = 1e6
            self.w_var_diff = 2e4
            self.w_tv_diff = 1e-6
