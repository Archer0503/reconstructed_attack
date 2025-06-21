import torch
import scipy.stats
from DP_util import get_std
from file_path import temp_path
from metrics import tv_element_wise, psnr_compute, mse_compute


class MyAttacker():
    def __init__(self, cfg_attack, data_cfg):
        self.data_cfg = data_cfg
        self.attack_cfg = cfg_attack

    def reconstruct(self, server_payload, shared_data, clipping_bound, server_secrets=None, dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        bias_grad = shared_data[0]["gradients"][bias_idx].clone()
        weight_grad = shared_data[0]["gradients"][weight_idx].clone()

        '''if self.cfg.use_dp:
            # reduce bias noise, to find the original 0 elements in bias_grad
            ci = scipy.stats.norm.interval(0.95, loc=0, scale=get_std(clipping_bound=clipping_bound))
            for idx in range(len(bias_grad)):
                if ci[0] <= bias_grad[idx] <= ci[1]:
                    bias_grad[idx] = 0'''

        if self.cfg.sort_by_bias:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            _, order = server_payload[0]["parameters"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        valid_bias = bias_grad[bias_grad.nonzero()]
        valid_weight = weight_grad[bias_grad.nonzero()]

        # the last non-zero idx of each fig in res, only used in metrics calculation
        res_non_zero_idx = torch.where(bias_grad != 0)[0]
        torch.save(res_non_zero_idx, r'intermediate results/res non zero idx.pt')

        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]
            bias_grad[0] = 0  # the first element contains all pictures

        # This is the attack:
        layer_inputs = self.invert_fc_layer(valid_weight, valid_bias, [])

        # Reduce hits if necessary:
        # layer_inputs = self.reduce_hits(layer_inputs, weight_grad, bias_grad, shared_data)

        # Reshape images, re-identify token embeddings:
        reconstructed_inputs = self.reformat_data(layer_inputs, rec_models, shared_data, server_payload, server_secrets)
        reconstructed_user_data = dict(data=reconstructed_inputs, labels=labels)

        return reconstructed_user_data, stats

    def conv_reconstruct(self, gradients, optim=True, filter=True):
        conv_idx = 0
        shape = self.data_cfg.shape
        channel = shape[0]
        w_grad = gradients[1]
        w_grad = torch.reshape(w_grad,
                               (self.attack_cfg.num_bins, channel*self.attack_cfg.attack_conv_kernel, *shape[-2:]))
        attack_b = gradients[2][:, :100].mean(dim=1)
        bin_idx = torch.nonzero(attack_b, as_tuple=True)
        attack_w = w_grad[:, conv_idx * channel:conv_idx * channel + channel, :, :]  # one of the kernel
        attack_b = attack_b[bin_idx]
        reconstruction = attack_w[bin_idx] / attack_b.reshape((len(attack_b), 1, 1, 1))

        last_non_zero_idx = torch.tensor(torch.load(temp_path + r'/last non zero idx.pt')).to(w_grad.device)
        valid_image_bias_idx = bin_idx[0]

        separated_res_idx = []
        input_idx = []  # the input index only used in metrics calculation
        for res_idx, res_bias_idx in enumerate(valid_image_bias_idx):
            if res_bias_idx in last_non_zero_idx and len(torch.where(last_non_zero_idx == res_bias_idx)[0]) == 1:
                separated_res_idx.append(res_idx)
                input_idx.append(torch.where(last_non_zero_idx == res_bias_idx)[0].to('cpu'))

        # This step should manually distinguish the noise samples from all reconstructed samples
        # for the convenience of experiment, directly use the input idx to obtain the noisy meaningless samples
        res = reconstruction[torch.tensor(separated_res_idx)]

        if filter:
            print('Filtering...')
            # get std first, then filter pixel according to the confidence interval
            noised_w = w_grad[:, 3:-1, :, :]
            fitting_std = - noised_w[noised_w < 0].mean() / torch.sqrt(torch.tensor(2 / torch.pi))
            ci = scipy.stats.norm.interval(0.99, loc=0, scale=fitting_std.to('cpu'))
            scale = torch.abs(attack_b[separated_res_idx])

            if not optim:
                for idx in range(len(res)):
                    condition = torch.logical_and(ci[0] / scale[idx] < res[idx], res[idx] < ci[1] / scale[idx])
                    res[idx] = torch.where(condition, 0, res[idx])

        qualified_res = True
        if optim:
            print('Optimizing...')
            metrics = (gradients[3].mean(dim=0) / gradients[4].mean()).reshape((self.data_cfg.batch_size, -1))[:, :-1]
            metrics = metrics.reshape((self.data_cfg.batch_size, self.data_cfg.shape[0], 3))
            x_mean = (metrics[:, 0, :] / self.attack_cfg.ap.w_mean)[input_idx, :]
            x_var = (metrics[:, 1, :] / self.attack_cfg.ap.w_var)[input_idx, :]
            x_tv = (metrics[:, 2, :] / self.attack_cfg.ap.w_tv)[input_idx, :]
            lr = 5e-3
            optim = torch.optim.Adam([res], lr=lr)
            loss = torch.nn.MSELoss()
            for r in range(2000):
                res.requires_grad = True

                optim.zero_grad()
                res_mean = res.reshape((len(res), 3, -1)).mean(dim=2).to('cuda')
                res_var = res.reshape((len(res), 3, -1)).var(dim=2).to('cuda')
                res_tv = tv_element_wise(res, device=res.device).to('cuda')
                mean_diff = self.attack_cfg.ap.w_mean_diff * loss(res_mean, x_mean)
                var_diff = self.attack_cfg.ap.w_var_diff  * loss(res_var, x_var)
                tv_diff = self.attack_cfg.ap.w_tv_diff * loss(res_tv, x_tv)
                target = mean_diff + var_diff + tv_diff
                target.backward()
                '''if r % 100 == 0:
                    print(r)
                    print(mean_diff, var_diff, tv_diff)
                    print(target)
                    print('-' * 40)'''

                if target > 1000:
                    qualified_res = False
                    break

                optim.step()

            if filter:
                # get std first, then filter pixel according to the confidence interval
                res.requires_grad = False
                for idx in range(len(res)):
                    condition = torch.logical_and(ci[0] / scale[idx] < res[idx], res[idx] < ci[1] / scale[idx])
                    res[idx] = torch.where(condition, 0, res[idx])
                res.requires_grad = True
        return res, input_idx, qualified_res
