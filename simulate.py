"""
New simulation file for 24 security usenix summer, created by 2023.05.31
"""
import torchvision.transforms

from DP_util import *
from hyperparameters import AttackParameters
from file_path import sam_path, processed_files_path, temp_path
import pandas as pd
import torch
from metrics import cw_ssim, mse_compute, psnr_compute
from DP_model import ConvAttackModel
from example import perform_attacks


def bs_to_performance():
    dataset = 'ImageNet'
    dc = data_cfg_default()
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    ac.compress_image = False
    record = pd.DataFrame(columns=['batch size', 'bin num', 'separated res', 'separated input', 'psnr', 'ssim', 'mse'])
    for bs in range(2, 3):
        dc.batch_size = int(2 ** bs)
        for rounds in range(10):
            for bin_num in (1024,):
                torch.cuda.empty_cache()
                print('bs: {0}, bn: {1}, r:{2}'.format(bs, bin_num, rounds))
                ac.num_bins = bin_num
                # print('batch size: {0}, bins num: {1}, round {2}'.fbkormat(dc.batch_size, ac.num_bins, rounds))
                res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, r=rounds,
                                      use_dp=True, new_samples=True, optim=True)
                if not res[2]:
                    # cannot optimize from noised gradients
                    continue
                res_list = [dc.batch_size, ac.num_bins, *list(map(len, res[:-1]))]

                if len(res[0]) == 0:
                    print('No separated res')
                    continue

                separated_res = res[0]
                separated_input = res[1]
                ssim = cw_ssim(separated_res, separated_input)[0]
                psnr = psnr_compute(separated_res, separated_input)[0]
                mse = mse_compute(separated_res, separated_input).cpu()
                res_list += [psnr, ssim, mse]
                record.loc[len(record)] = res_list
                print(ssim, psnr, mse)

    record.to_json(r'evaluation/factor/bs_not_optim_cifar.json', default_handler=str)


def privacy_factor_to_performance():
    dataset = 'ImageNet'
    dc = data_cfg_default()
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    ac.compress_image = False
    record = pd.DataFrame(columns=['epsilon', 'CB', 'BS', 'psnr', 'ssim', 'mse'])
    # for epsilon in (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20):
    for CB in (2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20):
        dc.epsilon = 10
        dc.clipping_bound = CB
        for rounds in range(10):
            for batch_size in (8, 16, 32):
                torch.cuda.empty_cache()
                dc.batch_size = batch_size
                print('epsilon: {0}, bs: {1}, r:{2}, CB:{3}'.format(dc.epsilon, batch_size, rounds, CB))
                ac.num_bins = 1024
                res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, r=rounds,
                                      use_dp=True, new_samples=False, optim=True)

                if not res[2]:
                    # cannot optimize from noised gradients
                    continue

                if len(res[0]) == 0:
                    print('No separated res')
                    continue

                separated_res = res[0]
                separated_input = res[1]
                ssim = cw_ssim(separated_res, separated_input)[0]
                psnr = psnr_compute(separated_res, separated_input)[0]
                mse = mse_compute(separated_res, separated_input).cpu()
                res_list = [dc.epsilon, dc.clipping_bound, dc.batch_size, psnr, ssim, mse]
                record.loc[len(record)] = res_list
                print(ssim, psnr, mse)

    record.to_json(r'evaluation/factor/CB_optim.json', default_handler=str)


def different_protection(mode='DP'):
    sam = sam_model_registry["default"](checkpoint=sam_path)
    sam.to('cuda')
    dataset = 'ImageNet'
    dc = data_cfg_default()
    dc.batch_size = 16
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    ac.compress_image = False

    if mode == 'after_processed':
        datapoints, labels = get_samples([1200], 'cuda', 'ImageNet', 1)
        datapoints, optimal_masks = select_main_object(sam, datapoints)  # process by SAM
        torch.save(datapoints, 'evaluation/different_protection/after_processed.pt')
    elif mode == 'original':
        datapoints, labels = get_samples([1200], 'cuda', 'ImageNet', 1)
        torch.save(datapoints, 'evaluation/different_protection/original.pt')
    elif mode == 'none':
        res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True,
                              use_dp=False, new_samples=True, optim=True, sam=sam)
        separated_res = res[0]
        torch.save(separated_res, 'evaluation/different_protection/none.pt')
    elif mode == 'only clip':
        # change the code in perform attacks
        res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True,
                              use_dp=True, new_samples=True, optim=False, sam=sam)
        separated_res = res[0]
        torch.save(separated_res, 'evaluation/different_protection/only_clip.pt')
    elif mode == 'only perturb':
        # change the code in perform attacks
        res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True,
                              use_dp=True, new_samples=True, optim=T, sam=sam)
        separated_res = res[0]
        torch.save(separated_res, 'evaluation/different_protection/only_perturb.pt')
    elif mode == 'DP':
        res = perform_attacks(save_fig=True, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True,
                              use_dp=True, new_samples=True, optim=True, sam=sam)
        separated_res = res[0]
        torch.save(separated_res, 'evaluation/different_protection/DP.pt')


def value_comparison(mode):
    dataset = 'ImageNet'
    device = torch.device('cuda')
    dc = data_cfg_default()
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    dc.shape = (3, 224, 224)
    loss_fn = torch.nn.CrossEntropyLoss()
    if mode == 'abs':
        res = pd.DataFrame(columns=['Batch size', 'ABS value'])
        for bs in (4, 8, 16, 32, 64):
            dc.batch_size = bs
            for r in range(10):
                print(bs, r)
                torch.cuda.empty_cache()
                datapoints = torch.load(processed_files_path + '/main_object_{0}bs_{1}.pt'.format(bs, r))
                model = ConvAttackModel(dc, ac, is_victim=True).to(device)
                labels = torch.randint(0, 1000, (bs,), device=device)
                loss = loss_fn(model(datapoints), labels)
                gradients = torch.autograd.grad(loss, model.parameters())
                gradients = clip_and_perturb(gradients, 10, 10, perturb=False)
                attack_b = gradients[2][:, :100].mean(dim=1)
                bin_idx = torch.nonzero(attack_b, as_tuple=True)[0]
                w_grad = torch.abs(gradients[1][bin_idx]).reshape((len(bin_idx), 6, 224, 224))
                res.loc[len(res)] = [bs, torch.abs(w_grad[:, :3, :, :]).mean().item()]
        res.to_json('evaluation/norm/abs_value_clip.json', default_handler=str)
    elif mode == 'norm':
        res = pd.DataFrame(columns=['Batch size', 'norm'])
        for bs in (4, 8, 16, 32):
            dc.batch_size = bs
            for r in range(10):
                print(bs, r)
                torch.cuda.empty_cache()
                datapoints = torch.load(processed_files_path + '/main_object_{0}bs_{1}.pt'.format(bs, r))
                model = ConvAttackModel(dc, ac, is_victim=True).to(device)
                labels = torch.randint(0, 1000, (bs,), device=device)
                loss = loss_fn(model(datapoints), labels)
                gradients = torch.autograd.grad(loss, model.parameters())
                gradients = clip_and_perturb(gradients, 10, 10, perturb=False)
                norm = sum([g.norm() ** 2 for g in gradients]) ** 0.5
                res.loc[len(res)] = [bs, norm.item()]
        res.to_json('evaluation/norm/norm_clip.json', default_handler=str)
    elif mode == 'unit_norm':
        res = pd.DataFrame(columns=['unit', 'Batch size', 'norm'])
        for unit in (64, 128, 256, 512, 1024):
            dc.batch_size = unit
            for bs in (16, 64):
                dc.batch_size = bs
                for r in range(10):
                    print(unit, bs, r)
                    torch.cuda.empty_cache()
                    datapoints = torch.load(processed_files_path + '/main_object_{0}bs_{1}.pt'.format(bs, r))
                    model = ConvAttackModel(dc, ac, is_victim=True).to(device)
                    labels = torch.randint(0, 1000, (bs,), device=device)
                    loss = loss_fn(model(datapoints), labels)
                    gradients = torch.autograd.grad(loss, model.parameters())
                    # gradients = clip_and_perturb(gradients, 10, 10, perturb=False)
                    norm = sum([g.norm() ** 2 for g in gradients]) ** 0.5
                    res.loc[len(res)] = [unit, bs, norm.item()]
        res.to_json('evaluation/norm/unit_norm.json', default_handler=str)
    elif mode == 'unit_abs':
        res = pd.DataFrame(columns=['unit', 'Batch size', 'norm'])
        for unit in (64, 128, 256, 512, 1024):
            dc.batch_size = unit
            for bs in (16, 64):
                dc.batch_size = bs
                for r in range(10):
                    print(unit, bs, r)
                    torch.cuda.empty_cache()
                    datapoints = torch.load(processed_files_path + '/main_object_{0}bs_{1}.pt'.format(bs, r))
                    model = ConvAttackModel(dc, ac, is_victim=True).to(device)
                    labels = torch.randint(0, 1000, (bs,), device=device)
                    loss = loss_fn(model(datapoints), labels)
                    gradients = torch.autograd.grad(loss, model.parameters())
                    # gradients = clip_and_perturb(gradients, 10, 10, perturb=False)
                    attack_b = gradients[2][:, :100].mean(dim=1)
                    bin_idx = torch.nonzero(attack_b, as_tuple=True)[0]
                    w_grad = torch.abs(gradients[1][bin_idx]).reshape((len(bin_idx), 6, 224, 224))
                    res.loc[len(res)] = [unit, bs, torch.abs(w_grad[:, :3, :, :]).mean().item()]
        res.to_json('evaluation/norm/unit_abs.json', default_handler=str)


def fig_quality_three_metrics(dataset='Caltech256', optim=False):
    # used in table 3, quality comparison with different alg
    dc = data_cfg_default()
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    ac.compress_image = False
    sam = sam_model_registry["default"](checkpoint=sam_path)
    sam.to('cuda')
    record = pd.DataFrame(columns=['ssim', 'psnr', 'mse'])
    for bs in (4,):
        dc.batch_size = int(2 ** bs)
        for rounds in range(10):
            for bin_num in (1024,):
                torch.cuda.empty_cache()
                print('bs: {0}, bn: {1}, r:{2}'.format(bs, bin_num, rounds))
                ac.num_bins = bin_num
                # print('batch size: {0}, bins num: {1}, round {2}'.fbkormat(dc.batch_size, ac.num_bins, rounds))
                res = perform_attacks(save_fig=False, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=False,
                                      use_dp=True, new_samples=False, optim=optim, sam=sam, r=rounds)
                if not res[2]:
                    # cannot optimize from noised gradients
                    continue

                accuracy = len(res[0]) / dc.batch_size
                if len(res[0]) == 0:
                    print('No separated res')
                    continue

                separated_res = res[0]
                separated_input = res[1]
                ssim = cw_ssim(separated_res, separated_input)[0]
                psnr = psnr_compute(separated_res, separated_input)[0]
                mse = mse_compute(separated_res, separated_input).item()
                record.loc[len(record)] = [ssim, psnr, mse]
    if optim:
        record.to_json(r'evaluation/dataset/{0}/proposed_noisy.json'.format(dataset))
    else:
        record.to_json(r'evaluation/dataset/{0}/proposed_no_optim_noisy.json'.format(dataset))


def other_attack(attack='random', dataset='ImageNet'):
    device = torch.device('cuda')
    record = pd.DataFrame(columns=['mse', 'psnr', 'ssim'])
    index_set = index.index_set
    if dataset == 'Flowers102':
        index_set = index.flower_index_set
    if attack == 'random':
        for fig_idx in index_set:
            for _ in range(5):
                datapoints, _ = get_samples(dataset=dataset, image_idx=fig_idx, device=device)
                datapoints = torch.stack([torchvision.transforms.ToTensor()(d) for d in datapoints]).to(device)
                res = torch.rand(size=datapoints.size()).to(device)
                mse = mse_compute(datapoints, res).cpu().item()
                psnr = psnr_compute(datapoints, res)[0]
                ssim = cw_ssim(datapoints, res)[0]
                print([mse, psnr, ssim])
                record.loc[len(record)] = [mse, psnr, ssim]

        print(record.mean(axis=0))


bs_to_performance()
# privacy_factor_to_performance()
# different_protection('only clip')
# value_comparison(mode='unit_abs')
# fig_quality_three_metrics(dataset='Flowers102', optim=True)
# other_attack(dataset='Flowers102')
