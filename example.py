import torch
from DP_model import ConvAttackModel
from my_attacker import MyAttacker
from DP_util import *
from hyperparameters import AttackParameters
from metrics import cw_ssim, psnr_compute, mse_compute
from file_path import temp_path, processed_data_path, processed_files_path


def perform_attacks(save_fig=True, data_cfg=data_cfg_default(), attack_cfg=attack_cfg_default(), use_dp=True, r=0,
                    dataset='ImageNet', new_samples=True, fixed_idx=False, optim=True, filter=True, sam=None):
    """
        Get the samples and grad, recover the samples through grad
    """
    # setup
    device = data_cfg.device
    setup = dict(device=device, dtype=torch.float)
    batch_size = data_cfg.batch_size
    loss_fn = torch.nn.CrossEntropyLoss()

    # get samples
    fig_idx = []
    if fixed_idx:
        fig_idx = torch.tensor([1200, 17000, 10696, 18000, 10800, 1201, 9629, 9022,
                                5081, 5205, 10697, 10810, 1202, 9032, 6608, 7045])
    if new_samples:
        datapoints, labels = get_samples(fig_idx[:batch_size], device, dataset, batch_size)
        if save_fig:
            torch.save(datapoints, processed_data_path + '/original data.pt')
        datapoints, optimal_masks = select_main_object(sam, datapoints)  # process by SAM
        if save_fig:
            torch.save(datapoints, processed_data_path + '/main_object.pt')
            torch.save(optimal_masks, processed_data_path + '/mask.pt')
    else:
        '''_, labels = get_samples(fig_idx[:batch_size], device, dataset, batch_size)
        datapoints = torch.load(processed_data_path + '/main_object.pt')
        optimal_masks = torch.load(processed_data_path + '/mask.pt')'''
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        datapoints = torch.load(processed_files_path + '/main_object_{0}bs_{1}.pt'.format(batch_size, r))
        optimal_masks = torch.load(processed_files_path + '/mask_{0}bs_{1}.pt'.format(batch_size, r))
    data_cfg.shape = datapoints[0].shape

    # generate gradient
    model = ConvAttackModel(data_cfg, attack_cfg, is_victim=True).to(device)
    loss = loss_fn(model(datapoints), labels)
    gradients = torch.autograd.grad(loss, model.parameters())

    # add noise
    if use_dp:
        gradients = clip_and_perturb(gradients, data_cfg.clipping_bound, data_cfg.epsilon, clip=False)

    # attack:
    attacker = MyAttacker(attack_cfg, data_cfg)
    victim_samples = datapoints
    reconstructed_user_data, input_idx, qualified_res = attacker.conv_reconstruct(gradients, optim=optim, filter=filter)

    #  To distinguish the complete images, overlapped images and meaningless images
    separated_input = victim_samples[input_idx, :, :, :]

    scale_vector = list()
    torch.save(scale_vector, temp_path + '/scale.pt')

    if save_fig:
        # save the training samples and recovered samples locally
        print('Batch size:', batch_size)
        print('Separated image:', len(reconstructed_user_data))

        torch.save(victim_samples, temp_path + '/original processed data.pt')
        torch.save(separated_input, temp_path + '/separated input.pt')
        torch.save(reconstructed_user_data, temp_path + '/separated res.pt')

    return reconstructed_user_data, separated_input, qualified_res


if __name__ == "__main__":
    dataset = 'ImageNet'
    dc = data_cfg_default()
    ac = attack_cfg_default()
    ac.ap = AttackParameters(dataset)
    ac.compress_image = False
    sam = sam_model_registry["default"](checkpoint=sam_path)
    sam.to('cuda')
    record = pd.DataFrame(columns=['batch size', 'bin num', 'separated res', 'separated input'])
    bsz = 4  # batch size
    dc.batch_size = bsz
    rounds = 1
    bin_num = 1024
    torch.cuda.empty_cache()
    print('bs: {0}, bn: {1}, r:{2}'.format(bsz, bin_num, rounds))
    ac.num_bins = bin_num
    # print('batch size: {0}, bins num: {1}, round {2}'.fbkormat(dc.batch_size, ac.num_bins, rounds))
    res = perform_attacks(save_fig=True, data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True,
                          use_dp=True, new_samples=True, optim=False, sam=sam, filter=True)
    if not res[2]:
        print("Cannot optimize from noised gradients")
    record.loc[len(record)] = [dc.batch_size, ac.num_bins, *list(map(len, res[:-1]))]

    accuracy = len(res[0]) / dc.batch_size
    if len(res[0]) == 0:
        print('No separated res')

    separated_res = res[0]
    separated_input = res[1]
    ssim = cw_ssim(separated_res, separated_input)[0]
    psnr = psnr_compute(separated_res, separated_input)[0]
    mse = mse_compute(separated_res, separated_input)
    print(ssim, psnr, mse)

    # record.to_json(r'evaluation/factor/separation ratio.json')
