import torch
from DP_model import ConvAttackModel
from attacker import Attacker
from DP_util import *
from attack_params import AttackParameters
from metrics import cw_ssim, psnr_compute, mse_compute
from file_path import temp_path, processed_data_path, processed_files_path
import matplotlib.gridspec as gridspec


def perform_attacks(save_fig=True, data_cfg=data_cfg_default(), attack_cfg=attack_cfg_default(), use_dp=True,
                    dataset='ImageNet', fixed_idx=False, optim=False, filter=True, sam=None):
    """
        Get the samples and grad, recover the samples through grad
    """
    # setup
    print(f"Performing attack with parameters: "
          f"batch_size={data_cfg.batch_size}, bin_num={attack_cfg.num_bins}, "
          f"used_dp={use_dp}, dataset={dataset}, fixed_idx={fixed_idx}, optim={optim}, filter={filter}")
    device = data_cfg.device
    setup = dict(device=device, dtype=torch.float)
    batch_size = data_cfg.batch_size
    loss_fn = torch.nn.CrossEntropyLoss()

    # Process samples
    print('Processing samples...')
    fig_idx = []
    if fixed_idx:
        fig_idx = torch.tensor([1200, 17000, 10696, 18000, 10800, 1201, 9629, 9022,
                                5081, 5205, 10697, 10810, 1202, 9032, 6608, 7045])

    datapoints, labels = get_samples(fig_idx[:batch_size], device, dataset, batch_size)
    if save_fig:
        torch.save(datapoints, processed_data_path + '/original data.pt')
    datapoints, optimal_masks = select_main_object(sam, datapoints)  # process by SAM
    if save_fig:
        torch.save(datapoints, processed_data_path + '/main_object.pt')
        torch.save(optimal_masks, processed_data_path + '/mask.pt')

    data_cfg.shape = datapoints[0].shape

    # generate gradient
    print('Generating gradients...')
    model = ConvAttackModel(data_cfg, attack_cfg, is_victim=True).to(device)
    loss = loss_fn(model(datapoints), labels)
    gradients = torch.autograd.grad(loss, model.parameters())

    # add noise
    if use_dp:
        print('Adding noise...')
        gradients = clip_and_perturb(gradients, data_cfg.clipping_bound, data_cfg.epsilon, clip=False)

    # attack:
    print('Performing attack...')
    attacker = Attacker(attack_cfg, data_cfg)
    victim_samples = datapoints
    reconstructed_user_data, input_idx, qualified_res = attacker.conv_reconstruct(gradients, optim=optim, filter=filter)
    print(f"Reconstructed {len(reconstructed_user_data)} samples from gradients.")

    #  To distinguish the complete images, overlapped images and meaningless images
    separated_input = victim_samples[input_idx, :, :, :]

    scale_vector = list()
    torch.save(scale_vector, temp_path + '/scale.pt')

    if save_fig:
        print('Saving results...')
        torch.save(victim_samples, temp_path + '/original processed data.pt')
        torch.save(separated_input, temp_path + '/separated input.pt')
        torch.save(reconstructed_user_data, temp_path + '/separated res.pt')
    
    print("Done!")

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
    bsz = 16  # batch size
    dc.batch_size = bsz
    bin_num = 1024
    torch.cuda.empty_cache()
    ac.num_bins = bin_num
    res = perform_attacks(data_cfg=dc, attack_cfg=ac, dataset=dataset, fixed_idx=True, use_dp=True, optim=False, sam=sam, filter=True)
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

    original_path = temp_path + '/original data.pt'
    original_res = torch.load(original_path)[:16]
    reconstructed_res = torch.load(temp_path + '/separated res.pt')[:16]
    tp = torchvision.transforms.ToPILImage()

    cols = 8
    rows = 5
    fig = plt.figure(figsize=(16, 8))

    gs = gridspec.GridSpec(rows, cols, height_ratios=[1, 1, 0.1, 1, 1], hspace=0.3)

    for idx in range(len(original_res)):
        r = idx // cols     # 行 0 或 1
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(original_res[idx])
        ax.axis('off')

    for idx in range(len(reconstructed_res)):
        r = idx // cols + 3 
        c = idx % cols
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(tp(reconstructed_res[idx]))
        ax.axis('off')

    plt.figtext(0.5, 0.9, 'Original Samples', ha='center', fontsize=14)
    plt.figtext(0.5, 0.46, 'Reconstructed Samples', ha='center', fontsize=14)
    plt.figtext(0.5, 0.05, f'SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, MSE: {mse}', ha='center', fontsize=14)
    plt.show()
    # plt.savefig("fig/display.png", dpi=600)