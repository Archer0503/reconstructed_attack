import matplotlib.pyplot as plt
import pandas as pd
import torchvision
import torch
from collections import namedtuple
import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from file_path import dataset_path, sam_path
from PIL import Image


class attack_cfg_default:
    def __init__(self):
        self.type = "analytic"
        self.attack_type = "imprint-readout"
        self.label_strategy = None  # Labels are not actually required for this attack
        self.normalize_gradients = False
        self.sort_by_bias = False
        self.text_strategy = "no-preprocessing"
        self.token_strategy = "decoder-bias"
        self.token_recovery = "from-limited-embedding"
        self.breach_reduction = "weight"
        self.impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")
        self.breach_padding = None
        self.num_bins = 1024
        self.bins_scale_factor = 1  # used in make bins when generating attack model
        self.use_dp = True
        self.compress_image = False
        self.attack_conv_kernel = 2  # num of kernels in the attack structure


class data_cfg_default:
    def __init__(self, compress_image=False):
        self.compress_image = compress_image

        self.batch_size = 32
        self.modality = "vision"
        self.size = (1_281_167,)
        self.classes = 1000
        self.original_shape = (3, 224, 224)
        self.normalize = True
        self.mean = torch.tensor((0.485, 0.456, 0.406))
        self.std = torch.tensor((0.229, 0.224, 0.225))
        if self.compress_image:
            self.image_pool_stride = 2
            self.image_pool_kernel_size = 2
        else:
            self.image_pool_stride = 1
            self.image_pool_kernel_size = 1

        new_shape_x = (self.original_shape[1] - self.image_pool_kernel_size) / self.image_pool_stride + 1
        new_shape_y = (self.original_shape[2] - self.image_pool_kernel_size) / self.image_pool_stride + 1
        self.shape = (3, int(new_shape_x), int(new_shape_y))
        self.model = torchvision.models.resnet50
        self.device = 'cuda'
        self.clipping_bound = 10
        self.epsilon = 10


def get_samples(image_idx=[], device='cuda', dataset='ImageNet', batch_size=16):
    # And your dataset:
    if dataset == 'ImageNet':
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.CenterCrop(224),])
        # torchvision.transforms.ToTensor()
        # torchvision.transforms.Normalize(mean=data_cfg_default().mean, std=data_cfg_default().std)
        dataset = torchvision.datasets.ImageNet(root=dataset_path + "/ImageNet2012",
                                                split="val", transform=transforms)
    elif dataset == 'CIFAR100':
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(32)])
        dataset = torchvision.datasets.CIFAR100(root=dataset_path + "/CIFAR100", transform=transforms)
    elif dataset == 'Caltech256':
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.CenterCrop(224)])
        dataset = torchvision.datasets.Caltech256(root=dataset_path + "/Caltech256", transform=transforms)
    elif dataset == 'Flowers102':
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                     torchvision.transforms.CenterCrop(224)])
        dataset = torchvision.datasets.Flowers102(root=dataset_path + "\Flowers102", transform=transforms)
    else:
        raise ValueError('No such database')

    datapoint, labels = [], []
    if len(image_idx) == 0:
        image_idx = torch.randint(0, 30000, (batch_size,))

    for idx in image_idx:
        img, label = dataset[idx]
        datapoint.append(img)
        labels.append(label)
    labels = torch.tensor(labels).to(device)
    # datapoint = torch.stack(datapoint).to(device)
    return datapoint, labels


def get_std(clipping_bound, epsilon=100, constant=1, aggregation_time=25, max_upload=2, user_num=100,
            min_db_size=1000):
    '''condition = aggregation_time ** 2 - max_upload ** 2 * user_num
    if condition < 0:
        return 0
    else:
        return 2 * constant * clipping_bound * condition ** 0.5 / (min_db_size * user_num * epsilon)'''
    return 2 * constant * clipping_bound / min_db_size / epsilon


def compress_to_original(separated_res, original_size):
    # to expand the res for having the same size as input: 1 -> 2*2
    m = torch.zeros(original_size)
    # upper left corner
    m[:, :, ::2, ::2] = separated_res
    # upper right corner: 1/2 * (left + right)
    m[:, :, ::2, 1:-1:2] = 0.5 * (m[:, :, ::2, :-2:2] + m[:, :, ::2, 2::2])
    # upper right corner, lst column
    m[:, :, ::2, -1] = m[:, :, ::2, -2]
    # bottom left corner: 1/2 * (up + down)
    m[:, :, 1:-1:2, ::2] = 0.5 * (m[:, :, :-2:2, ::2] + m[:, :, 2::2, ::2])
    # bottom right corner: 1/4 * (u + d + l + r)
    m[:, :, 1:-1:2, 1:-1:2] = 0.25 * (m[:, :, :-2:2, 1:-1:2] + m[:, :, 2::2, 1:-1:2] +
                                      m[:, :, 1:-1:2, :-2:2] + m[:, :, 1:-1:2, 2:-1:2])
    # bottom right, last column : 1/3 * (u + d + l) = 1/3 (2l + l) = l
    m[:, :, 1:-1:2, -1] = m[:, :, 1:-1:2, -2]
    # last row
    m[:, :, -1, :] = m[:, :, -2, :]
    return m


def clip_and_perturb(gradients, clipping_bound, epsilon, clip=True, perturb=True):
    # clipping  add noise to gradients
    grad_w = []
    std = get_std(clipping_bound=clipping_bound, epsilon=epsilon)
    for grad in gradients:
        grad_w.append(torch.norm(grad) ** 2)
    grad_norm = sum(grad_w) ** 0.5

    clipping_factor = grad_norm / clipping_bound
    for grad in gradients:
        if grad_norm > clipping_bound and clip:
            grad /= clipping_factor  # clipping
        if perturb:
            noise = torch.normal(mean=0, std=std, size=grad.size(), device=grad.device)
            grad += noise  # add noise
    return gradients


def select_main_object(sam, samples, device='cuda'):
    tt = torchvision.transforms.ToTensor()
    optimal_mask_list = []
    for i, image in enumerate(samples):
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        predictor = SamPredictor(sam)
        predictor.set_image(image_cv2)
        input_point = np.array([[112, 112]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        highest_idx = scores.argmax()
        optimal_mask = masks[highest_idx]
        optimal_mask_list.append(optimal_mask)
        samples[i] = tt(image).to(device)
        samples[i][:, np.invert(optimal_mask)] = 0
    return torch.stack(samples).to(device), optimal_mask_list


if __name__ == '__main__':
    print(get_std(10, 10))
