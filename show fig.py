import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from DP_model import SGM
from proposed import data_cfg_default
from metrics import *
from file_path import temp_path, fig_path, processed_data_path


device = torch.device('cpu')
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_cfg_default().mean, std=data_cfg_default().std),
    ]
)

sns.set_style('whitegrid')


def show_image(image_path=r'recovered data.pt', col_num=16):
    # idx_list = [10, 14, 6, 15, 5, 4, 8, 9, 0, 3, 12, 2, 11, 13, 1, 7]
    idx_list = [i for i in range(16)]
    res = torch.load(image_path)[:16]
    if len(res) == 0:
        print('No image')
    temp = None
    if type(res) == dict:
        for k, v in res.items():
            if temp is None:
                temp = v
            else:
                res = torch.cat((temp, v), dim=0)
        res = temp
    elif type(res) == list:
        if not type(res[0]) == Image.Image:
            res = torch.stack(res)
            res = res.to('cpu')

    plt.figure(figsize=(6.4, 1.8))

    tp = torchvision.transforms.ToPILImage()
    for idx in range(len(res)):
        plt.subplot(len(res) // col_num + 1, col_num, idx + 1)
        plt.axis('off')
        plt.imshow(tp(res[idx_list[idx]]))
        # plt.imshow(tp(res[idx]))
        # plt.imshow(res[idx_list[idx]])
    plt.subplots_adjust(wspace=0.07, hspace=0)
    plt.show()
    # plt.savefig('evaluation/vision/imagenet when.png', dpi=600, bbox_inches='tight', pad_inches=0)


def compare_grad(mode='grad'):
    if mode == 'grad':
        attack_norm = pd.read_json(r'evaluation\norm\attack norm.json')
        original_norm = pd.read_json(r'evaluation\norm\original norm.json')
        norm = attack_norm.append(original_norm, ignore_index=True)
        norm.loc[norm['type'] == True, ['type']] = 'Compressed attack'
        norm.loc[norm['type'] == False, ['type']] = 'Uncompressed attack'
        norm.loc[norm['type'] == 'original', ['type']] = 'Regular model'
        sns.barplot(data=norm, x='type', y='norm', palette='bright', capsize=0.09)
        plt.xlabel('Model', fontsize=18)
        plt.ylabel('Norm', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=14)
        # plt.show()
        plt.savefig(r'evaluation\norm\norm comparison.png', dpi=600, bbox_inches='tight')
    elif mode == 'value':
        value = pd.read_json(r'evaluation\norm\value.json')
        value['Clipping'] = value['clipping']
        value.loc[value['compressed'] == True, ['compressed']] = 'Compressed attack'
        value.loc[value['compressed'] == False, ['compressed']] = 'Uncompressed attack'
        graph = sns.barplot(data=value, x='compressed', y='value', hue='Clipping', palette='bright', capsize=0.06)
        graph.axhline(0.09, c='black', alpha=0.7)
        graph.text(0.4, 0.05, 'Noise', fontsize=13)
        plt.xlabel('Model', fontsize=18)
        plt.ylabel('Absolute value of gradients', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.show()
        plt.savefig(r'evaluation\norm\value comparison.png', dpi=600, bbox_inches='tight')
    elif mode == 'unit norm':
        res = pd.read_json(r'evaluation/norm/batch size norm.json')
        f= sns.barplot(data=res, x='bin num', y='norm', hue='batch size', palette='bright', capsize=0.09)
        plt.xlabel('Number of units', fontsize=18)
        plt.ylabel('Norm', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.setp(f.get_legend().get_texts(), fontsize='14')
        plt.setp(f.get_legend().get_title(), fontsize='14')
        # plt.show()
        plt.savefig(r'evaluation/norm/unit norm.png', dpi=600, bbox_inches='tight')
    elif mode == 'unit value':
        res = pd.read_json(r'evaluation/norm/batch size norm.json')
        f = sns.barplot(data=res, x='bin num', y='value', hue='batch size', palette='bright', capsize=0.09)
        plt.xlabel('Number of units', fontsize=18)
        plt.ylabel('Norm', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.setp(f.get_legend().get_texts(), fontsize='14')
        plt.setp(f.get_legend().get_title(), fontsize='14')
        # plt.show()
        plt.savefig(r'evaluation/norm/unit value.png', dpi=600, bbox_inches='tight')


def res_metrics(performance_path=r'recovered data.pt', metric='accuracy'):
    sns.set_style('darkgrid')
    _, ax = plt.subplots()
    record = pd.read_json(performance_path)
    if metric == 'accuracy':
        sns.barplot(data=record[record['bin num'] == 1024], x='batch size', y=metric)
    elif metric == 'Meaningless':
        record[metric] = record['meaningless']
        sns.lineplot(data=record, x=np.log2(record['batch size']), y=metric, markers=True, ms=10,
                     ci=None, palette='bright', lw=3, hue='bin num', style='bin num')
        # sns.barplot(data=record, x='bin num', y=metric)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel('Batch size', fontsize=12)
    plt.show()
    # plt.save_fig(r'fig/6.jpg', dpi=600)


def robbing_compare():
    record = pd.read_json(r'performance/robbing/robbing res DP.json')
    '''record1 = pd.read_json(r'performance/robbing/robbing res DP.json')
    record = record.append(record1, ignore_index=True)'''
    record['accuracy'] = record['separated fig'] / record['batch size']
    # sns.barplot(data=record[record['bin num'] == 512], x='batch size', y='accuracy', hue='clipping bound')
    sns.barplot(data=record, x='batch size', y='accuracy', hue='bin num')
    plt.show()
    # plt.save_fig(r'fig/robbing the fed/2.jpg', dpi=600)


def factor_to_performance(factor, metrics, mode='bs'):
    if mode == 'bs':
        record = pd.read_json(r'evaluation/factor/batch size.json')
        f = sns.lineplot(data=record[record['bin num'] == 512], x=factor, y=metrics, hue='Compressed input',
                     palette='bright', style='Compressed input', markers=True, ms=8, sizes=10)
        plt.xlabel('Batch size', fontsize=18)
        # plt.ylabel('PSNR', fontsize=18)
        plt.ylabel('CW-SSIM', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.setp(f.get_legend().get_texts(), fontsize='14')
        plt.setp(f.get_legend().get_title(), fontsize='14')
    elif mode == 'bin num':
        record = pd.read_json(r'evaluation/factor/all.json')
        sns.barplot(data=record[record['batch size'] == 64], x='bin num', y='accuracy', palette='bright', capsize=0.06)
        plt.xlabel('Number of units', fontsize=18)
        plt.ylabel('Separation ratio', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(r'evaluation/factor/bin num accuracy 64bs.png', dpi=600, bbox_inches='tight')
    # plt.show()


def show_image_single(image_path=r'recovered data.pt'):

    res = torch.load(image_path)[2]  # 6:dog; 9:rabbit / 2:dog; 7:rabbit
    # res = torch.load(image_path)[0]

    # res = res * std.view(-1, 1, 1) + mean.view(-1, 1, 1)

    tp = torchvision.transforms.ToPILImage()
    plt.axis('off')
    plt.imshow(tp(res))
    # plt.imshow(res)
    plt.subplots_adjust(wspace=0, hspace=5)
    # plt.show()
    plt.savefig(r'fig/draw_framework/dog res.png', dpi=600, bbox_inches='tight')


def privacy_parameters(factor='epsilon'):
    if factor == 'epsilon':
        res = pd.read_json(r'evaluation/privacy parameters/epsilon plus.json')
        f = sns.lineplot(data=res[res['batch size'] > 8], x='epsilon', y='ssim', hue='batch size', style='batch size',
                     palette='bright', markers=True, ms=10, ci='sd', lw=3)
        plt.xlabel(r'$\varepsilon$', fontsize=18)
        # plt.ylabel('PSNR', fontsize=18)
        plt.ylabel('CW-SSIM', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.setp(f.get_legend().get_texts(), fontsize='14')
        plt.setp(f.get_legend().get_title(), fontsize='14')
        # plt.show()
        plt.savefig(r'evaluation/privacy parameters/epsilon-ssim.png', dpi=600, bbox_inches='tight')
    elif factor == 'clipping bound':
        res = pd.read_json(r'evaluation/privacy parameters/clipping bound.json')
        f = sns.barplot(data=res, x='clipping bound', y='psnr', hue='batch size', palette='bright', capsize=0.09)
        plt.xlabel('Clipping bound', fontsize=18)
        plt.ylabel('PSNR', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.setp(f.get_legend().get_texts(), fontsize='14')
        plt.setp(f.get_legend().get_title(), fontsize='14')
        # plt.show()
        plt.savefig(r'evaluation/privacy parameters/clipping bound psnr.png', dpi=600, bbox_inches='tight')


# show_image_single(temp_path + '/separated res.pt')
show_image(temp_path + '/separated res.pt')
# show_image('intermediate_results/original data.pt')
# compare_grad('unit value')
# factor_to_performance('batch size', 'ssim', 'bs')
# factor_to_performance('bin num', 'accuracy', 'bin num')
# privacy_parameters('epsilon')
