"""Script for computing the outputs of CATNetv2 on a dataset.

Version: 2.0.0

Updated by Dimitris Karageorgiou
email: dkarageo@iti.gr
May 2023

Created by Myung-Joon Kwon
mjkwon2021@gmail.com
June 7, 2021
"""
import os
import pathlib
import sys

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.utils.utils import FullModel

from Splicing.data.data_core import SplicingDataset
import seaborn as sns

sns.set_theme()
import matplotlib.pyplot as plt
import click


@click.command()
@click.option("--input_dir",
              type=click.Path(file_okay=False, exists=True, path_type=pathlib.Path),
              default="./input",
              help="Directory where input images are located.")
@click.option("--output_dir",
              type=click.Path(file_okay=False, path_type=pathlib.Path),
              default="./output_pred",
              help="Directory where output masks will be written.")
def main(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path
) -> None:
    ## CHOOSE ##
    args = argparse.Namespace(
        cfg='experiments/CAT_full.yaml',
        opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar',
              'TEST.FLIP_TEST', 'False',
              'TEST.NUM_SAMPLES', '0']
    )
    # args = argparse.Namespace(
    #     cfg='experiments/CAT_DCT_only.yaml',
    #     opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar',
    #           'TEST.FLIP_TEST', 'False',
    #           'TEST.NUM_SAMPLES', '0']
    # )
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    # full model
    test_dataset = SplicingDataset(crop_size=None, grid_crop=True,
                                   blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1,
                                   mode='arbitrary', read_from_jpeg=True,
                                   arbitrary_input_dir=input_dir)
    # DCT stream
    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True,
    #                                 blocks=('DCTvol', 'qtable'), DCT_channels=1,
    #                                 mode='arbitrary', read_from_jpeg=True,
    #                                 arbitrary_input_dir=input_dir)

    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False
    )

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda()

    model = eval('models.' + config.MODEL.NAME + '.get_seg_model')(config)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    print('=> loading model from {}'.format(model_state_file))
    model = FullModel(model, criterion)
    checkpoint = torch.load(model_state_file)
    model.model.load_state_dict(checkpoint['state_dict'])
    print("Epoch: {}".format(checkpoint['epoch']))
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    output_dir.mkdir(parents=True, exist_ok=True)

    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            name = os.path.split(name)[-1]
            return name

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image.cuda()
            label = label.long().cuda()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filepath: pathlib.Path = output_dir / filename

            # plot
            try:
                width = pred.shape[1]  # in pixels
                fig = plt.figure(frameon=False)
                dpi = 40  # fig.dpi
                fig.set_size_inches(width / dpi, ((width * pred.shape[0]) / pred.shape[1]) / dpi)
                sns.heatmap(pred, vmin=0, vmax=1, cbar=False, cmap='gray', )
                plt.axis('off')
                plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig)
            except:
                print(f"Error occurred while saving output. ({get_next_filename(index)})")

            # Clear cache after each sample.
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
