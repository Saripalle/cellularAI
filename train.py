from __future__ import print_function
import os
import sys
import argparse
import numpy as np

import torch
import torchvision

from torch.utils.data import DataLoader
import torch.optim as neuralOptimizer
import torch.optim.lr_scheduler as Scheduler

from core import UNetPreTrained
from core import DiceLoss

from generate_dataset import nuclei_images

from core import essentials

sys.path.append("/Users/sashi/TensorMONK-nuclei/")


class argsTest:
    network = "nuclei_seg_resnet_pretrain"
    tensor_size = 128
    batch_size = 1
    cpus = 8
    gpus = 2
    default_gpu = 0
    learning_rate = 0.001
    optimizer = "adam"


def trainMONK():
    # args = parse_args()
    args = argsTest
    tensor_size = (1, 3, args.tensor_size, args.tensor_size)

    train_dataset = nuclei_images(tensor_size, mode='train')
    valid_dataset = nuclei_images(tensor_size, mode='valid')

    train = DataLoader(train_dataset,
                       shuffle=True,
                       batch_size=args.batch_size,
                       num_workers=args.cpus)
    valid = DataLoader(valid_dataset,
                       shuffle=True,
                       batch_size=args.batch_size,
                       num_workers=args.cpus)

    print("tensor size is {}".format(tensor_size))
    if not os.path.isdir('./models/'):
        os.makedirs("models")
    file_name = args.network
    print("model name is " + file_name)

    embed_net = UNetPreTrained
    embed_net_kwargs = {}
    embed_net_kwargs["tensor_size"] = tensor_size
    embed_net_kwargs["out_channels"] = 64
    embed_net_kwargs["n_classes"] = 2

    """
    This is where the model is built --> core/essentials/makemodel is used
    """
    Model = essentials.MakeModel(file_name, tensor_size, n_labels=2,
                                 embedding_net=embed_net,
                                 loss_net=DiceLoss,
                                 embedding_net_kwargs=embed_net_kwargs,
                                 default_gpu=args.default_gpu,
                                 gpus=args.gpus,
                                 ignore_trained=True)

    # Even though I use pretrained resnet weights, I tune all layers.
    # we can freeze the layers by {layer_name}.grad=False here
    params = list(Model.netEmbedding.parameters()) + \
        list(Model.netLoss.parameters())

    if args.optimizer.lower() == "adam":
        Optimizer = neuralOptimizer.Adam(params)
        scheduler = Scheduler.StepLR(Optimizer, step_size=30, gamma=0.1)
    elif args.optimizer.lower() == "sgd":
        Optimizer = neuralOptimizer.SGD(params, lr=args.learning_rate)
        scheduler = Scheduler.StepLR(Optimizer, step_size=30, gamma=0.1)
    else:
        raise NotImplementedError

    # Usual training
    for Epoch in range(args.epochs):
        if True:
            Model.netEmbedding.train()
            Model.netLoss.train()

        for i, data in enumerate(train):
            Model.meterIterations += 1
            Model.netEmbedding.zero_grad()
            Model.netLoss.zero_grad()
            tensor, targets = data[0], data[1]
            break
            if torch.max(targets) > 1.:
                targets = targets.div(255.)
            if torch.max(tensor) > 1.:
                tensor = tensor.div(255.)
            feature = Model.netEmbedding(tensor)
            feature = torch.softmax(feature, 1)
            # break
            targets = targets.squeeze().float()
            loss = Model.netLoss((feature, targets))[0]
            loss = 1. - loss
            loss.backward()
            Optimizer.step()
            Model.meterLoss.append(loss.data.numpy())
            print("Batch-{}, losses::{:1.3f}".
                  format(i, Model.meterLoss[-1]), end="\r")
            sys.stdout.flush()
            if i % 2 == 0 and i != 0:
                try:
                    final = (feature[:, 0, :, :]-feature[:, 1, :, :]) > 0.0
                    images = torch.cat((tensor[:, 0, :, :].unsqueeze(1),
                                        torch.cat((final.unsqueeze(1).float(),
                                                  targets.unsqueeze(0).unsqueeze(1).float()))))
                    torchvision.utils.save_image(images.cpu().data,
                                                 'train_responses_resnet' +
                                                 '_pretrain.png')
                except IOError:
                    print("failed to write images")
                    pass
        print("train loss ::{:1.3f}".format(np.mean(Model.meterLoss[-i:])))
        scheduler.step()

        # For every 5 epochs, test the model --> Can be convereted to inference later.
        # I prefer to strip down all the functions to bare nn modules for faster inference. However,

        if Epoch % 5 == 0 and Epoch != 0:
            if True:
                Model.netEmbedding.eval()
                Model.netLoss.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for i, data in enumerate(valid):
                    tensor, targets = data[0], data[1]
                    feature = Model.netEmbedding(tensor)
                    feature = torch.softmax(feature, 1)
                    targets = targets.squeeze().float()
                    loss = Model.netLoss((feature.type(torch.FloatTensor),
                                         targets))[0]
                    loss = 1. - loss
                    Model.meterTeAC.append(loss.data.numpy())
                    print("Batch-{}, losses::{:1.3f}".
                          format(i, Model.meterLoss[-1]), end="\r")
                    sys.stdout.flush()
                    if i % 2 == 0 & i != 0:
                        final = (feature[:, 0, :, :]-feature[:, 1, :, :]) > 0.0
                        images = torch.cat((tensor[:, 0, :, :].unsqueeze(1),
                                            torch.cat((final.unsqueeze(1),
                                                      targets.float()))))
                        torchvision.utils.save_image(images.cpu().data,
                                                     'test_responses_resnet' +
                                                     '_pretrain.png')
                print("train loss ::{:1.3f}".
                      format(np.mean(Model.meterTeAC[-i:])))
                Model.netEmbedding.train()
                Model.netLoss.train()
                essentials.SaveModel(Model)


# ========================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(description="nuclei segmentation!!!")
    parser.add_argument("-network", "--network", type=str)
    parser.add_argument("-tS", "--tensor_size", type=int)
    parser.add_argument("-B", "--batch_size", type=int)
    parser.add_argument("-E", "--epochs", type=int)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd"])
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--default_gpu", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--cpus", type=int)
    parser.add_argument("-I", "--ignore_trained", action="store_true",
                        default=False)
    return parser.parse_args()


if __name__ == '__main__':
    Model = trainMONK()  # currently igniores parse_args, and uses argsTest --> This is for running line by line
