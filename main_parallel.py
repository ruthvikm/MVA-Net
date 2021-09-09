# !/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020/10/01
# @Author  : jet li
# @Email   : jet_uestc@hotmail.com
# @File    : main_parallel.py
# @SoftWare: PyCharm

import time
import os
from PIL import Image
import torch
import torch.optim as om
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


from MVA_Net import MVA_Net

batch_size = 8

train_list = pd.read_csv(
    '/media/jet/970/datasets/drive360_224/Drive360challenge_csvs/drive360challenge_train_360_new.csv')

test_list = pd.read_csv(
    '/media/jet/970/datasets/drive360_224/Drive360challenge_csvs/drive360challenge_validation_360_new.csv')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

radian_variable = 57.295779513

def loader(path):
    data_root = r'/media/jet/970/datasets/'
    path = data_root + r"drive360_224/Drive360Images_surround/" + path
    img_pil = Image.open(path).convert('RGB')
    img_tensor = preprocess(img_pil)
    return img_tensor

class trainset_Normal(Dataset):
    def __init__(self, loader=loader):
        self.loader = loader

    def __getitem__(self, index):
        # remove two sense cross part
        dir_path = []
        for i in range(12):
            dir_path.append(os.path.dirname(train_list[index + i:index + i + 1:1]['cameraFront'].tolist()[0]))
        clean_path = list(set(dir_path))
        if len(clean_path) > 1:
            index = index + 11

        train_rows = train_list[index:index + 7:2]
        cameraRight = torch.stack([self.loader(x) for x in train_rows['cameraRight'].tolist()])
        cameraFront = torch.stack([self.loader(x) for x in train_rows['cameraFront'].tolist()])
        cameraRear = torch.stack([self.loader(x) for x in train_rows['cameraRear'].tolist()])
        cameraLeft = torch.stack([self.loader(x) for x in train_rows['cameraLeft'].tolist()])
        tomtom = [self.loader(x) for x in train_rows['tomtom'].tolist()][-1]
        canSpeed = torch.Tensor(train_rows['canSpeed'].tolist())
        canSteering = torch.div(torch.Tensor(train_rows['canSteering'].tolist()), radian_variable)

        return cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering

    def __len__(self):
        return len(train_list) - 100


class testset_Normal(Dataset):
    def __init__(self, loader=loader):
        self.loader = loader

    def __getitem__(self, index):
        # remove two sense cross part
        dir_path = []
        for i in range(12):
            dir_path.append(os.path.dirname(test_list[index + i:index + i + 1:1]['cameraFront'].tolist()[0]))
        clean_path = list(set(dir_path))
        if len(clean_path) > 1:
            index = index + 11

        test_rows = test_list[index:index + 7:2]
        cameraRight = torch.stack([self.loader(x) for x in test_rows['cameraRight'].tolist()])
        cameraFront = torch.stack([self.loader(x) for x in test_rows['cameraFront'].tolist()])
        cameraRear  = torch.stack([self.loader(x) for x in test_rows['cameraRear'].tolist()])
        cameraLeft  = torch.stack([self.loader(x) for x in test_rows['cameraLeft'].tolist()])
        tomtom      = [self.loader(x) for x in test_rows['tomtom'].tolist()][-1]
        canSpeed = torch.Tensor(test_rows['canSpeed'].tolist())
        canSteering = torch.div(torch.Tensor(test_rows['canSteering'].tolist()), radian_variable)

        return cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering

    def __len__(self):
        return len(test_list) - 100


def setup(rank, world_size):
    # rank: now gpu
    # world_size: how many gpus
    # communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def run_demo(demo_fn, world_size):
    # demo_fn: parallel function eg：train
    # world_size: how many gpus
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    torch.backends.cudnn.benchmark = True

    print(f"Running main(**args) on rank {rank}.")
    setup(rank, world_size)
    # datasets
    train_data = trainset_Normal()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                             sampler=train_sampler, num_workers=12, drop_last=True)

    test_data = testset_Normal()
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                            sampler=test_sampler, num_workers=12, drop_last=True)

    # torch.manual_seed(2020)
    model = MVA_Net(batch_size).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # load weight
    weight_path = 'mva_20210205.pkl'
    if os.path.exists(weight_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(checkpoint)
        torch.distributed.barrier()

    # torch summary
    tensor_board_tag = None
    writer_steer_train_loss = None
    writer_speed_train_loss = None
    writer_steer_eval_loss = None
    writer_speed_eval_loss = None
    if rank == 0:  # only gpu0 can record torch summary
        time_record = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        time_record = time_record + weight_path
        tensor_board_tag = time_record

        writer_steer_train_loss = SummaryWriter('runs/train_steer_mse_loss')
        writer_speed_train_loss = SummaryWriter('runs/train_speed_mse_loss')
        writer_steer_eval_loss = SummaryWriter('runs/eval_steer_mse_loss')
        writer_speed_eval_loss = SummaryWriter('runs/eval_speed_mse_loss')

    # train
    learnRate = 0.0001
    maxEpochs = 100
    min_loss  = 1000
    trainBatch = len(trainloader)
    validBatch = len(testloader)

    opti = om.SGD(model.parameters(), lr=learnRate, momentum=0.9)
    # opti = om.Adam(model.parameters(), lr=learnRate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opti, 'min', verbose=True, patience=3)

    lossFun = torch.nn.MSELoss().to(rank)
    for epoch in range(maxEpochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epochLoss1 = 0.0
        epochLoss2 = 0.0
        for data in tqdm(trainloader):
            opti.zero_grad()
            cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering = data
            cameraRight_input = cameraRight.view(4 * batch_size, 3, 224, 224).to(rank)
            cameraFront_input = cameraFront.view(4 * batch_size, 3, 224, 224).to(rank)
            cameraRear_input = cameraRear.view(4 * batch_size, 3, 224, 224).to(rank)
            cameraLeft_input = cameraLeft.view(4 * batch_size, 3, 224, 224).to(rank)
            tomtom_input = tomtom.view(batch_size, 3, 224, 224).to(rank)
            canSpeed_input = canSpeed.view(4 * batch_size, 1)[3:4 * batch_size + 1:4].to(rank)
            canSteering_input = canSteering.view(4 * batch_size, 1)[3:4 * batch_size + 1:4].to(rank)

            canSpeed_previous = canSpeed.view(4 * batch_size, 1)[2:4 * batch_size - 1:4].to(rank)

            steer_out, speed_out = model(cameraFront_input, cameraRear_input, cameraLeft_input, cameraRight_input,
                                         tomtom_input, canSpeed_previous)

            loss1 = lossFun(steer_out, canSteering_input)
            loss2 = lossFun(speed_out, canSpeed_input)
            loss = (loss1 + loss2)

            epochLoss1 += loss1.item()
            epochLoss2 += loss2.item()
            loss.backward()
            opti.step()

        if rank == 0:
            validLoss1 = 0.0
            validLoss2 = 0.0
            model.eval()
            with torch.no_grad():
                for data in tqdm(testloader):
                    opti.zero_grad()
                    cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering = data
                    cameraRight_input = cameraRight.view(4 * batch_size, 3, 224, 224).to(rank)
                    cameraFront_input = cameraFront.view(4 * batch_size, 3, 224, 224).to(rank)
                    cameraRear_input = cameraRear.view(4 * batch_size, 3, 224, 224).to(rank)
                    cameraLeft_input = cameraLeft.view(4 * batch_size, 3, 224, 224).to(rank)
                    tomtom_input = tomtom.view(batch_size, 3, 224, 224).to(rank)
                    canSpeed_input = canSpeed.view(4 * batch_size, 1)[3:4 * batch_size + 1:4].to(rank)
                    canSteering_input = canSteering.view(4 * batch_size, 1)[3:4 * batch_size + 1:4].to(rank)

                    canSpeed_previous = canSpeed.view(4 * batch_size, 1)[2:4 * batch_size - 1:4].to(rank)

                    steer_out, speed_out = model(cameraFront_input, cameraRear_input, cameraLeft_input,
                                                 cameraRight_input, tomtom_input, canSpeed_previous)

                    validLoss11 = lossFun(steer_out, canSteering_input).item()
                    validLoss1 += validLoss11
                    validLoss22 = lossFun(speed_out, canSpeed_input).item()
                    validLoss2 += validLoss22

            scheduler.step(validLoss1)
            epochLoss1 /= trainBatch  # steer
            epochLoss2 /= trainBatch  # speed
            validLoss1 /= validBatch  # steer
            validLoss2 /= validBatch  # speed

            if validLoss1 < min_loss:
                torch.save(model.state_dict(), weight_path)
                print('save model')
                min_loss = validLoss1
            print(
                "Epoch = {:-3}; steer Train loss = {:.4f}; Speed Train loss = {:.4f}; Steer Validation loss = {:.4f} ;Speed Validation loss = {:.4f}".format(
                    epoch, epochLoss1, epochLoss2, validLoss1, validLoss2))
            writer_steer_train_loss.add_scalar(tensor_board_tag, epochLoss1, global_step=epoch)
            writer_speed_train_loss.add_scalar(tensor_board_tag, epochLoss2, global_step=epoch)
            writer_steer_eval_loss.add_scalar(tensor_board_tag, validLoss1, global_step=epoch)
            writer_speed_eval_loss.add_scalar(tensor_board_tag, validLoss2, global_step=epoch)
    cleanup()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]  = '0,1,2,3'
    n_gpus = torch.cuda.device_count()
    print(f"You have {n_gpus} GPUs.")
    run_demo(main, n_gpus)