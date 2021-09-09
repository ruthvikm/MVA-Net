#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @Time    : 2020/10/01
# @Author  : jet li
# @Email   : jet_uestc@hotmail.com
# @File    : visual_attention.py
# @SoftWare: PyCharm

# paper reference： https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
# code reference: https://github.com/jacobgil/pytorch-grad-cam/blob/master/gradcam.py

import os
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image


from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from MVA_Net import MVA_Net

train_list = pd.read_csv(
    '/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360challenge_csvs/drive360challenge_train_360_new.csv')

test_list = pd.read_csv(
    '/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360challenge_csvs/drive360challenge_validation_360_new.csv')

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

radian_variable = 57.295779513

def loader(path):
    data_root = r'/home/dl2/jet/ssd/jet/datasets/'
    path = data_root + r"drive360_224/Drive360Images_surround/" + path
    img_pil = Image.open(path).convert('RGB')
    img_tensor = preprocess(img_pil)
    return img_tensor


class testset_Normal(Dataset):
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

        test_rows = test_list[index:index + 7:2]
        cameraRight = torch.stack([self.loader(x) for x in test_rows['cameraRight'].tolist()])
        cameraFront = torch.stack([self.loader(x) for x in test_rows['cameraFront'].tolist()])
        cameraRear  = torch.stack([self.loader(x) for x in test_rows['cameraRear'].tolist()])
        cameraLeft  = torch.stack([self.loader(x) for x in test_rows['cameraLeft'].tolist()])
        tomtom      = [self.loader(x) for x in test_rows['tomtom'].tolist()][-1]
        canSpeed = torch.Tensor(test_rows['canSpeed'].tolist())
        canSteering = torch.div(torch.Tensor(test_rows['canSteering'].tolist()), radian_variable)

        img_path_front = "/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360Images_surround/" + test_rows['cameraFront'].tolist()[0]
        img_path_rear  = "/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360Images_surround/" + test_rows['cameraRear'].tolist()[0]
        img_path_left  = "/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360Images_surround/" + test_rows['cameraLeft'].tolist()[0]
        img_path_right = "/home/dl2/jet/ssd/jet/datasets/drive360_224/Drive360Images_surround/" + test_rows['cameraRight'].tolist()[0]

        return img_path_front, img_path_rear, img_path_left, img_path_right, cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering

    def __len__(self):
        return len(test_list) - 100

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # modify for only one layer
        x = self.model(x)
        x.register_hook(self.save_gradient)
        outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module_front, feature_module_rear, feature_module_left, feature_module_right, target_layer):
        self.model = model
        self.feature_module_front = feature_module_front
        self.feature_module_rear  = feature_module_rear
        self.feature_module_left  = feature_module_left
        self.feature_module_right = feature_module_right
        self.feature_extractor_front = FeatureExtractor(self.feature_module_front, target_layer)
        self.feature_extractor_rear  = FeatureExtractor(self.feature_module_rear,  target_layer)
        self.feature_extractor_left  = FeatureExtractor(self.feature_module_left,  target_layer)
        self.feature_extractor_right = FeatureExtractor(self.feature_module_right, target_layer)

    def get_gradients_front(self):
        return self.feature_extractor_front.gradients
    def get_gradients_rear(self):
        return self.feature_extractor_rear.gradients
    def get_gradients_left(self):
        return self.feature_extractor_left.gradients
    def get_gradients_right(self):
        return self.feature_extractor_right.gradients

    def __call__(self, front_in, rear_in, left_in, right_in):
        target_activations_front = []
        target_activations_rear = []
        target_activations_left = []
        target_activations_right = []

        for name, module in self.model.resnet34_mva._modules.items():  # jet 这里要修改下逻辑
            # every layer process
            if ("front_conv1" in name.lower()) or ("front_bn1" in name.lower()):
                front_in = module(front_in)
            elif ("rear_conv1" in name.lower()) or ("rear_bn1" in name.lower()):
                rear_in = module(rear_in)
            elif ("left_conv1" in name.lower()) or ("left_bn1" in name.lower()):
                left_in = module(left_in)
            elif ("right_conv1" in name.lower()) or ("right_bn1" in name.lower()):
                right_in = module(right_in)
            elif "maxpool" in name.lower():
                front_in = module(front_in)
                rear_in = module(rear_in)
                left_in = module(left_in)
                right_in = module(right_in)
            elif "layer1" in name.lower():
                tmp = torch.stack((front_in, rear_in, left_in, right_in), dim=0)
                tmp = module(tmp)
            elif ("layer2" in name.lower()) or ("layer3" in name.lower()):
                tmp = module(tmp)
            elif "layer4" in name.lower():
                tmp = module(tmp)
                front_in, rear_in, left_in, right_in = torch.chunk(tmp, 4, dim=0)
                front_in = front_in.squeeze(0)
                rear_in = rear_in.squeeze(0)
                left_in = left_in.squeeze(0)
                right_in = right_in.squeeze(0)
            elif "front_conv2" in name.lower():
                # front_in = module(front_in)
                target_activations_front, front_in = self.feature_extractor_front(front_in)
            elif "rear_conv2" in name.lower():
                target_activations_rear, rear_in = self.feature_extractor_rear(rear_in)
            elif "left_conv2" in name.lower():
                target_activations_left, left_in = self.feature_extractor_left(left_in)
            elif "right_conv2" in name.lower():
                target_activations_right, right_in = self.feature_extractor_right(right_in)
            elif "avgpool" in name.lower():
                front_in = module(front_in)
                rear_in = module(rear_in)
                left_in = module(left_in)
                right_in = module(right_in)

                front_in = front_in.view(front_in.size(0), -1)
                rear_in = rear_in.view(rear_in.size(0), -1)
                left_in = left_in.view(left_in.size(0), -1)
                right_in = right_in.view(right_in.size(0), -1)

            elif "front_fc" in name.lower():
                front_in = module(front_in)
            elif "rear_fc" in name.lower():
                rear_in = module(rear_in)
            elif "left_fc" in name.lower():
                left_in = module(left_in)
            elif "right_fc" in name.lower():
                right_in = module(right_in)

        return target_activations_front, target_activations_rear, target_activations_left, target_activations_right, \
               front_in[0], rear_in[0], left_in[0], right_in[0]


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap*0.6 + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module_front, feature_module_rear, feature_module_left, feature_module_right, target_layer, use_cuda, dev):
        self.model = model
        self.dev = dev
        self.feature_module_front = feature_module_front
        self.feature_module_rear = feature_module_rear
        self.feature_module_left = feature_module_left
        self.feature_module_right = feature_module_right
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to(self.dev)

        self.extractor = ModelOutputs(self.model, self.feature_module_front, self.feature_module_rear, self.feature_module_left,self.feature_module_right, target_layer)

    def forward(self, front_in, rear_in, left_in, right_in):
        return self.model(front_in, rear_in, left_in, right_in)

    def __call__(self, front_in, rear_in, left_in, right_in, target_category=None):
        if self.cuda:
            front_in  = front_in.to(self.dev)
            rear_in   = rear_in.to(self.dev)
            left_in   = left_in.to(self.dev)
            right_in  = right_in.to(self.dev)

        features_front, features_rear, features_left, features_right, output_front, output_rear, output_left, output_right = \
            self.extractor(front_in, rear_in, left_in, right_in)

        # front
        target_category_front = np.argmax(output_front.cpu().data.numpy())
        one_hot_front = np.zeros((1, output_front.size()[-1]), dtype=np.float32)
        one_hot_front[0][target_category_front] = 1
        one_hot_front = torch.from_numpy(one_hot_front).requires_grad_(True)
        if self.cuda:
            one_hot_front = one_hot_front.to(self.dev)

        # rear
        target_category_rear = np.argmax(output_rear.cpu().data.numpy())
        one_hot_rear = np.zeros((1, output_rear.size()[-1]), dtype=np.float32)
        one_hot_rear[0][target_category_rear] = 1
        one_hot_rear = torch.from_numpy(one_hot_rear).requires_grad_(True)
        if self.cuda:
            one_hot_rear = one_hot_rear.to(self.dev)

        # left
        target_category_left = np.argmax(output_left.cpu().data.numpy())
        one_hot_left = np.zeros((1, output_left.size()[-1]), dtype=np.float32)
        one_hot_left[0][target_category_left] = 1
        one_hot_left = torch.from_numpy(one_hot_left).requires_grad_(True)
        if self.cuda:
            one_hot_left = one_hot_left.to(self.dev)

        # right
        target_category_right = np.argmax(output_right.cpu().data.numpy())
        one_hot_right = np.zeros((1, output_right.size()[-1]), dtype=np.float32)
        one_hot_right[0][target_category_right] = 1
        one_hot_right = torch.from_numpy(one_hot_right).requires_grad_(True)
        if self.cuda:
            one_hot_right = one_hot_right.to(self.dev)

        output_front = one_hot_front * output_front
        output_rear = one_hot_rear * output_rear
        output_left = one_hot_left * output_left
        output_right = one_hot_right * output_right
        loss = torch.sum(output_front + output_rear + output_left + output_right)
        # loss = output

        self.feature_module_front.zero_grad()
        self.feature_module_rear.zero_grad()
        self.feature_module_left.zero_grad()
        self.feature_module_right.zero_grad()

        self.model.zero_grad()
        loss.backward(retain_graph=True)

        grads_val_front = self.extractor.get_gradients_front()[-1].cpu().data.numpy()
        grads_val_rear  = self.extractor.get_gradients_rear()[-1].cpu().data.numpy()
        grads_val_left  = self.extractor.get_gradients_left()[-1].cpu().data.numpy()
        grads_val_right = self.extractor.get_gradients_right()[-1].cpu().data.numpy()

        features = [features_front, features_rear, features_left, features_right]
        cam_total = []
        for (i, value) in enumerate([grads_val_front, grads_val_rear, grads_val_left, grads_val_right]):
            target = features[i][-1]
            target = target.cpu().data.numpy()[0, :]

            weights = np.mean(value, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, input_img[0].shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cam_total.append(cam)
            pass

        return cam_total[0], cam_total[1], cam_total[2], cam_total[3]


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for MVA_Net and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    batch_size = 1
    dev = torch.device("cuda:0")

    model = MVA_Net(batch_size).to(dev)

    # load weight from multi gpus platform saved weight
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('MVA_20210227.pkl', map_location='cpu').items()})

    grad_cam = GradCam(model=model, feature_module_front=model.resnet34_mva.front_conv2, \
                       feature_module_rear=model.resnet34_mva.rear_conv2, \
                       feature_module_left=model.resnet34_mva.left_conv2, \
                       feature_module_right=model.resnet34_mva.right_conv2, \
                       target_layer=["1"], use_cuda=True, dev=dev)

    test_data = testset_Normal()
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
    run_count = 0
    for data in tqdm(testloader):
        img_path_front, img_path_rear, img_path_left, img_path_right, cameraRight, cameraFront, cameraRear, cameraLeft, tomtom, canSpeed, canSteering = data
        cameraRight_input = cameraRight.view(4*batch_size, 3, 224, 224).to(dev)
        cameraFront_input = cameraFront.view(4*batch_size, 3, 224, 224).to(dev)
        cameraRear_input = cameraRear.view(4*batch_size, 3, 224, 224).to(dev)
        cameraLeft_input = cameraLeft.view(4*batch_size, 3, 224, 224).to(dev)

        img_series = []
        input_img = []
        for (i, value) in enumerate([img_path_front[0], img_path_rear[0], img_path_left[0], img_path_right[0]]):
            img_tmp = cv2.imread(value, 1)
            img_tmp = np.float32(img_tmp) / 255
            img_tmp = img_tmp[:, :, ::-1]
            img_series.append(img_tmp)
            input_img_tensor = preprocess_image(img_tmp)
            input_img.append(input_img_tensor)
            pass

        grayscale_cam_front, grayscale_cam_rear, grayscale_cam_left, grayscale_cam_right= grad_cam(cameraFront_input, cameraRear_input, cameraLeft_input, cameraRight_input, None)  # 这里只有先不用，目前没有办法验证。
        cam = []
        for (i, value) in enumerate([grayscale_cam_front, grayscale_cam_rear, grayscale_cam_left, grayscale_cam_right]):
            value = cv2.resize(value, (img_series[i].shape[1], img_series[i].shape[0]))
            cam_tmp = show_cam_on_image(img_series[i], value)
            cam.append(cam_tmp)

        full_pic = np.full((int(1*224), int(4*224), 3), 255, dtype=np.uint8)

        full_pic[0:224, 0:224, :]   = cam[2]
        full_pic[0:224, 224:448, :] = cam[1]
        full_pic[0:224, 448:672, :] = cam[0]
        full_pic[0:224, 672:896, :] = cam[3]

        run_count += 1
        cv2.imwrite("/home/dl2/jet/ssd/jet/process/mva/cam_full" + str(run_count) + '.jpg', full_pic)
