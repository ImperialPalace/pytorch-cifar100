"""
Copyright 2018-2020  Firmin.Sun (fmsunyh@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# -----------------------------------------------------
# @Time    : 4/19/2021 3:20 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torch.onnx
import onnx
# from net.rknn_model import MobilenetV2_CGD as Model
from net.rknn_model import Resnet_CGD as Model
from models.resnet import resnet50

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-w', '--weight_path',help='the path of weight', default='./weights/cgd_sm_mobilenetv2.pth')
args = parser.parse_args()

class preprocess_conv_layer(nn.Module):
    """docstring for preprocess_conv_layer"""
    #   input_module 为输入模型，即为预导出模型
    #   mean_value 的值可以是 [m1, m2, m3] 或 常数m
    #   std_value 的值可以是 [s1, s2, s3] 或 常数s
    #   BGR2RGB的操作默认为首先执行，既替代的原有操作顺序为 BGR2RGB -> minus mean -> minus std
    #
    #   使用示例伪代码：
    #       from add_preprocess_conv_layer import preprocess_conv_layer
    #       model_A = create_model()
    #       model_output = preprocess_conv_layer(model_A, mean_value, std_value, BGR2RGB)
    #       onnx_export(model_output)

    def __init__(self, input_module, mean_value, std_value, BGR2RGB=False):
        super(preprocess_conv_layer, self).__init__()
        if isinstance(mean_value, int):
            mean_value = [mean_value for i in range(3)]
        if isinstance(std_value, int):
            std_value = [std_value for i in range(3)]

        assert len(mean_value) <= 3, 'mean_value should be int, or list with 3 element'
        assert len(std_value) <= 3, 'std_value should be int, or list with 3 element'

        self.input_module = input_module

        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 3, (1, 1), groups=1, bias=True, stride=(1, 1))

            if BGR2RGB is False:
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 0, :, :] = 1/std_value[0]
                self.conv1.weight[1, 1, :, :] = 1/std_value[1]
                self.conv1.weight[2, 2, :, :] = 1/std_value[2]
            elif BGR2RGB is True:
                self.conv1.weight[:, :, :, :] = 0
                self.conv1.weight[0, 2, :, :] = 1/std_value[0]
                self.conv1.weight[1, 1, :, :] = 1/std_value[1]
                self.conv1.weight[2, 0, :, :] = 1/std_value[2]

            self.conv1.bias[0] = -mean_value[0]/std_value[0]
            self.conv1.bias[1] = -mean_value[1]/std_value[1]
            self.conv1.bias[2] = -mean_value[2]/std_value[2]

        self.conv1.eval()
        # print(self.conv1.weight)
        # print(self.conv1.bias)

    def forward(self, x):
        x = self.conv1(x)
        return self.input_module(x)



def load_model(weight, num_classes=11):
    backbone_type = 'resnet50' # 'mobilenetv2' 'resnet50'
    gd_config = 'SM'
    feature_dim = 512
    model = Model(backbone_type, gd_config, feature_dim, num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(weight), strict=False)
    model.eval()
    return model


def export_pytorch_model(weight):
    net = load_model(weight)

    net.to('cpu')
    net.eval()
    return net

def pytorch2onnx(model, path):
    img = torch.zeros(1, 3, 224,224).to('cpu')

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    torch.onnx.export(model, img, path, opset_version=10, verbose=False)

    # Checks
    onnx_model = onnx.load(path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % path)

def rebuild_model(weight_path, mean_value, std_value, BGR2RGB):

    model = export_pytorch_model(weight_path)

    pre_model = preprocess_conv_layer(model, mean_value, std_value, BGR2RGB)
    return pre_model


if __name__ == '__main__':
    weight_path = args.weight_path

    mean_value = [123.675, 123.675, 123.675]
    std_value = [58.395, 58.395, 58.395]
    BGR2RGB = True


    f = weight_path.replace('.pth', '_pre.onnx')
    save_path = f.replace('weights', 'outputs')

    model = rebuild_model(weight_path, mean_value, std_value, BGR2RGB)
    pytorch2onnx(model, save_path)

'''
python rknn_convert_v2/cls2onnx_with_preprocess.py --weight_path ./rknn_convert_v2/weights/model_5.pth
'''