#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: pgm

import argparse
import yaml
import torch
import torch.nn as nn
import torch.onnx
import onnx
from rknn.api import RKNN
from net.rknn_model import MobilenetV2_CGD as Model

parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-w', '--weight_path',help='the path of weight', default='./weights/cgd_sm_mobilenetv2.pth')
parser.add_argument('-yaml', '--yaml',help='the path of yaml', default='./rknn_convert_v2/config.yaml')

args = parser.parse_args()


def get_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    return config


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



def load_model(weight, config):

    model = Model(backbone_type=config['onnx']['backbone_type'],
                  gd_config=config['onnx']['gd_config'],
                  feature_dim=config['onnx']['feature_dim'],
                  num_classes=config['onnx']['num_classes']).cuda()
    model.load_state_dict(torch.load(weight), strict=False)
    model.eval()
    return model

def export_pytorch_model(weight, config):
    net = load_model(weight, config)

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


def cls2onnx(config, weight, onnx_path):

    model = export_pytorch_model(weight, config)
    model = preprocess_conv_layer(model,
                                  config['onnx']['mean_value'],
                                  config['onnx']['std_value'],
                                  config['onnx']['BGR2RGB'])
    pytorch2onnx(model, onnx_path)


def onnx2rknn(config, onnx_path, rknn_path):
    rknn = RKNN()

    print('--> config model')
    #不需要做预处理了
    rknn.config(target_platform=["rv1126"], reorder_channel='0 1 2',
                batch_size=1, quantized_dtype='asymmetric_quantized-u8', force_builtin_perm=True)
    # rknn.config(batch_size=1,target_platform=["rk1806", "rk1808", "rk3399pro"], mean_values='0 0 0 255')
    print('done')

    print('--> Loading model: {}'.format(onnx_path))
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=config['rknn']['do_quantization'],
                     dataset=config['rknn']['dataset'],
                     pre_compile=True)  # pre_compile=True
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model:{}'.format(rknn_path))
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # if ret != 0:
    #     print('Init runtime environment failed')
    #     exit(ret)
    # print('done')
    rknn.release()


if __name__ == '__main__':
    config = get_config(args.yaml)

    weight_path = args.weight_path
    onnx_path = weight_path.replace('weights', 'outputs')
    onnx_path = onnx_path.replace('.pth', '.onnx')
    rknn_path = onnx_path.replace('.onnx', '.rknn')


    cls2onnx(config, weight_path, onnx_path)
    onnx2rknn(config, onnx_path, rknn_path)

'''
python rknn_convert_v2/cgd2rknn.py -w ./weights/cgd_sm_mobilenetv2.pth
'''