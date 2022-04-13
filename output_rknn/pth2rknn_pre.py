import os

import numpy as np
import cv2
from rknn.api import RKNN
from models.resnet import resnet18
import torch
import torch.nn as nn


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


def export_pytorch_model(weight_path, outpath, mean_value, std_value, BGR2RGB):
    net = resnet18()
    net.load_state_dict(torch.load(weight_path), strict=True)
    net = preprocess_conv_layer(net, mean_value, std_value, BGR2RGB)

    trace_model = torch.jit.trace(net, torch.Tensor(1, 3, 224, 224))
    net.eval()

    trace_model.save(outpath)

def show_outputs(output):
    output_sorted = sorted(output, reverse=True)
    top5_str = '\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def show_perfs(perfs):
    perfs = 'perfs: {}\n'.format(perfs)
    print(perfs)


def softmax(x):
    return np.exp(x)/sum(np.exp(x))


if __name__ == '__main__':

    input_model = './resnet18-11-best.pth'
    model = './resnet18.pth'

    mean_value = [123.675, 123.675, 123.675]
    std_value = [58.395, 58.395, 58.395]
    BGR2RGB = True

    input_size = [[3, 224, 224]]

    export_pytorch_model(input_model, model, mean_value, std_value, BGR2RGB)

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    # force_builtin_perm = True, export NHWC model, use in zere copy
    rknn.config(reorder_channel='0 1 2',target_platform=["rv1126"], batch_size=1, force_builtin_perm=True)
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # pre_compile=True, can't not run in pc
    ret = rknn.build(do_quantization=True, dataset='./datasets.txt', pre_compile=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./resnet_18.rknn')
    if ret != 0:
        print('Export resnet_18.rknn failed!')
        exit(ret)
    print('done')

    # ret = rknn.load_rknn('./resnet_18.rknn')

    # Set inputs

    img = cv2.imread('images/c11_7.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    print(outputs)
    show_outputs(softmax(np.array(outputs[0][0])))
    print('done')

    rknn.release()
