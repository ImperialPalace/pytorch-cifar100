import numpy as np
import cv2
from rknn.api import RKNN
# import torchvision.models as models
from models.resnet import resnet18
import torch


def export_pytorch_model():
    net = resnet18()
    net.load_state_dict(torch.load('./resnet18-11-best.pth'), strict=True)
    net.eval()

    trace_model = torch.jit.trace(net, torch.Tensor(1,3,224,224))
    trace_model.save('./resnet18.pth')


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

    export_pytorch_model()

    model = './resnet18.pth'

    input_size = [[3, 224, 224]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[123.675, 123.675, 123.675]], std_values=[[58.395, 58.395, 58.395]], reorder_channel='0 1 2',target_platform=["rv1126"],
                batch_size=1, quantized_dtype='asymmetric_quantized-u8')
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
    ret = rknn.build(do_quantization=True, dataset='./datasets.txt')
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
    img = cv2.imread('images/c11_5.jpg')
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
