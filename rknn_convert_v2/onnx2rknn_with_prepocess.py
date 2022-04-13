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
# @Time    : 3/29/2021 4:27 PM
# @Author  : Firmin.Sun (fmsunyh@gmail.com)
# @Software: ZJ_AI
# -----------------------------------------------------
# -*- coding: utf-8 -*-

""""
将onnx模型转换为rknn模型
"""

from rknn.api import RKNN
import argparse
parser = argparse.ArgumentParser(description='Get the data info')
parser.add_argument('-i', '--onnx_model',help='input: the path onnx model', default='')
parser.add_argument('-o', '--rknn_model',help='output: the path rkn model', default='')
args = parser.parse_args()


if __name__ == '__main__':
    # ONNX_MODEL = './outputs/cgd_sm_mobilenetv2_with_preprocess.onnx'
    # RKNN_MODEL = './outputs/cgd_sm_mobilenetv2_with_preprocess.rknn'

    ONNX_MODEL = args.onnx_model
    RKNN_MODEL = args.rknn_model

    # Create RKNN object
    rknn = RKNN()


    print('--> config model')
    #不需要做预处理了
    rknn.config(target_platform=["rv1126"], reorder_channel='0 1 2',
                batch_size=1, force_builtin_perm=True)
    # rknn.config(batch_size=1,target_platform=["rk1806", "rk1808", "rk3399pro"], mean_values='0 0 0 255')
    print('done')

    # Load tensorflow model
    print('--> Loading model: {}'.format(ONNX_MODEL))
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load yolov5s failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./quantization_data/c11_quantization_data.txt', pre_compile=False)  # pre_compile=True
    # ret = rknn.build(do_quantization=True)  # pre_compile=True

    if ret != 0:
        print('Build yolov5s failed!')
        exit(ret)
    print('done')


    # Export rknn model
    print('--> Export RKNN model:{}'.format(RKNN_MODEL))
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export yolov5s.rknn failed!')
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')


    rknn.release()

'''
cd cgd_for_rknn
python rknn_convert_v2/onnx2rknn_with_prepocess.py -i ./rknn_convert_v2/outputs/model_5_with_preprocess.onnx -o ./rknn_convert_v2/outputs/model_5_with_preprocess.rknn
'''