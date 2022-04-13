# RKNN CGD

考虑将预处理，整合到模型中，作为模型的头部.
优势：预处理 在npu上完成计算.


## 环境说明
- onnx 1.6.0
- tensorflow-gpu 1.11.0
- pytorch 1.6.0
- rknn-toolkit

## rknn
torch -> onnx -> rknn -> 验证测试

## 说明
````python 
ret = rknn.build(do_quantization=True, dataset='./cgd.txt', pre_compile=False)
```
pre_compile=False， 可以在PC运行，调试时，使用.
pre_compile=Ture, 不能在PC运行，放在板子上跑，加载速度快.

- step 1
```bash
python rknn_convert_v2/cls2onnx_with_preprocess.py --weight_path ./rknn_convert_v2/weights/model_5.pth
```

- step 2
```bash
python rknn_convert_v2/onnx2rknn_with_prepocess.py -i ./rknn_convert_v2/outputs/model_5_with_preprocess.onnx -o ./rknn_convert_v2/outputs/model_5_with_preprocess.rknn
```

- step 3
```bash
python rknn_convert/rknn_detect_cgd.py -i ./rknn_convert_v2/outputs/model_5_with_preprocess_debug.rknn -d ./demo/test2.bmp
```

有一个分类分支，可用来观察，提取到的特征是否正确，目前这是最直观的一种验证方式.
