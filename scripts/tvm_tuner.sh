#!/bin/bash

wget -O models/resnet18-v2-7.onnx https://media.githubusercontent.com/media/onnx/models/main/vision/classification/resnet/model/resnet18-v2-7.onnx
tvmc tune --target "cuda" --output models/resnet18-v2-7-autotuner_records.json models/resnet18-v2-7.onnx
tvmc compile --target "cuda" --tuning-records models/resnet18-v2-7-autotuner_records.json --output models/resnet18-v2-7-autotuned-tvm.tar models/resnet18-v2-7.onnx