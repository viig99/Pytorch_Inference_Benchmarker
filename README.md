# Inference Benchmark


## Benchmark Results
```shell script
# OS: Ubuntu 21.10 x86_64 
# Kernel: 5.15.17-xanmod2 
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz 
# GPU: NVIDIA GeForce RTX 3070 
pytest benchmark.py --benchmark-compare
```
![Benchmark Results](https://github.com/viig99/Pytorch_Inference_Benchmarker/blob/main/image/infer.png?raw=true)

## Dependencies
* [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
* Add TensorRT path to LD_LIBRARY_PATH, LD_LIBRARY_PATH, LD_LIBRARY_PATH
* Link nvinfer `sudo ln -s /media/sdd_1tb/Software_Downloads/TensorRT-8.2.3.0/lib/libnvinfer.so /usr/lib/libnvinfer.so`
* Install pycuda and then `https://github.com/NVIDIA-AI-IOT/torch2trt` with plugins.