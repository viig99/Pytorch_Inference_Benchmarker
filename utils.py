import torchvision.transforms as transforms
from torchvision import models
import torch
from PIL import Image
from torch2trt import torch2trt
import os
import onnxruntime as ort
from scipy.special import softmax

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_model(fp_16=False, scripted=False, frozen=False, optimized=False, channel_last=False):
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.cuda()
    if fp_16:
        model = model.half()
    if channel_last:
        model = model.to(memory_format=torch.channels_last)
    if scripted:
        model = torch.jit.script(model)
    if scripted and frozen:
        model = torch.jit.freeze(model)
    if scripted and frozen and optimized:
        model = torch.jit.optimize_for_inference(model)
    return model

def load_trt_model():
    model = models.resnet18(pretrained=True).cuda().half()
    model.eval()
    data = torch.randn((1, 3, 224, 224)).cuda().half()
    return torch2trt(model, [data], fp16_mode=True, max_batch_size=8)

def load_onnx_model():

    if not os.path.exists("models/resnet_18.onnx"):
        os.makedirs("models", exist_ok=True)
        model = load_model(fp_16=True, scripted=True, frozen=False, optimized=True, channel_last=False)
        torch.onnx.export(model,                                # model being run
                    torch.randn(8, 3, 224, 224).cuda().half(),    # model input (or a tuple for multiple inputs)
                    "models/resnet_18.onnx",
                    export_params=True,
                    do_constant_folding=True,
                    input_names = ['input'],
                    output_names = ['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        })
    ]

    return ort.InferenceSession("models/resnet_18.onnx", providers=providers)

def get_tensor(fp_16=False, channel_last=False, batch_size=8):
    input_image = Image.open("image/cat.jpg")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).cuda()
    input_batch = torch.cat([input_batch for i in range(8)], dim=0)
    if fp_16:
        input_batch = input_batch.half()
    if channel_last:
        input_batch = input_batch.to(memory_format=torch.channels_last)
    return input_batch

@torch.no_grad()
def infer(model, tensor):
    ans = torch.softmax(model(tensor), dim=1)
    torch.cuda.synchronize()
    return ans

@torch.inference_mode()
def infer_o(model, tensor):
    ans = torch.softmax(model(tensor), dim=1)
    torch.cuda.synchronize()
    return ans

def infer_onnx(ort_session, tensor):
    return softmax(ort_session.run(None, {'input': tensor})[0], axis=1)

def warmup(model, tensor, times=3):
    for i in range(times):
        infer(model, tensor)