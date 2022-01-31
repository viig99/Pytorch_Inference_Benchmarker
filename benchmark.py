import warnings
warnings.filterwarnings("ignore")
import pytest
from utils import load_model, get_tensor, infer, warmup, infer_o, load_trt_model

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp32(benchmark):
    model = load_model()
    tensor = get_tensor()
    warmup(model, tensor)
    benchmark(infer, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16(benchmark):
    model = load_model(fp_16=True)
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_script(benchmark):
    model = load_model(fp_16=True, scripted=True)
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_sf(benchmark):
    model = load_model(fp_16=True, scripted=True, frozen=True)
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_sfo(benchmark):
    model = load_model(fp_16=True, scripted=True, frozen=True, optimized=True)
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_sfo_io(benchmark):
    model = load_model(fp_16=True, scripted=True, frozen=True, optimized=True)
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer_o, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_sfo_channel_last(benchmark):
    model = load_model(fp_16=True, scripted=True, frozen=True, optimized=True, channel_last=True)
    tensor = get_tensor(fp_16=True, channel_last=True)
    warmup(model, tensor)
    benchmark(infer_o, model, tensor)

@pytest.mark.benchmark(
    group="cuda_inference",
    min_rounds=3,
    disable_gc=True,
    warmup=False
)
def test_fp16_trt(benchmark):
    model = load_trt_model()
    tensor = get_tensor(fp_16=True)
    warmup(model, tensor)
    benchmark(infer_o, model, tensor)