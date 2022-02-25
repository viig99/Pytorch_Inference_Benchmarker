from tvm.driver.tvmc.model import TVMCPackage
from tvm.contrib import graph_executor as runtime
from tvm import rpc
import numpy as np

def load_tvm_model():
    autotuned_tm = TVMCPackage("models/resnet18-v2-7-tvm.tar")
    session = rpc.LocalSession()
    lib = session.load_module(autotuned_tm.lib_path)
    dev = session.cuda()
    module = runtime.create(autotuned_tm.graph, lib, dev)
    module.load_params(autotuned_tm.params)
    return module

def infer_tvm(module, tensor):
    ans = None
    for i in range(tensor.shape[0]):
        module.set_input("data", np.expand_dims(tensor[i, :, :, :], 0))
        module.run()
        ans = module.get_output(0).numpy()
    return ans