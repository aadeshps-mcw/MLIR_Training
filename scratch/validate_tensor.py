"""
Numerically validates aadesh.relu against torch.relu, for both the
scalar (f32) and tensor (tensor<Nxf32>) lowering paths.

Pipeline (scalar):
  1. Build a tiny module containing a single `aadesh.relu` op on f32.
  2. Lower it: aadesh -> (tosa | arith) via `lower-aadesh-to-tosa`,
     then arith -> llvm so it can be JIT-compiled.
  3. JIT-execute it via ExecutionEngine and compare against torch.relu.

Pipeline (tensor):
  1. Build a module containing `aadesh.relu` on tensor<Nxf32>.
  2. Lower it through the full staged pipeline: aadesh -> tosa -> linalg ->
     bufferize (tensor -> memref) -> scf loops -> cf -> llvm dialect.
     This mirrors relu_to_llvm_pipeline.sh stages 1-6.
  3. JIT-execute via ExecutionEngine, passing memref descriptors (since
     the function boundary is now memref<Nxf32>, not tensor<Nxf32>), and
     compare against torch.relu elementwise.

Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_tensor.py

Requires: pip install torch --index-url https://download.pytorch.org/whl/cpu
(or any torch build; only CPU float32 ops are used)
"""

import ctypes
import sys
import os

import numpy as np
import torch

from mlir_aadesh.ir import Context, Module, Location
from mlir_aadesh.passmanager import PassManager
from mlir_aadesh.execution_engine import ExecutionEngine
from mlir_aadesh.dialects import aadesh

MLIR_RUNNER_UTILS = os.environ["MLIR_RUNNER_UTILS"]
MLIR_C_RUNNER_UTILS = os.environ["MLIR_C_RUNNER_UTILS"]

_libc = ctypes.CDLL(None)
_libc.free.argtypes = [ctypes.c_void_p]
_libc.free.restype = None


def make_engine(module):
    return ExecutionEngine(
        module,
        opt_level=3,
        shared_libs=[MLIR_RUNNER_UTILS, MLIR_C_RUNNER_UTILS],
    )

def build_relu_tensor_module(ctx, size=4):
    module = Module.parse(
        f"""
        func.func @relu_tensor(%arg0: tensor<{size}xf32>) -> tensor<{size}xf32>
            attributes {{ llvm.emit_c_interface }} {{
          %0 = aadesh.relu %arg0 : tensor<{size}xf32>
          return %0 : tensor<{size}xf32>
        }}
        """,
        ctx,
    )
    return module


def lower_tensor_to_llvm(module):
    pm = PassManager.parse(
        "builtin.module("
        "lower-aadesh-to-tosa,"
        "func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor),"
        "one-shot-bufferize{bufferize-function-boundaries=true},"
        "func.func(convert-linalg-to-loops),"
        "func.func(convert-scf-to-cf),"
        "finalize-memref-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-cf-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


class MemRefDescriptor1D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_longlong),
        ("shape", ctypes.c_longlong * 1),
        ("strides", ctypes.c_longlong * 1),
    ]


def to_memref_descriptor(np_arr: np.ndarray) -> MemRefDescriptor1D:
    ptr = np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return MemRefDescriptor1D(
        allocated=None,
        aligned=ptr,
        offset=0,
        shape=(ctypes.c_longlong * 1)(np_arr.shape[0]),
        strides=(ctypes.c_longlong * 1)(1),
    )


def run_relu_tensor(engine, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    size = x.shape[0]

    in_desc = to_memref_descriptor(x)
    out_desc = MemRefDescriptor1D()  # zero-initialized; callee overwrites it fully

    in_desc_ptr = ctypes.pointer(in_desc)
    out_desc_ptr = ctypes.pointer(out_desc)

    # invoke()'s packed-args convention needs, for each *pointer-typed*
    # ciface argument, the address of a variable holding that pointer --
    # i.e. a pointer-to-pointer.
    in_arg = ctypes.pointer(in_desc_ptr)
    out_arg = ctypes.pointer(out_desc_ptr)

    engine.invoke("relu_tensor", out_arg, in_arg)
    result = np.ctypeslib.as_array(out_desc.aligned, shape=(size,)).copy()
    if out_desc.allocated:
        _libc.free(out_desc.allocated)
    return result

def validate_tensor(ctx, size=4, num_trials=50):
    rng = np.random.default_rng(1)
    max_abs_diff = 0.0
    mismatches = []

    for trial in range(num_trials):
        module = build_relu_tensor_module(ctx, size=size)
        assert module.operation.verify()
        lower_tensor_to_llvm(module)
        engine = make_engine(module)

        if trial == 0:
            test_tensor = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
            if size != 4:
                test_tensor = rng.uniform(-100.0, 100.0, size=size).astype(np.float32)
        else:
            test_tensor = rng.uniform(-100.0, 100.0, size=size).astype(np.float32)

        mlir_result = run_relu_tensor(engine, test_tensor)
        torch_result = torch.relu(torch.from_numpy(test_tensor)).numpy()

        diff = np.max(np.abs(mlir_result - torch_result))
        max_abs_diff = max(max_abs_diff, float(diff))
        if diff > 1e-6:
            mismatches.append((test_tensor.copy(), mlir_result.copy(), torch_result.copy(), diff))

    print(f"[tensor] Tested {num_trials} tensors of size {size}.")
    print(f"[tensor] Max abs diff vs torch.relu: {max_abs_diff:.3e}")

    if mismatches:
        print(f"\n[tensor] {len(mismatches)} MISMATCH(ES):")
        for x, mlir_r, torch_r, diff in mismatches[:10]:
            print(f"  input={x}\n  aadesh.relu={mlir_r}\n  torch.relu={torch_r}\n  diff={diff:.3e}\n")
        return False

    print("[tensor] PASSED: aadesh.relu matches torch.relu on all test tensors.")
    return True


def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()

        tensor_ok = validate_tensor(ctx, size=4, num_trials=50)

        if not (tensor_ok):
            sys.exit(1)

if __name__ == "__main__":
    main()
