"""
Numerically validates aadesh.relu against torch.relu, for the scalar (f32)
path and the tensor (tensor<...xf32>) path across ranks 1D/2D/3D.

Pipeline (tensor, any rank):
  1. Build a module containing `aadesh.relu` on tensor<d0 x d1 x ... x f32>.
  2. Lower it through the full staged pipeline: aadesh -> tosa -> linalg ->
     bufferize (tensor -> memref) -> scf loops -> cf -> llvm dialect.
  3. JIT-execute via ExecutionEngine, passing rank-sized memref descriptors,
     and compare against torch.relu elementwise.

Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_tensor.py
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


def build_relu_tensor_module(ctx, shape):
    """shape: tuple of ints, e.g. (4,), (4, 8), (2, 4, 8)."""
    shape_str = "x".join(str(d) for d in shape) + "xf32"
    tensor_type = f"tensor<{shape_str}>"
    module = Module.parse(
        f"""
        func.func @relu_tensor(%arg0: {tensor_type}) -> {tensor_type}
            attributes {{ llvm.emit_c_interface }} {{
          %0 = aadesh.relu %arg0 : {tensor_type}
          return %0 : {tensor_type}
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


# ---------------------------------------------------------------------------
# Rank-generic memref descriptor handling.
#
# The C ABI for a memref<D0 x D1 x ... x Dn-1 x f32> is:
#   { float* allocated, float* aligned, int64_t offset,
#     int64_t shape[rank], int64_t strides[rank] }
# `shape`/`strides` must be arrays of exactly `rank` elements, so a single
# hardcoded struct only works for rank 1. Build the struct type dynamically
# per-rank instead.
# ---------------------------------------------------------------------------

_descriptor_type_cache = {}


def memref_descriptor_type(rank: int):
    if rank in _descriptor_type_cache:
        return _descriptor_type_cache[rank]
    fields = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.POINTER(ctypes.c_float)),
        ("offset", ctypes.c_longlong),
        ("shape", ctypes.c_longlong * max(rank, 1)),
        ("strides", ctypes.c_longlong * max(rank, 1)),
    ]
    DescType = type(f"MemRefDescriptor{rank}D", (ctypes.Structure,), {"_fields_": fields})
    _descriptor_type_cache[rank] = DescType
    return DescType


def to_memref_descriptor(np_arr: np.ndarray):
    """Build an input memref descriptor matching np_arr's rank/strides."""
    rank = np_arr.ndim
    DescType = memref_descriptor_type(rank)
    ptr = np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    shape = np_arr.shape
    # np strides are in bytes; memref strides are in elements.
    strides_elems = tuple(s // np_arr.itemsize for s in np_arr.strides)
    desc = DescType(
        allocated=None,
        aligned=ptr,
        offset=0,
        shape=(ctypes.c_longlong * rank)(*shape),
        strides=(ctypes.c_longlong * rank)(*strides_elems),
    )
    return desc


def run_relu_tensor(engine, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    shape = x.shape
    rank = x.ndim

    in_desc = to_memref_descriptor(x)
    OutDescType = memref_descriptor_type(rank)
    out_desc = OutDescType()  # zero-initialized; callee overwrites it fully

    in_desc_ptr = ctypes.pointer(in_desc)
    out_desc_ptr = ctypes.pointer(out_desc)

    # invoke()'s packed-args convention needs, for each *pointer-typed*
    # ciface argument, the address of a variable holding that pointer --
    # i.e. a pointer-to-pointer.
    in_arg = ctypes.pointer(in_desc_ptr)
    out_arg = ctypes.pointer(out_desc_ptr)

    engine.invoke("relu_tensor", out_arg, in_arg)

    total_size = int(np.prod(shape))
    flat = np.ctypeslib.as_array(out_desc.aligned, shape=(total_size,)).copy()
    if out_desc.allocated:
        _libc.free(out_desc.allocated)

    # Bufferization allocates a fresh contiguous row-major buffer for the
    # result, so a straight reshape recovers the original shape.
    return flat.reshape(shape)


def validate_tensor(ctx, shape, num_trials=20):
    rng = np.random.default_rng(1)
    max_abs_diff = 0.0
    mismatches = []

    for trial in range(num_trials):
        module = build_relu_tensor_module(ctx, shape)
        assert module.operation.verify()
        lower_tensor_to_llvm(module)
        engine = make_engine(module)

        if trial == 0 and shape == (4,):
            test_tensor = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
        else:
            test_tensor = rng.uniform(-100.0, 100.0, size=shape).astype(np.float32)

        mlir_result = run_relu_tensor(engine, test_tensor)
        torch_result = torch.relu(torch.from_numpy(test_tensor)).numpy()

        diff = np.max(np.abs(mlir_result - torch_result))
        max_abs_diff = max(max_abs_diff, float(diff))
        if diff > 1e-6:
            mismatches.append((test_tensor.copy(), mlir_result.copy(), torch_result.copy(), diff))

    shape_str = "x".join(str(d) for d in shape)
    print(f"[tensor {shape_str}] Tested {num_trials} tensors of shape {shape}.")
    print(f"[tensor {shape_str}] Max abs diff vs torch.relu: {max_abs_diff:.3e}")

    if mismatches:
        print(f"\n[tensor {shape_str}] {len(mismatches)} MISMATCH(ES):")
        for x, mlir_r, torch_r, diff in mismatches[:10]:
            print(f"  input={x}\n  aadesh.relu={mlir_r}\n  torch.relu={torch_r}\n  diff={diff:.3e}\n")
        return False

    print(f"[tensor {shape_str}] PASSED: aadesh.relu matches torch.relu.")
    return True


def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()

        shapes = [
            (4,),        # 1D
            (4, 8),      # 2D
            (2, 4, 8),   # 3D
        ]

        all_ok = True
        for shape in shapes:
            ok = validate_tensor(ctx, shape, num_trials=20)
            all_ok = all_ok and ok

        if not all_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
