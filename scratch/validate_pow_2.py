"""
Numerically validates aadesh.pow's exp-by-squaring loop lowering
(lower-aadesh-pow-to-loops) against numpy.power, for:
  - scalar float base / integer exponent (the exp-by-squaring path itself)
  - tensor-tensor broadcasting (same shape, size-1 dims, rank mismatch)
  - scalar-tensor broadcasting

Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_pow_loops.py
"""

import ctypes
import sys

import numpy as np

from mlir_aadesh.ir import Context, Module, Location
from mlir_aadesh.passmanager import PassManager
from mlir_aadesh.execution_engine import ExecutionEngine
from mlir_aadesh.dialects import aadesh

MLIR_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_runner_utils.so"
MLIR_C_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_c_runner_utils.so"

MODULE_SRC = """
func.func @pow_scalar_int_exp(%base: f32, %exp: i32) -> f32
    attributes { llvm.emit_c_interface } {
  %0 = aadesh.pow %base, %exp : (f32, i32) -> f32
  return %0 : f32
}

func.func @pow_tensor_same_shape(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32>
    attributes { llvm.emit_c_interface } {
  %0 = aadesh.pow %a, %b : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @pow_scalar_base_tensor_exp(%base: f32, %exp: tensor<4xi32>) -> tensor<4xf32>
    attributes { llvm.emit_c_interface } {
  %0 = aadesh.pow %base, %exp : (f32, tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @pow_broadcast_dim(%a: tensor<3x1xf32>, %b: tensor<1x4xi32>) -> tensor<3x4xf32>
    attributes { llvm.emit_c_interface } {
  %0 = aadesh.pow %a, %b : (tensor<3x1xf32>, tensor<1x4xi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

func.func @pow_rank_broadcast(%a: tensor<3x4xf32>, %b: tensor<4xi32>) -> tensor<3x4xf32>
    attributes { llvm.emit_c_interface } {
  %0 = aadesh.pow %a, %b : (tensor<3x4xf32>, tensor<4xi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
"""


def build_module(ctx):
    return Module.parse(MODULE_SRC, ctx)


def lower_to_llvm(module):
    # Tensor-valued aadesh.pow lowers to linalg.generic; that needs
    # bufferizing before it can reach LLVM. buffer-results-to-out-params
    # turns each tensor-returning func into one taking a pre-allocated
    # memref out-param -- avoids ever having to marshal a *returned*
    # memref back across the Python/ctypes boundary.
    pm = PassManager.parse(
        "builtin.module("
        "lower-aadesh-pow-to-loops,"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "convert-linalg-to-loops,"
        "lower-affine,"
        "convert-scf-to-cf,"
        "expand-strided-metadata,"
        "finalize-memref-to-llvm,"
        "convert-cf-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


def make_engine(module):
    return ExecutionEngine(
        module, opt_level=3, shared_libs=[MLIR_RUNNER_UTILS, MLIR_C_RUNNER_UTILS]
    )


def _memref_descriptor_type(rank, ctype):
    class MemRefDescriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.POINTER(ctype)),
            ("aligned", ctypes.POINTER(ctype)),
            ("offset", ctypes.c_int64),
            ("shape", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRefDescriptor


def as_memref_arg(arr: np.ndarray, ctype):
    """Wrap a numpy array as a (pointer-to-pointer-to-descriptor), matching
    the packed ExecutionEngine.invoke ABI for a memref-typed argument (an
    emit_c_interface memref arg is itself passed by pointer, so the packed
    args array needs a pointer to *that* pointer)."""
    arr = np.ascontiguousarray(arr)
    rank = arr.ndim
    Desc = _memref_descriptor_type(rank, ctype)
    data_ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    elem_strides = tuple(s // arr.itemsize for s in arr.strides)
    desc = Desc(
        allocated=data_ptr,
        aligned=data_ptr,
        offset=0,
        shape=(ctypes.c_int64 * rank)(*arr.shape),
        strides=(ctypes.c_int64 * rank)(*elem_strides),
    )
    desc_ptr = ctypes.pointer(desc)
    # Return the double pointer plus everything that must stay alive
    # (desc and arr) for the duration of the call.
    return ctypes.pointer(desc_ptr), (arr, desc, desc_ptr)


def run_scalar_int_exp(engine, base: float, exp: int) -> float:
    c_float_p = ctypes.c_float * 1
    c_int_p = ctypes.c_int32 * 1
    arg0 = c_float_p(base)
    arg1 = c_int_p(exp)
    res = c_float_p(0.0)
    engine.invoke("pow_scalar_int_exp", arg0, arg1, res)
    return res[0]


def run_tensor_case(engine, fn_name, out_shape, base, exp_i32, base_is_scalar=False):
    rank = len(out_shape)
    Desc = _memref_descriptor_type(rank, ctypes.c_float)
    result_desc = Desc()
    result_desc_ptr = ctypes.pointer(result_desc)   # pointer-to-descriptor

    if base_is_scalar:
        c_float_p = ctypes.c_float * 1
        base_arg = c_float_p(base)
    else:
        base_arg, _keep1 = as_memref_arg(base.astype(np.float32), ctypes.c_float)
    exp_arg, _keep2 = as_memref_arg(exp_i32.astype(np.int32), ctypes.c_int32)
    engine.invoke(fn_name, ctypes.pointer(result_desc_ptr), base_arg, exp_arg)

    shape = tuple(result_desc.shape[i] for i in range(rank))
    strides_elems = tuple(result_desc.strides[i] for i in range(rank))
    itemsize = ctypes.sizeof(ctypes.c_float)
    strides_bytes = tuple(s * itemsize for s in strides_elems)
    buf_ptr = ctypes.cast(result_desc.aligned, ctypes.POINTER(ctypes.c_float))
    flat = np.ctypeslib.as_array(buf_ptr, shape=(int(np.prod(shape)),))
    arr = np.lib.stride_tricks.as_strided(flat, shape=shape, strides=strides_bytes)
    return arr.copy()


def check(name, got, ref, tol=1e-4):
    diff = np.max(np.abs(got - ref))
    ok = diff <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: max abs diff = {diff:.3e}")
    return ok


def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()

        module = build_module(ctx)
        assert module.operation.verify()
        lower_to_llvm(module)
        engine = make_engine(module)

        all_ok = True

        # --- Scalar float base, non-negative int exponent: exercises the
        # exp-by-squaring loop body directly. ---
        bases = np.array([0.0, 1.0, 2.0, 0.5, 10.0, 3.3], dtype=np.float32)
        exps = np.array([0, 1, 2, 3, 7, 15], dtype=np.int32)
        for b in bases:
            for e in exps:
                got = run_scalar_int_exp(engine, float(b), int(e))
                ref = float(np.power(np.float32(b), np.int64(e)).astype(np.float32))
                if abs(got - ref) > 1e-2 * max(1.0, abs(ref)):
                    print(f"[FAIL] scalar_int_exp base={b} exp={e} "
                          f"got={got} ref={ref}")
                    all_ok = False
        print("[INFO] scalar_int_exp: swept", len(bases) * len(exps), "pairs")

        # --- Tensor-tensor, identical shapes. ---
        a = np.array([2.0, 3.0, 0.5, 10.0], dtype=np.float32)
        e = np.array([3, 2, 4, 0], dtype=np.int32)
        got = run_tensor_case(engine, "pow_tensor_same_shape", (4,), a, e)
        ref = np.power(a, e.astype(np.int64)).astype(np.float32)
        all_ok &= check("tensor_same_shape", got, ref)

        # --- Scalar base broadcast against a tensor exponent. ---
        e = np.array([0, 1, 2, 3], dtype=np.int32)
        got = run_tensor_case(engine, "pow_scalar_base_tensor_exp", (4,),
                               2.0, e, base_is_scalar=True)
        ref = np.power(np.float32(2.0), e.astype(np.int64)).astype(np.float32)
        all_ok &= check("scalar_base_tensor_exp", got, ref)

        # --- Genuine broadcast: (3,1) x (1,4) -> (3,4). ---
        a = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)          # (3,1)
        e = np.array([[0, 1, 2, 3]], dtype=np.int32)                    # (1,4)
        got = run_tensor_case(engine, "pow_broadcast_dim", (3, 4), a, e)
        ref = np.power(a, e.astype(np.int64)).astype(np.float32)        # numpy broadcasts natively
        all_ok &= check("broadcast_dim", got, ref)

        # --- Rank broadcast: (3,4) x (4,) -> (3,4). ---
        a = np.arange(1, 13, dtype=np.float32).reshape(3, 4) * 0.5
        e = np.array([0, 1, 2, 3], dtype=np.int32)
        got = run_tensor_case(engine, "pow_rank_broadcast", (3, 4), a, e)
        ref = np.power(a, e.astype(np.int64)).astype(np.float32)
        all_ok &= check("rank_broadcast", got, ref)

        if not all_ok:
            sys.exit(1)
        print("\nPASSED: aadesh.pow (loops lowering) matches numpy.power "
              "on all scalar and broadcasting test cases.")


if __name__ == "__main__":
    main()
