"""
Numerically validates aadesh.argmax against torch.argmax.

Pipeline:
  1. Build a module with `aadesh.argmax` on a 1-D f32 tensor, in two
     forms: dim=0/keep_dim=true, and no-dim (flattened argmax).
  2. Lower it: aadesh -> tosa (lower-aadesh-to-tosa), then
     tosa -> linalg -> bufferized memrefs -> llvm. Unlike relu (which
     stays scalar and only needs arith->llvm), argmax is tensor-shaped
     from the start, so this pipeline is substantially longer.
  3. JIT-execute via ExecutionEngine using memref descriptors, and
     compare against torch.argmax across structured edge cases and
     random 1-D tensors.

Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_argmax.py

Requires: pip install torch --index-url https://download.pytorch.org/whl/cpu

NOTE: the pass pipeline below uses standard upstream MLIR pass names for
tosa->linalg->bufferization->llvm. Some of these may need adjusting to
match what's actually registered in your build -- run
`aadesh-opt --help` and grep for `tosa-to-linalg`, `one-shot-bufferize`,
`convert-linalg-to-loops`, etc. if any of these fail to parse.
"""

import ctypes
import sys

import numpy as np
import torch

from mlir_aadesh.ir import Context, Module, Location
from mlir_aadesh.passmanager import PassManager
from mlir_aadesh.execution_engine import ExecutionEngine
from mlir_aadesh.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
from mlir_aadesh.dialects import aadesh

MLIR_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_runner_utils.so"
MLIR_C_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_c_runner_utils.so"

N = 16  # length of the test tensor; keep in sync with the .mlir below


def build_argmax_module(ctx):
    module = Module.parse(
        f"""
        func.func @argmax_dim0_keepdim(%arg0: tensor<{N}xf32>) -> tensor<1xi32>
            attributes {{ llvm.emit_c_interface }} {{
          %0 = aadesh.argmax %arg0 {{dim = 0 : i64, keep_dim = true}}
               : (tensor<{N}xf32>) -> tensor<1xi32>
          return %0 : tensor<1xi32>
        }}

        func.func @argmax_nodim(%arg0: tensor<{N}xf32>) -> tensor<i32>
            attributes {{ llvm.emit_c_interface }} {{
          %0 = aadesh.argmax %arg0 {{keep_dim = true}}
               : (tensor<{N}xf32>) -> tensor<i32>
          return %0 : tensor<i32>
        }}
        """,
        ctx,
    )
    return module


def lower_to_llvm(module):
    # Longer pipeline than relu's, because argmax stays tensor-shaped
    # through tosa and needs a full bufferization + memref lowering
    # before it can reach llvm. Verify each stage against your build;
    # if a pass name below doesn't exist, `aadesh-opt --help` will show
    # you the closest match registered in your tree.
    pm = PassManager.parse(
        "builtin.module("
        "lower-aadesh-to-tosa,"
        "func.func(tosa-to-linalg-named),"
        "func.func(tosa-to-linalg),"
        "func.func(tosa-to-tensor),"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "func.func(convert-linalg-to-loops),"
        "func.func(lower-affine),"
        "convert-scf-to-cf,"
        "expand-strided-metadata,"
        "convert-index-to-llvm,"
        "convert-cf-to-llvm,"
        "finalize-memref-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    pm.run(module.operation)
    return module


def make_engine(module):
    return ExecutionEngine(
        module,
        opt_level=3,
        shared_libs=[MLIR_RUNNER_UTILS, MLIR_C_RUNNER_UTILS],
    )


_descriptor_type_cache = {}


def memref_descriptor_type(rank: int, elem_ctype):
    key = (rank, elem_ctype)
    if key in _descriptor_type_cache:
        return _descriptor_type_cache[key]
    if rank == 0:
        # Rank-0 memrefs carry no shape/strides at all -- matches the
        # dumped IR: !llvm.struct<(ptr, ptr, i64)>.
        fields = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.POINTER(elem_ctype)),
            ("offset", ctypes.c_longlong),
        ]
    else:
        fields = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.POINTER(elem_ctype)),
            ("offset", ctypes.c_longlong),
            ("shape", ctypes.c_longlong * rank),
            ("strides", ctypes.c_longlong * rank),
        ]
    DescType = type(f"MemRefDescriptor{rank}D_{elem_ctype.__name__}",
                     (ctypes.Structure,), {"_fields_": fields})
    _descriptor_type_cache[key] = DescType
    return DescType


def to_input_descriptor(np_arr: np.ndarray):
    rank = np_arr.ndim
    DescType = memref_descriptor_type(rank, ctypes.c_float)
    ptr = np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    strides_elems = tuple(s // np_arr.itemsize for s in np_arr.strides)
    return DescType(
        allocated=None, aligned=ptr, offset=0,
        shape=(ctypes.c_longlong * rank)(*np_arr.shape),
        strides=(ctypes.c_longlong * rank)(*strides_elems),
    )


def run_argmax_dim0(engine, x: np.ndarray) -> int:
    assert x.shape == (N,) and x.dtype == np.float32
    in_desc = to_input_descriptor(x)
    OutDescType = memref_descriptor_type(1, ctypes.c_int32)
    out_desc = OutDescType()

    in_arg = ctypes.pointer(ctypes.pointer(in_desc))
    out_arg = ctypes.pointer(ctypes.pointer(out_desc))

    # output first, matching the ciface signature -- this was backwards before.
    engine.invoke("argmax_dim0_keepdim", out_arg, in_arg)

    return int(out_desc.aligned[out_desc.offset])


def run_argmax_nodim(engine, x: np.ndarray) -> int:
    assert x.shape == (N,) and x.dtype == np.float32
    in_desc = to_input_descriptor(x)
    OutDescType = memref_descriptor_type(0, ctypes.c_int32)
    out_desc = OutDescType()

    in_arg = ctypes.pointer(ctypes.pointer(in_desc))
    out_arg = ctypes.pointer(ctypes.pointer(out_desc))

    engine.invoke("argmax_nodim", out_arg, in_arg)

    return int(out_desc.aligned[out_desc.offset])

def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()

        module = build_argmax_module(ctx)
        assert module.operation.verify()

        lower_to_llvm(module)
        print("=== FINAL IR BEFORE EXECUTION ENGINE ===", flush=True)
        print(module, flush=True)
        engine = make_engine(module)

        rng = np.random.default_rng(0)
        test_cases = []

        # Structured edge cases: max at start / end / middle, and a tie
        # (must resolve to the FIRST occurrence, per torch semantics).
        base = np.zeros(N, dtype=np.float32)
        first = base.copy(); first[0] = 10.0
        last = base.copy(); last[-1] = 10.0
        middle = base.copy(); middle[N // 2] = 10.0
        tie = base.copy(); tie[2] = 5.0; tie[7] = 5.0
        test_cases.extend([first, last, middle, tie])

        for _ in range(50):
            test_cases.append(rng.uniform(-50.0, 50.0, size=N).astype(np.float32))

        max_abs_diff = 0
        mismatches = []

        for x in test_cases:
            torch_x = torch.tensor(x, dtype=torch.float32)

            mlir_dim0 = run_argmax_dim0(engine, x)
            torch_dim0 = torch.argmax(torch_x, dim=0, keepdim=True).item()

            mlir_nodim = run_argmax_nodim(engine, x)
            torch_nodim = torch.argmax(torch_x).item()

            diff = abs(mlir_dim0 - torch_dim0) + abs(mlir_nodim - torch_nodim)
            max_abs_diff = max(max_abs_diff, diff)
            if mlir_dim0 != torch_dim0 or mlir_nodim != torch_nodim:
                mismatches.append((x, mlir_dim0, torch_dim0, mlir_nodim, torch_nodim))

        print(f"Tested {len(test_cases)} tensors.")
        print(f"Max abs diff vs torch.argmax: {max_abs_diff}")

        if mismatches:
            print(f"\n{len(mismatches)} MISMATCH(ES):")
            for x, md, td, mn, tn in mismatches[:20]:
                print(f"  input={x}\n    dim0: aadesh={md} torch={td}"
                      f"    nodim: aadesh={mn} torch={tn}")
            sys.exit(1)
        else:
            print("\nPASSED: aadesh.argmax matches torch.argmax on all test cases.")


if __name__ == "__main__":
    main()