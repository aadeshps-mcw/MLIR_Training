"""
Numerically validates aadesh.relu against torch.relu.

Pipeline:
  1. Build a tiny module containing a single `aadesh.relu` op on f32.
  2. Lower it: aadesh -> (tosa | arith) via `lower-aadesh-to-tosa`,
     then arith -> llvm so it can be JIT-compiled.
  3. JIT-execute it via ExecutionEngine and compare against torch.relu
     for a batch of random values (positive, negative, zero, edge cases).

Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_relu.py

Requires: pip install torch --index-url https://download.pytorch.org/whl/cpu
(or any torch build; only CPU float32 ops are used)
"""

import ctypes
import sys

import numpy as np
import torch

from mlir_aadesh.ir import Context, Module, Location
from mlir_aadesh.passmanager import PassManager
from mlir_aadesh.execution_engine import ExecutionEngine
from mlir_aadesh.dialects import aadesh

MLIR_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_runner_utils.so"
MLIR_C_RUNNER_UTILS = "/home/mcw/aadesh/mlir/llvm-project/build/lib/libmlir_c_runner_utils.so"


def build_relu_module(ctx):
    module = Module.parse(
        """
        func.func @relu_scalar(%arg0: f32) -> f32 attributes { llvm.emit_c_interface } {
          %0 = aadesh.relu %arg0 : f32
          return %0 : f32
        }
        """,
        ctx,
    )
    return module


def lower_to_llvm(module):
    pm = PassManager.parse(
        "builtin.module("
        "lower-aadesh-to-tosa,"
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


def run_relu_scalar(engine, x: float) -> float:
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(x)
    res = c_float_p(-1.0)
    engine.invoke("relu_scalar", arg0, res)
    return res[0]


def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()

        module = build_relu_module(ctx)

        assert module.operation.verify()

        lower_to_llvm(module)
        engine = make_engine(module)

        rng = np.random.default_rng(0)
        test_values = np.concatenate([
            np.array([0.0, -0.0, 1.0, -1.0, 3.5, -3.5,
                      np.finfo(np.float32).max, -np.finfo(np.float32).max],
                     dtype=np.float32),
            rng.uniform(-100.0, 100.0, size=200).astype(np.float32),
        ])

        max_abs_diff = 0.0
        mismatches = []

        for x in test_values:
            mlir_result = run_relu_scalar(engine, float(x))
            torch_result = torch.relu(torch.tensor(x, dtype=torch.float32)).item()

            diff = abs(mlir_result - torch_result)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-6:
                mismatches.append((x, mlir_result, torch_result, diff))

        print(f"Tested {len(test_values)} values.")
        print(f"Max abs diff vs torch.relu: {max_abs_diff:.3e}")

        if mismatches:
            print(f"\n{len(mismatches)} MISMATCH(ES):")
            for x, mlir_r, torch_r, diff in mismatches[:20]:
                print(f"  input={x!r:>15}  aadesh.relu={mlir_r!r:>15}  "
                      f"torch.relu={torch_r!r:>15}  diff={diff:.3e}")
            sys.exit(1)
        else:
            print("\nPASSED: aadesh.relu matches torch.relu on all test values.")


if __name__ == "__main__":
    main()
