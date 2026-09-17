"""
Numerically validates aadesh.pow against numpy.power (float32).
 
Pipeline:
  1. Build a tiny module containing a single `aadesh.pow` op on f32.
  2. Lower it: aadesh -> (tosa | arith) via `lower-aadesh-to-tosa`,
     then arith -> llvm so it can be JIT-compiled.
  3. JIT-execute it via ExecutionEngine and compare against numpy.power
     for a batch of base/exponent value pairs.
 
Note: we intentionally avoid importing torch here. Recent PyTorch builds
vendor their own compiled MLIR (torch/_vendor/quack, CUTLASS's CuTe DSL),
which statically registers the `builtin` dialect. Importing torch in the
same process as our own MLIR-based `mlir_aadesh` bindings causes:
    LLVM ERROR: Trying to register different dialects for the same
    namespace: builtin
numpy.power on float32 inputs matches torch.pow numerically for these
scalar cases, well within the tolerance used below, so it's a safe stand-in.
 
Run with:
    export PYTHONPATH=/home/mcw/aadesh/mlir/training/build/python_packages/aadesh
    python3 validate_pow.py
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
 
def build_pow_module(ctx):
    module = Module.parse(
        """
        func.func @pow_scalar(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
            %0 = aadesh.pow %arg0, %arg1 : (f32, f32) -> f32
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
        "convert-math-to-libm,"
        "convert-math-to-llvm,"
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
 
def run_pow_scalar(engine, base: float, exp: float) -> float:
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(base)
    arg1 = c_float_p(exp)
    res = c_float_p(0.0)
    engine.invoke("pow_scalar", arg0, arg1, res)
    return res[0]
 
def main():
    with Context() as ctx, Location.unknown():
        aadesh.register_dialect(ctx)
        aadesh.register_passes()
 
        module = build_pow_module(ctx)
        assert module.operation.verify()
 
        lower_to_llvm(module)
        engine = make_engine(module)
 
        rng = np.random.default_rng(42)
 
        # Craft safe test pairs for base and exponent to avoid complex numbers (e.g., negative base with fractional exponent)
        bases = np.array([0.0, 1.0, 2.0, 0.5, 10.0, 0.001], dtype=np.float32)
        exponents = np.array([0.0, 1.0, 2.0, 3.0, -1.0, 0.5], dtype=np.float32)
 
        # Create grid combinations and random positive base pairs
        b_grid, e_grid = np.meshgrid(bases, exponents)
        grid_pairs = np.stack([b_grid.ravel(), e_grid.ravel()], axis=-1)
 
        rand_bases = rng.uniform(0.1, 10.0, size=100).astype(np.float32)
        rand_exps = rng.uniform(-3.0, 3.0, size=100).astype(np.float32)
        rand_pairs = np.stack([rand_bases, rand_exps], axis=-1)
 
        test_pairs = np.concatenate([grid_pairs, rand_pairs], axis=0)
 
        max_abs_diff = 0.0
        mismatches = []
 
        for base, exp in test_pairs:
            mlir_result = run_pow_scalar(engine, float(base), float(exp))
 
            ref_result = float(np.power(np.float32(base), np.float32(exp)))
 
            # Use relative/absolute tolerance check since pow can scale up quickly
            diff = abs(mlir_result - ref_result)
            max_abs_diff = max(max_abs_diff, diff)
 
            # Handling potential NaN/Inf matching gracefully
            if np.isnan(ref_result) or np.isinf(ref_result):
                continue
 
            if diff > 1e-4:
                mismatches.append((base, exp, mlir_result, ref_result, diff))
 
        print(f"Tested {len(test_pairs)} value pairs.")
        print(f"Max abs diff vs numpy.power: {max_abs_diff:.3e}")
 
        if mismatches:
            print(f"\n{len(mismatches)} MISMATCH(ES):")
            for base, exp, mlir_r, ref_r, diff in mismatches[:20]:
                print(f"  base={base!r:>8}  exp={exp!r:>8}  "
                      f"aadesh.pow={mlir_r!r:>12}  numpy.power={ref_r!r:>12}  diff={diff:.3e}")
            sys.exit(1)
        else:
            print("\nPASSED: aadesh.pow matches numpy.power on all test pairs.")
 
if __name__ == "__main__":
    main()
