# MLIR_Training


`mlir-train` is a hands-on starter repository for training reference and learning MLIR (Multi-Level Intermediate Representation) compiler design, built on top of LLVM/MLIR. The repo demonstrates a basic custom out-of-tree MLIR dialect built from scratch using TableGen (`.td`), generating C++ bindings, implementing lowering, implementing custom ops and attributes, creating a standalone compiler tool (`aadesh-opt`), and writing `.mlir` tests and other MLIR concepts.

---

## Overview

The repo defines a custom dialect, **`aadesh`**, and shows the full lifecycle of an MLIR dialect:

1. Defining ops declaratively in TableGen (`AadeshOps.td`).
2. Generating C++ dialect/op boilerplate via `mlir-tablegen`.
3. Implementing a lowering pass that converts `aadesh` ops into standard MLIR dialects (`arith`, `tosa`).
4. Building a custom `mlir-opt`-style CLI driver, `aadesh-opt`, to run and test the pipeline.
5. Verifying correctness with `FileCheck`-based `.mlir` regression tests, run through `lit`.

---

## Supported Operations

The `aadesh` dialect currently defines three operations:

| Op | Description | Operand types supported |
|---|---|---|
| `aadesh.add` | Elementwise addition of two operands | Scalar float (`f32`, `f64`), scalar integer (`i32`, `si32`, `ui32`), and ranked tensors of the above (static or dynamic shape) |
| `aadesh.mul` | Elementwise multiplication of two operands | Same type coverage as `aadesh.add` |
| `aadesh.relu` | Rectified linear unit (single operand) | Same type coverage as `aadesh.add`/`aadesh.mul` |

Example usage in `.mlir`:

```mlir
func.func @example(%a: f32, %b: f32) -> f32 {
  %0 = aadesh.add %a, %b : f32
  %1 = aadesh.relu %0 : f32
  return %1 : f32
}
```

---

## Lowering: `-lower-aadesh-to-tosa`

The `LowerAadeshToTosa` pass (registered as the `-lower-aadesh-to-tosa` CLI flag) lowers every `aadesh` op into equivalent operations in the `arith` and `tosa` dialects, using MLIR's dialect conversion infrastructure (`OpConversionPattern` + `ConversionTarget` + `applyPartialConversion`).

The destination dialect is chosen based on operand type:

| Source op | Operand kind | Lowers to |
|---|---|---|
| `aadesh.add` | Scalar float | `arith.addf` |
| `aadesh.add` | Scalar integer | `arith.addi` |
| `aadesh.add` | Ranked tensor (float or int) | `tosa.add` |
| `aadesh.mul` | Scalar float | `arith.mulf` |
| `aadesh.mul` | Scalar integer | `arith.muli` |
| `aadesh.mul` | Ranked tensor (float or int) | `tosa.mul` (with a generated `tensor<1xi8>` zero-shift constant) |
| `aadesh.relu` | Scalar float | `arith.constant 0` + `arith.maximumf` |
| `aadesh.relu` | Scalar signed/signless integer | `arith.constant 0` + `arith.maxsi` |
| `aadesh.relu` | Scalar unsigned integer | Folded away to an identity (relu is a no-op on unsigned values) |
| `aadesh.relu` | Ranked tensor, float | `tosa.clamp` with `min_val = 0.0`, `max_val = +inf` |
| `aadesh.relu` | Ranked tensor, signed/signless int | `tosa.clamp` with `min_val = 0`, `max_val = <type max>` |
| `aadesh.relu` | Ranked tensor, unsigned int | Folded away to an identity |

**Note:** unranked tensors are rejected by all three patterns (`notifyMatchFailure`), since TOSA requires statically known rank.

Run the lowering directly with `aadesh-opt`:

```bash
aadesh-opt input.mlir -lower-aadesh-to-tosa
```

---

## Building

This is an out-of-tree MLIR project — it links against a pre-built LLVM/MLIR install rather than living inside `llvm-project/`.

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit
ninja aadesh-opt
```

This produces the `aadesh-opt` binary under `build/tools/`, which behaves like `mlir-opt` but with the `aadesh` dialect and its lowering pass registered.

---

## Running the Tests

Regression tests live under `test/` and use `FileCheck`-annotated `.mlir` files, run via `lit`.

```bash
cd build
ninja check-aadesh
```

This builds `aadesh-opt` (if needed) and runs every `.mlir` test under `test/`, reporting pass/fail/unresolved counts.

To run a single test file directly (useful when iterating on a specific pattern):

```bash
aadesh-opt test/Aadesh/relu.mlir -lower-aadesh-to-tosa | FileCheck test/Aadesh/relu.mlir
```

---

## Repository Layout

```
include/Aadesh/       # Dialect/op/pass TableGen (.td) and generated headers
lib/Dialect/           # Dialect and op C++ implementation
lib/Conversion/        # Lowering pass implementation (AadeshToTosa)
tools/                 # aadesh-opt driver
test/                  # FileCheck-based .mlir regression tests
```
