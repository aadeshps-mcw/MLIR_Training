#!/usr/bin/env bash
# =============================================================================
# relu_to_llvm_pipeline.sh
#
# Full staged lowering pipeline: aadesh.relu (tensor<4xf32>) -> LLVM IR
# Each stage dumps its output to its own .mlir file so every individual
# transformation can be inspected in isolation (see diff commands at the end).
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------
mkdir -p ~/aadesh/mlir/training/examples/relu-to-llvm
cd ~/aadesh/mlir/training/examples/relu-to-llvm

AADESH_OPT=~/aadesh/mlir/training/build/tools/aadesh-opt
MLIR_TRANSLATE=~/aadesh/mlir/llvm-project/build/bin/mlir-translate

cat > 00_input.mlir <<'EOF'
func.func @relu_tensor_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = aadesh.relu %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
EOF

# -----------------------------------------------------------------------------
# Stage 1: aadesh.relu -> tosa.clamp
# (Phase: Custom Dialect -> TOSA)
# -----------------------------------------------------------------------------
$AADESH_OPT 00_input.mlir \
  -lower-aadesh-to-tosa \
  -o 01_after_aadesh_to_tosa.mlir

# -----------------------------------------------------------------------------
# Stage 2: tosa -> linalg (+ arith/tensor) — func-anchored, needs nesting
# (Phase: TOSA -> Mid-level dialects)
# -----------------------------------------------------------------------------
$AADESH_OPT 01_after_aadesh_to_tosa.mlir \
  --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg,tosa-to-arith,tosa-to-tensor))' \
  -o 02_after_tosa_to_linalg.mlir

# -----------------------------------------------------------------------------
# Stage 3: tensor -> memref (bufferization) — module-level
# (Phase: Bufferization)
# -----------------------------------------------------------------------------
$AADESH_OPT 02_after_tosa_to_linalg.mlir \
  -one-shot-bufferize="bufferize-function-boundaries=true" \
  -o 03_after_bufferize.mlir

# -----------------------------------------------------------------------------
# Stage 4: linalg.generic -> scf.for loops — func-anchored, needs nesting
# (Phase: Linalg -> Loops)
# -----------------------------------------------------------------------------
$AADESH_OPT 03_after_bufferize.mlir \
  --pass-pipeline='builtin.module(func.func(convert-linalg-to-loops))' \
  -o 04_after_linalg_to_loops.mlir

# -----------------------------------------------------------------------------
# Stage 5: scf.for -> cf (basic blocks) — func-anchored, needs nesting
# (Phase: Structured -> Unstructured control flow)
# -----------------------------------------------------------------------------
$AADESH_OPT 04_after_linalg_to_loops.mlir \
  --pass-pipeline='builtin.module(func.func(convert-scf-to-cf))' \
  -o 05_after_scf_to_cf.mlir

# -----------------------------------------------------------------------------
# Stage 6: everything remaining -> LLVM dialect — all module-level, flat flags
# (Phase: Final descent into LLVM dialect)
# -----------------------------------------------------------------------------
$AADESH_OPT 05_after_scf_to_cf.mlir \
  -finalize-memref-to-llvm \
  -convert-arith-to-llvm \
  -convert-cf-to-llvm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o 06_after_llvm_dialect.mlir

# -----------------------------------------------------------------------------
# Stage 7: MLIR's LLVM dialect -> real LLVM IR text
# (Phase: Translation — a separate tool, not a pass)
# -----------------------------------------------------------------------------
$MLIR_TRANSLATE 06_after_llvm_dialect.mlir --mlir-to-llvmir \
  -o 07_final.ll

# -----------------------------------------------------------------------------
# Inspect the trail afterward
# -----------------------------------------------------------------------------
echo "=== Diffs between each stage ==="
diff 00_input.mlir 01_after_aadesh_to_tosa.mlir || true
diff 01_after_aadesh_to_tosa.mlir 02_after_tosa_to_linalg.mlir || true
diff 02_after_tosa_to_linalg.mlir 03_after_bufferize.mlir || true
diff 03_after_bufferize.mlir 04_after_linalg_to_loops.mlir || true
diff 04_after_linalg_to_loops.mlir 05_after_scf_to_cf.mlir || true
diff 05_after_scf_to_cf.mlir 06_after_llvm_dialect.mlir || true

echo "=== Final LLVM IR ==="
cat 07_final.ll