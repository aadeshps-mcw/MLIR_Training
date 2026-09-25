// RUN: aadesh-opt %s -split-input-file --lower-aadesh-argmax-to-loops | FileCheck %s

// Case: No dim -- flatten a rank-1 tensor and reduce over all elements.
// Rank-1 input needs no collapse_shape (already flat).
// CHECK-LABEL: @argmax_flat_rank1
func.func @argmax_flat_rank1(%a: tensor<4xf32>) -> tensor<i32> {
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: scf.for
  // CHECK: arith.cmpf ogt
  // CHECK: tensor.insert
  %0 = aadesh.argmax %a : (tensor<4xf32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Case: No dim -- rank-2 input must be collapsed to rank-1 before the
// flattened reduction loop runs.
// CHECK-LABEL: @argmax_flat_rank2
func.func @argmax_flat_rank2(%a: tensor<3x4xf32>) -> tensor<i32> {
  // CHECK: tensor.collapse_shape
  // CHECK-SAME: tensor<3x4xf32> into tensor<12xf32>
  // CHECK: scf.for
  // CHECK: tensor.extract
  // CHECK: arith.cmpf ogt
  // CHECK: tensor.insert
  %0 = aadesh.argmax %a : (tensor<3x4xf32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Case: No dim, integer element type -- uses arith.cmpi sgt, not cmpf.
// CHECK-LABEL: @argmax_flat_int
func.func @argmax_flat_int(%a: tensor<4xi32>) -> tensor<i32> {
  // CHECK: scf.for
  // CHECK: arith.cmpi sgt
  // CHECK-NOT: arith.cmpf
  %0 = aadesh.argmax %a : (tensor<4xi32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Case: dim specified, keep_dim = false -- reduced axis is dropped.
// One outer scf.for over dim 0, inner reduction loop over dim 1.
// CHECK-LABEL: @argmax_dim_no_keep
func.func @argmax_dim_no_keep(%a: tensor<4x4xf32>) -> tensor<4xi32> {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: arith.cmpf ogt
  // CHECK: tensor.insert
  %0 = aadesh.argmax %a {dim = 1 : i64} : (tensor<4x4xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Case: dim specified, keep_dim = true -- reduced axis retained with size 1.
// CHECK-LABEL: @argmax_dim_keep
func.func @argmax_dim_keep(%a: tensor<4x4xf32>) -> tensor<4x1xi32> {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.insert
  %0 = aadesh.argmax %a {dim = 1 : i64, keep_dim = true} : (tensor<4x4xf32>) -> tensor<4x1xi32>
  return %0 : tensor<4x1xi32>
}

// -----

// Case: negative dim -- normalized against input rank (-1 -> dim 1 for a
// rank-2 input), same reduction shape as the positive-dim case above.
// CHECK-LABEL: @argmax_negative_dim
func.func @argmax_negative_dim(%a: tensor<4x4xf32>) -> tensor<4xi32> {
  // CHECK: scf.for
  // CHECK: scf.for
  %0 = aadesh.argmax %a {dim = -1 : i64} : (tensor<4x4xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Case: dim = 0 on a rank-2 input -- outer loop walks the surviving dim
// (dim 1), inner reduction loop walks dim 0.
// CHECK-LABEL: @argmax_dim0
func.func @argmax_dim0(%a: tensor<3x4xf32>) -> tensor<4xi32> {
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.insert
  %0 = aadesh.argmax %a {dim = 0 : i64} : (tensor<3x4xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}
