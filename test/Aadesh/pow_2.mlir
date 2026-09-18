// RUN: aadesh-opt %s -split-input-file --lower-aadesh-pow-to-loops | FileCheck %s

// Case: Float Base with Integer Exponent (Valid for loops)
// CHECK-LABEL: @pow_float_base_int_exp
func.func @pow_float_base_int_exp(%a: f32, %b: i32) -> f32 {
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (f32, i32) -> f32
  return %0 : f32
}

// -----

// Case: Integer Base with Integer Exponent
// CHECK-LABEL: @pow_int_base_int_exp
func.func @pow_int_base_int_exp(%a: i32, %b: i32) -> i32 {
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (i32, i32) -> i32
  return %0 : i32
}

// -----

// Case: Tensor base, tensor exponent, identical shapes -- identity maps.
// CHECK-LABEL: @pow_tensor_tensor_same_shape
func.func @pow_tensor_tensor_same_shape(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map, #map]
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Case: Scalar base broadcast against a tensor exponent -- scalar operand
// gets a 0-result (rank-0) indexing map, not a splat.
// CHECK-LABEL: @pow_scalar_base_tensor_exp
func.func @pow_scalar_base_tensor_exp(%a: f32, %b: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK-NOT: tensor.splat
  // CHECK: linalg.generic
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (f32, tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Case: Genuine shape broadcast between two tensors, e.g. a size-1 dim
// against a size-4 dim, rather than scalar/splat or identical shapes.
// This is the case the old splat-based lowering could not express.
// CHECK-LABEL: @pow_tensor_tensor_broadcast_dim
func.func @pow_tensor_tensor_broadcast_dim(%a: tensor<3x1xf32>, %b: tensor<1x4xi32>) -> tensor<3x4xf32> {
  // CHECK: linalg.generic
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (tensor<3x1xf32>, tensor<1x4xi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// Case: Rank broadcast -- lower-rank tensor operand broadcast against a
// higher-rank result (right-aligned, NumPy-style), matching
// computeBroadcastShape's padding rule in PowOp::verify().
// CHECK-LABEL: @pow_tensor_rank_broadcast
func.func @pow_tensor_rank_broadcast(%a: tensor<3x4xf32>, %b: tensor<4xi32>) -> tensor<3x4xf32> {
  // CHECK: linalg.generic
  // CHECK: scf.for
  %0 = aadesh.pow %a, %b : (tensor<3x4xf32>, tensor<4xi32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}
