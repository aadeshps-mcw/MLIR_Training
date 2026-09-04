// RUN: aadesh-opt %s -lower-aadesh-to-tosa | FileCheck %s

// -----

// Tensor + float: should lower to tosa.clamp
// CHECK-LABEL: func.func @relu_tensor_f32
func.func @relu_tensor_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: aadesh.relu
  // CHECK: %[[RES:.*]] = tosa.clamp %arg0 {
  // CHECK-SAME: max_val = 0x7F800000 : f32
  // CHECK-SAME: min_val = 0.000000e+00 : f32
  // CHECK-SAME: } : (tensor<4xf32>) -> tensor<4xf32>
  %0 = aadesh.relu %arg0 : tensor<4xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<4xf32>
}

// -----

// Tensor + int: should lower to tosa.clamp
// CHECK-LABEL: func.func @relu_tensor_i32
func.func @relu_tensor_i32(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK-NOT: aadesh.relu
  // CHECK: %[[RES:.*]] = tosa.clamp %arg0 {
  // CHECK-SAME: max_val = 2147483647 : i32
  // CHECK-SAME: min_val = 0 : i32
  // CHECK-SAME: } : (tensor<4xi32>) -> tensor<4xi32>
  %0 = aadesh.relu %arg0 : tensor<4xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<4xi32>
}

// -----

// Scalar + float
// CHECK-LABEL: func.func @relu_scalar_f32
func.func @relu_scalar_f32(%arg0: f32) -> f32 {
  // CHECK-NOT: aadesh.relu
  // CHECK-NOT: tosa.clamp
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[RES:.*]] = arith.maximumf %arg0, %[[ZERO]] : f32
  %0 = aadesh.relu %arg0 : f32
  // CHECK: return %[[RES]]
  return %0 : f32
}

// -----

// Scalar + int
// CHECK-LABEL: func.func @relu_scalar_i32
func.func @relu_scalar_i32(%arg0: i32) -> i32 {
  // CHECK-NOT: aadesh.relu
  // CHECK-NOT: tosa.clamp
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: %[[RES:.*]] = arith.maxsi %arg0, %[[ZERO]] : i32
  %0 = aadesh.relu %arg0 : i32
  // CHECK: return %[[RES]]
  return %0 : i32
}

// -----

// Unsigned scalar: should fold to identity
// CHECK-LABEL: func.func @relu_scalar_ui32
// CHECK-SAME: (%[[ARG0:.*]]: ui32) -> ui32
func.func @relu_scalar_ui32(%arg0: ui32) -> ui32 {
  // CHECK-NOT: arith.constant
  // CHECK-NOT: arith.maxsi
  // CHECK: return %[[ARG0]]
  %0 = aadesh.relu %arg0 : ui32
  return %0 : ui32
}

// -----

// Unsigned tensor: should fold to identity
// CHECK-LABEL: func.func @relu_tensor_ui32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xui32>) -> tensor<4xui32>
func.func @relu_tensor_ui32(%arg0: tensor<4xui32>) -> tensor<4xui32> {
  // CHECK-NOT: tosa.clamp
  // CHECK: return %[[ARG0]]
  %0 = aadesh.relu %arg0 : tensor<4xui32>
  return %0 : tensor<4xui32>
}
