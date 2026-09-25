// RUN: aadesh-opt %s -lower-aadesh-to-tosa | FileCheck %s

// -----

// CHECK-LABEL: func.func @argmax_dim_keepdim
func.func @argmax_dim_keepdim(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x16xi32> {
  // CHECK: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 1 : i32}
  // CHECK-SAME: (tensor<4x8x16xf32>) -> tensor<4x16xi32>
  // CHECK: %[[SHAPE:.*]] = tosa.const_shape
  // CHECK-SAME: dense<[4, 1, 16]>
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[ARGMAX]], %[[SHAPE]]
  // CHECK-SAME: (tensor<4x16xi32>, !tosa.shape<3>) -> tensor<4x1x16xi32>
  // CHECK: return %[[RESHAPE]]
  %0 = aadesh.argmax %arg0 {dim = 1 : i64, keep_dim = true}
       : (tensor<4x8x16xf32>) -> tensor<4x1x16xi32>
  return %0 : tensor<4x1x16xi32>
}

// -----

// CHECK-LABEL: func.func @argmax_negdim_keepdim
func.func @argmax_negdim_keepdim(%arg0: tensor<4x8x16xf32>) -> tensor<4x8x1xi32> {
  // dim = -1 on a rank-3 input should normalize to axis = 2.
  // CHECK: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 2 : i32}
  // CHECK-SAME: (tensor<4x8x16xf32>) -> tensor<4x8xi32>
  // CHECK: %[[SHAPE:.*]] = tosa.const_shape
  // CHECK-SAME: dense<[4, 8, 1]>
  // CHECK: %[[RESHAPE:.*]] = tosa.reshape %[[ARGMAX]], %[[SHAPE]]
  // CHECK-SAME: (tensor<4x8xi32>, !tosa.shape<3>) -> tensor<4x8x1xi32>
  // CHECK: return %[[RESHAPE]]
  %0 = aadesh.argmax %arg0 {dim = -1 : i64, keep_dim = true}
       : (tensor<4x8x16xf32>) -> tensor<4x8x1xi32>
  return %0 : tensor<4x8x1xi32>
}

// -----

// CHECK-LABEL: func.func @argmax_nodim_keepdim
func.func @argmax_nodim_keepdim(%arg0: tensor<4x8x16xf32>) -> tensor<i32> {
  // No dim -> flatten to 1-D first. keep_dim is ignored per torch semantics
  // when dim is None, so the result stays rank-0 -- no reshape-back needed.
  // CHECK: %[[SHAPE:.*]] = tosa.const_shape
  // CHECK-SAME: dense<512>
  // CHECK: %[[FLAT:.*]] = tosa.reshape %arg0, %[[SHAPE]]
  // CHECK-SAME: (tensor<4x8x16xf32>, !tosa.shape<1>) -> tensor<512xf32>
  // CHECK: %[[ARGMAX:.*]] = tosa.argmax %[[FLAT]] {axis = 0 : i32}
  // CHECK-SAME: (tensor<512xf32>) -> tensor<i32>
  // CHECK: return %[[ARGMAX]]
  %0 = aadesh.argmax %arg0 {keep_dim = true}
       : (tensor<4x8x16xf32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: func.func @argmax_result_type_mismatch
func.func @argmax_result_type_mismatch(%arg0: tensor<4x8xf32>) -> tensor<4x1xi64> {
  // tosa.argmax in this build does NOT enforce i32 output -- it happily
  // emits i64 directly, so no cast is needed here.
  // CHECK: tosa.argmax
  // CHECK-SAME: -> tensor<4xi64>
  // CHECK: tosa.const_shape
  // CHECK: tosa.reshape
  %0 = aadesh.argmax %arg0 {dim = 1 : i64, keep_dim = true}
       : (tensor<4x8xf32>) -> tensor<4x1xi64>
  return %0 : tensor<4x1xi64>
}

// CHECK-LABEL: func.func @argmax_dim_no_keepdim
func.func @argmax_dim_no_keepdim(%arg0: tensor<4x8x16xf32>) -> tensor<4x16xi32> {
  // keep_dim=false (or omitted) -> squeezed shape is already final, no reshape.
  // CHECK: %[[ARGMAX:.*]] = tosa.argmax %arg0 {axis = 1 : i32}
  // CHECK-SAME: (tensor<4x8x16xf32>) -> tensor<4x16xi32>
  // CHECK: return %[[ARGMAX]]
  %0 = aadesh.argmax %arg0 {dim = 1 : i64}
       : (tensor<4x8x16xf32>) -> tensor<4x16xi32>
  return %0 : tensor<4x16xi32>
}