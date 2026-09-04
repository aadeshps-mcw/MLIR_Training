// RUN: aadesh-opt %s -lower-aadesh-to-tosa | FileCheck %s

// =======================================================================
// AddOp lowering
// =======================================================================

// -----

// Scalar float: should lower to arith.addf
// CHECK-LABEL: func.func @add_scalar_f32
func.func @add_scalar_f32(%a: f32, %b: f32) -> f32 {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = arith.addf %arg0, %arg1 : f32
  %0 = aadesh.add %a, %b : f32
  // CHECK: return %[[RES]]
  return %0 : f32
}

// -----

// Scalar int: should lower to arith.addi
// CHECK-LABEL: func.func @add_scalar_i32
func.func @add_scalar_i32(%a: i32, %b: i32) -> i32 {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = arith.addi %arg0, %arg1 : i32
  %0 = aadesh.add %a, %b : i32
  // CHECK: return %[[RES]]
  return %0 : i32
}

// -----

// Scalar float64: should lower to arith.addf
// CHECK-LABEL: func.func @add_scalar_f64
func.func @add_scalar_f64(%a: f64, %b: f64) -> f64 {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = arith.addf %arg0, %arg1 : f64
  %0 = aadesh.add %a, %b : f64
  // CHECK: return %[[RES]]
  return %0 : f64
}

// -----

// Tensor float: should lower to tosa.add
// CHECK-LABEL: func.func @add_tensor_f32
func.func @add_tensor_f32(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %0 = aadesh.add %a, %b : tensor<4xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<4xf32>
}

// -----

// Tensor int: should lower to tosa.add
// CHECK-LABEL: func.func @add_tensor_i32
func.func @add_tensor_i32(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = tosa.add %arg0, %arg1 : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  %0 = aadesh.add %a, %b : tensor<8xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<8xi32>
}

// -----

// Dynamic-shape tensor: should still lower to tosa.add
// CHECK-LABEL: func.func @add_tensor_dynamic
func.func @add_tensor_dynamic(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NOT: aadesh.add
  // CHECK: %[[RES:.*]] = tosa.add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %0 = aadesh.add %a, %b : tensor<?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?xf32>
}

// =======================================================================
// MulOp lowering
// =======================================================================

// -----

// Scalar float: should lower to arith.mulf
// CHECK-LABEL: func.func @mul_scalar_f32
func.func @mul_scalar_f32(%a: f32, %b: f32) -> f32 {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[RES:.*]] = arith.mulf %arg0, %arg1 : f32
  %0 = aadesh.mul %a, %b : f32
  // CHECK: return %[[RES]]
  return %0 : f32
}

// -----

// Scalar int: should lower to arith.muli
// CHECK-LABEL: func.func @mul_scalar_i32
func.func @mul_scalar_i32(%a: i32, %b: i32) -> i32 {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[RES:.*]] = arith.muli %arg0, %arg1 : i32
  %0 = aadesh.mul %a, %b : i32
  // CHECK: return %[[RES]]
  return %0 : i32
}

// -----

// Scalar float64: should lower to arith.mulf
// CHECK-LABEL: func.func @mul_scalar_f64
func.func @mul_scalar_f64(%a: f64, %b: f64) -> f64 {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[RES:.*]] = arith.mulf %arg0, %arg1 : f64
  %0 = aadesh.mul %a, %b : f64
  // CHECK: return %[[RES]]
  return %0 : f64
}

// -----

// Tensor float: should lower to tosa.mul with an i8 shift operand
// CHECK-LABEL: func.func @mul_tensor_f32
func.func @mul_tensor_f32(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[SHIFT:.*]] = "tosa.const"
  // CHECK: %[[RES:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
  %0 = aadesh.mul %a, %b : tensor<4xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<4xf32>
}

// -----

// Tensor int: should lower to tosa.mul with an i8 shift operand
// CHECK-LABEL: func.func @mul_tensor_i32
func.func @mul_tensor_i32(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[SHIFT:.*]] = "tosa.const"
  // CHECK: %[[RES:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
  %0 = aadesh.mul %a, %b : tensor<8xi32>
  // CHECK: return %[[RES]]
  return %0 : tensor<8xi32>
}

// -----

// Dynamic-shape tensor: should still lower to tosa.mul
// CHECK-LABEL: func.func @mul_tensor_dynamic
func.func @mul_tensor_dynamic(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NOT: aadesh.mul
  // CHECK: %[[SHIFT:.*]] = "tosa.const"
  // CHECK: %[[RES:.*]] = tosa.mul %arg0, %arg1, %[[SHIFT]]
  %0 = aadesh.mul %a, %b : tensor<?xf32>
  // CHECK: return %[[RES]]
  return %0 : tensor<?xf32>
}
