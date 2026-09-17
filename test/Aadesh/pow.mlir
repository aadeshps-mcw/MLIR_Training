// RUN: aadesh-opt %s -split-input-file --lower-aadesh-to-tosa | FileCheck %s
// RUN: aadesh-opt %s -split-input-file --lower-aadesh-to-tosa --verify-diagnostics

// ==========================================
// SHAPE COMBINATIONS — float-only (Cases 1 - 4)
// ==========================================

// Case 1: Scalar x Scalar
// CHECK-LABEL: @pow_scalar_float_float
func.func @pow_scalar_float_float(%a: f32, %b: f32) -> f32 {
  // CHECK: math.powf
  // CHECK-NOT: linalg.generic
  %0 = aadesh.pow %a, %b : (f32, f32) -> f32
  return %0 : f32
}

// -----

// Case 2: Tensor x Scalar (float exponent)
// CHECK-LABEL: @pow_tensor_scalar_exp
func.func @pow_tensor_scalar_exp(%a: tensor<4xf32>, %b: f32) -> tensor<4xf32> {
  // CHECK: tensor.splat
  // CHECK: linalg.generic
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (tensor<4xf32>, f32) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Case 2 (variant): Scalar base x Tensor exponent, both float
// CHECK-LABEL: @pow_scalar_base_tensor_exp
func.func @pow_scalar_base_tensor_exp(%a: f32, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tensor.splat
  // CHECK: linalg.generic
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (f32, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Case 3: Tensor x Tensor (Same Shape), float
// CHECK-LABEL: @pow_tensor_tensor_same_shape
func.func @pow_tensor_tensor_same_shape(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// Case 4: Tensor x Tensor (Broadcastable / Different Shapes)
// CHECK-LABEL: @pow_tensor_tensor_broadcast
func.func @pow_tensor_tensor_broadcast(%a: tensor<3x4xf32>, %b: tensor<4xf32>) -> tensor<3x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// Case 4 (Extra): Complex multi-dimensional broadcasting (size-1 dim alignment)
// CHECK-LABEL: @pow_tensor_tensor_complex_broadcast
func.func @pow_tensor_tensor_complex_broadcast(%a: tensor<1x4xf32>, %b: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// ==========================================
// DTYPE PROMOTIONS — float-only (Combination 1)
// ==========================================

// Combination 1 (Edge Case): Float x Float with mismatched bit-widths (ExtFOp)
// CHECK-LABEL: @pow_scalar_mismatched_float
func.func @pow_scalar_mismatched_float(%a: f32, %b: f64) -> f64 {
  // CHECK: arith.extf
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (f32, f64) -> f64
  return %0 : f64
}

// -----

// Reverse width order: f64 base, f32 exponent (exponent gets extended instead of base)
// CHECK-LABEL: @pow_scalar_mismatched_float_reverse
func.func @pow_scalar_mismatched_float_reverse(%a: f64, %b: f32) -> f64 {
  // CHECK: arith.extf
  // CHECK: math.powf
  %0 = aadesh.pow %a, %b : (f64, f32) -> f64
  return %0 : f64
}

// -----

// ==========================================
// INTEGER OPERANDS — not yet implemented, must fail to legalize
// ==========================================

// Integer exponent with float base
func.func @pow_scalar_float_int(%a: f32, %b: i32) -> f32 {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (f32, i32) -> f32
  return %0 : f32
}

// -----

// Integer base with float exponent
func.func @pow_scalar_int_float(%a: i32, %b: f32) -> f32 {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (i32, f32) -> f32
  return %0 : f32
}

// -----

// Integer x Integer, same width
func.func @pow_scalar_int_int(%a: i32, %b: i32) -> i32 {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (i32, i32) -> i32
  return %0 : i32
}

// -----

// Integer x Integer, mismatched bitwidths
func.func @pow_scalar_mismatched_int(%a: i16, %b: i32) -> i32 {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (i16, i32) -> i32
  return %0 : i32
}

// -----

// Integer tensors, same shape
func.func @pow_tensor_tensor_int(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// -----

// Mixed: float base tensor, integer exponent tensor
func.func @pow_tensor_float_int(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
  // expected-error@+2 {{aadesh.pow lowering currently only supports float base and exponent; integer operands are not yet supported}}
  // expected-error@+1 {{failed to legalize operation 'aadesh.pow' that was explicitly marked illegal}}
  %0 = aadesh.pow %a, %b : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
