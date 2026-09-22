// RUN: aadesh-opt %s -canonicalize | FileCheck %s

// ---------------------------------------------------------------------------
// Case A: signless integer constant, value has high bit set but is created
// as a *positive* signless constant (e.g. 200 : i8, stored as bit pattern
// 0xC8). Since arith.constant only supports signless types, there is no
// "unsigned type" information attached here -- APInt::isNegative() reads
// the bit pattern directly, so 200 : i8 IS read as negative (two's
// complement -56). This is expected/correct for a signless type: without
// sign information, relu has no choice but to treat the sign bit as the
// sign. This should fold to 0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_signless_const_high_bit
func.func @relu_signless_const_high_bit() -> i8 {
  %c200 = arith.constant -56 : i8   // bit pattern 0xC8, same as 200 : ui8 would be
  %0 = aadesh.relu %c200 : i8
  // CHECK: %[[C:.*]] = arith.constant 0 : i8
  // CHECK: return %[[C]]
  return %0 : i8
}

// ---------------------------------------------------------------------------
// Case B: signless integer constant, genuinely non-negative -> unchanged.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_signless_const_positive
func.func @relu_signless_const_positive() -> i8 {
  %c5 = arith.constant 5 : i8
  %0 = aadesh.relu %c5 : i8
  // CHECK: %[[C:.*]] = arith.constant 5 : i8
  // CHECK: return %[[C]]
  return %0 : i8
}

// ---------------------------------------------------------------------------
// Case C: signless integer constant, genuinely negative -> should fold to 0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_signless_const_negative
func.func @relu_signless_const_negative() -> i8 {
  %cn5 = arith.constant -5 : i8
  %0 = aadesh.relu %cn5 : i8
  // CHECK: %[[C:.*]] = arith.constant 0 : i8
  // CHECK: return %[[C]]
  return %0 : i8
}

// ---------------------------------------------------------------------------
// Case D: float constant, negative -> should fold to 0.0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_float_const_negative
func.func @relu_float_const_negative() -> f32 {
  %cn = arith.constant -3.5 : f32
  %0 = aadesh.relu %cn : f32
  // CHECK: %[[C:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: return %[[C]]
  return %0 : f32
}

// ---------------------------------------------------------------------------
// Case E: float constant, non-negative -> unchanged.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_float_const_positive
func.func @relu_float_const_positive() -> f32 {
  %c = arith.constant 3.5 : f32
  %0 = aadesh.relu %c : f32
  // CHECK: %[[C:.*]] = arith.constant 3.500000e+00 : f32
  // CHECK: return %[[C]]
  return %0 : f32
}

// ---------------------------------------------------------------------------
// Case F: non-constant, unsigned integer type -> relu is a no-op,
// operand should be forwarded directly (op erased). This is the real,
// reachable test of your Case 2 branch (intType.isUnsigned()), since
// ui8 is a valid *type* for a block argument even though arith.constant
// can't materialize a ui8-typed constant.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_unsigned_dynamic
func.func @relu_unsigned_dynamic(%arg0: ui8) -> ui8 {
  %0 = aadesh.relu %arg0 : ui8
  // CHECK-NOT: aadesh.relu
  // CHECK: return %arg0
  return %0 : ui8
}

// ---------------------------------------------------------------------------
// Case G: non-constant, signless integer type -> NOT foldable
// (relu must remain, since we don't know the sign at compile time).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_signless_dynamic
func.func @relu_signless_dynamic(%arg0: i32) -> i32 {
  %0 = aadesh.relu %arg0 : i32
  // CHECK: aadesh.relu
  return %0 : i32
}

// ---------------------------------------------------------------------------
// Case H: float constant, negative NaN -> NaN sign bit must NOT trigger the
// zero-fold (relu must match arith.maximumf's NaN-propagating semantics).
// The op still folds away, but to the NaN constant itself, not to 0.0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_float_const_negative_nan
func.func @relu_float_const_negative_nan() -> f32 {
  %cnan = arith.constant 0xFFC00000 : f32   // negative-signed NaN bit pattern
  %0 = aadesh.relu %cnan : f32
  // CHECK-NOT: aadesh.relu
  // CHECK: %[[C:.*]] = arith.constant 0xFFC00000 : f32
  // CHECK: return %[[C]]
  return %0 : f32
}

// ---------------------------------------------------------------------------
// Case I: splat float tensor constant, negative -> folds to a splat of 0.0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_splat_float_tensor_negative
func.func @relu_splat_float_tensor_negative() -> tensor<4xf32> {
  %c = arith.constant dense<-2.0> : tensor<4xf32>
  %0 = aadesh.relu %c : tensor<4xf32>
  // CHECK: %[[C:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
  // CHECK: return %[[C]]
  return %0 : tensor<4xf32>
}

// ---------------------------------------------------------------------------
// Case J: splat float tensor constant, non-negative -> unchanged.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_splat_float_tensor_positive
func.func @relu_splat_float_tensor_positive() -> tensor<4xf32> {
  %c = arith.constant dense<2.0> : tensor<4xf32>
  %0 = aadesh.relu %c : tensor<4xf32>
  // CHECK: %[[C:.*]] = arith.constant dense<2.000000e+00> : tensor<4xf32>
  // CHECK: return %[[C]]
  return %0 : tensor<4xf32>
}

// ---------------------------------------------------------------------------
// Case K: non-splat float tensor constant, mixed signs -> folds elementwise.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_dense_float_tensor_mixed
func.func @relu_dense_float_tensor_mixed() -> tensor<4xf32> {
  %c = arith.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %0 = aadesh.relu %c : tensor<4xf32>
  // CHECK: %[[C:.*]] = arith.constant dense<[0.000000e+00, 2.000000e+00, 0.000000e+00, 4.000000e+00]> : tensor<4xf32>
  // CHECK: return %[[C]]
  return %0 : tensor<4xf32>
}

// ---------------------------------------------------------------------------
// Case L: splat signless integer tensor constant, negative -> folds to a
// splat of 0.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_splat_int_tensor_negative
func.func @relu_splat_int_tensor_negative() -> tensor<4xi8> {
  %c = arith.constant dense<-5> : tensor<4xi8>
  %0 = aadesh.relu %c : tensor<4xi8>
  // CHECK: %[[C:.*]] = arith.constant dense<0> : tensor<4xi8>
  // CHECK: return %[[C]]
  return %0 : tensor<4xi8>
}

// ---------------------------------------------------------------------------
// Case M: non-splat signless integer tensor constant, mixed signs -> folds
// elementwise.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_dense_int_tensor_mixed
func.func @relu_dense_int_tensor_mixed() -> tensor<4xi8> {
  %c = arith.constant dense<[-1, 2, -3, 4]> : tensor<4xi8>
  %0 = aadesh.relu %c : tensor<4xi8>
  // CHECK: %[[C:.*]] = arith.constant dense<[0, 2, 0, 4]> : tensor<4xi8>
  // CHECK: return %[[C]]
  return %0 : tensor<4xi8>
}

// ---------------------------------------------------------------------------
// Case N: non-constant, unsigned integer TENSOR type -> relu is a no-op,
// operand forwarded directly (op erased). Tensor analogue of Case F.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_unsigned_tensor_dynamic
func.func @relu_unsigned_tensor_dynamic(%arg0: tensor<4xui8>) -> tensor<4xui8> {
  %0 = aadesh.relu %arg0 : tensor<4xui8>
  // CHECK-NOT: aadesh.relu
  // CHECK: return %arg0
  return %0 : tensor<4xui8>
}

// ---------------------------------------------------------------------------
// Case O: non-constant, signless integer TENSOR type -> NOT foldable
// (sign unknown at compile time). Tensor analogue of Case G.
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_signless_tensor_dynamic
func.func @relu_signless_tensor_dynamic(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = aadesh.relu %arg0 : tensor<4xi32>
  // CHECK: aadesh.relu
  return %0 : tensor<4xi32>
}

// ---------------------------------------------------------------------------
// Case P: non-constant, F32 TENSOR type -> NOT foldable (Case 2 in the
// folder only ever forwards unsigned-integer inputs; float has no such
// no-op shortcut).
// ---------------------------------------------------------------------------

// CHECK-LABEL: func @relu_float_tensor_dynamic
func.func @relu_float_tensor_dynamic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = aadesh.relu %arg0 : tensor<4xf32>
  // CHECK: aadesh.relu
  return %0 : tensor<4xf32>
}
