// RUN: aadesh-opt --split-input-file %s | FileCheck %s

// Basic round-trip: parse, print, and check the custom assembly form survives.
// CHECK-LABEL: func.func @test_add
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
func.func @test_add(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : f32
  %0 = aadesh.add %a, %b : f32
  // CHECK: return %[[RES]] : f32
  return %0 : f32
}


// Confirm the generic form also parses correctly (i.e. the op registered
// properly and isn't only reachable via the custom assemblyFormat).
// CHECK-LABEL: func.func @test_add_generic
func.func @test_add_generic(%a: f32, %b: f32) -> f32 {
  // CHECK: aadesh.add %{{.*}}, %{{.*}} : f32
  %0 = "aadesh.add"(%a, %b) : (f32, f32) -> f32
  return %0 : f32
}


// Verify the op is usable inside a larger, more realistic function body,
// chained with a standard dialect op.
// CHECK-LABEL: func.func @test_add_chain
func.func @test_add_chain(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK: %[[SUM1:.*]] = aadesh.add
  %0 = aadesh.add %a, %b : f32
  // CHECK: %[[SUM2:.*]] = aadesh.add %[[SUM1]]
  %1 = aadesh.add %0, %c : f32
  return %1 : f32
}


// -----
// Additional scalar float width: f64
// -----
// CHECK-LABEL: func.func @test_add_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64) -> f64
func.func @test_add_f64(%a: f64, %b: f64) -> f64 {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : f64
  %0 = aadesh.add %a, %b : f64
  return %0 : f64
}


// -----
// Signless integer: i32
// -----
// CHECK-LABEL: func.func @test_add_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @test_add_i32(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : i32
  %0 = aadesh.add %a, %b : i32
  return %0 : i32
}


// -----
// Signed integer: si32
// -----
// CHECK-LABEL: func.func @test_add_si32
// CHECK-SAME: (%[[ARG0:.*]]: si32, %[[ARG1:.*]]: si32) -> si32
func.func @test_add_si32(%a: si32, %b: si32) -> si32 {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : si32
  %0 = aadesh.add %a, %b : si32
  return %0 : si32
}


// -----
// Unsigned integer: ui32
// -----
// CHECK-LABEL: func.func @test_add_ui32
// CHECK-SAME: (%[[ARG0:.*]]: ui32, %[[ARG1:.*]]: ui32) -> ui32
func.func @test_add_ui32(%a: ui32, %b: ui32) -> ui32 {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : ui32
  %0 = aadesh.add %a, %b : ui32
  return %0 : ui32
}


// -----
// Tensor of float: tensor<4xf32>
// -----
// CHECK-LABEL: func.func @test_add_tensor_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_add_tensor_f32(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : tensor<4xf32>
  %0 = aadesh.add %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}


// -----
// Tensor of integer: tensor<8xi32>
// -----
// CHECK-LABEL: func.func @test_add_tensor_i32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>, %[[ARG1:.*]]: tensor<8xi32>) -> tensor<8xi32>
func.func @test_add_tensor_i32(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : tensor<8xi32>
  %0 = aadesh.add %a, %b : tensor<8xi32>
  return %0 : tensor<8xi32>
}


// -----
// Dynamic-shape tensor: tensor<?xf32>
// -----
// CHECK-LABEL: func.func @test_add_tensor_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32>
func.func @test_add_tensor_dynamic(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[RES:.*]] = aadesh.add %[[ARG0]], %[[ARG1]] : tensor<?xf32>
  %0 = aadesh.add %a, %b : tensor<?xf32>
  return %0 : tensor<?xf32>
}


// =======================================================================
// MulOp tests — mirror AddOp coverage
// =======================================================================

// Basic round-trip: parse, print, and check the custom assembly form survives.
// CHECK-LABEL: func.func @test_mul
// CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> f32
func.func @test_mul(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : f32
  %0 = aadesh.mul %a, %b : f32
  // CHECK: return %[[RES]] : f32
  return %0 : f32
}


// Confirm the generic form also parses correctly (i.e. the op registered
// properly and isn't only reachable via the custom assemblyFormat).
// CHECK-LABEL: func.func @test_mul_generic
func.func @test_mul_generic(%a: f32, %b: f32) -> f32 {
  // CHECK: aadesh.mul %{{.*}}, %{{.*}} : f32
  %0 = "aadesh.mul"(%a, %b) : (f32, f32) -> f32
  return %0 : f32
}


// Verify the op is usable inside a larger, more realistic function body,
// chained with a standard dialect op.
// CHECK-LABEL: func.func @test_mul_chain
func.func @test_mul_chain(%a: f32, %b: f32, %c: f32) -> f32 {
  // CHECK: %[[PROD1:.*]] = aadesh.mul
  %0 = aadesh.mul %a, %b : f32
  // CHECK: %[[PROD2:.*]] = aadesh.mul %[[PROD1]]
  %1 = aadesh.mul %0, %c : f32
  return %1 : f32
}


// -----
// Additional scalar float width: f64
// -----
// CHECK-LABEL: func.func @test_mul_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64) -> f64
func.func @test_mul_f64(%a: f64, %b: f64) -> f64 {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : f64
  %0 = aadesh.mul %a, %b : f64
  return %0 : f64
}


// -----
// Signless integer: i32
// -----
// CHECK-LABEL: func.func @test_mul_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32) -> i32
func.func @test_mul_i32(%a: i32, %b: i32) -> i32 {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : i32
  %0 = aadesh.mul %a, %b : i32
  return %0 : i32
}


// -----
// Signed integer: si32
// -----
// CHECK-LABEL: func.func @test_mul_si32
// CHECK-SAME: (%[[ARG0:.*]]: si32, %[[ARG1:.*]]: si32) -> si32
func.func @test_mul_si32(%a: si32, %b: si32) -> si32 {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : si32
  %0 = aadesh.mul %a, %b : si32
  return %0 : si32
}


// -----
// Unsigned integer: ui32
// -----
// CHECK-LABEL: func.func @test_mul_ui32
// CHECK-SAME: (%[[ARG0:.*]]: ui32, %[[ARG1:.*]]: ui32) -> ui32
func.func @test_mul_ui32(%a: ui32, %b: ui32) -> ui32 {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : ui32
  %0 = aadesh.mul %a, %b : ui32
  return %0 : ui32
}


// -----
// Tensor of float: tensor<4xf32>
// -----
// CHECK-LABEL: func.func @test_mul_tensor_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_mul_tensor_f32(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : tensor<4xf32>
  %0 = aadesh.mul %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}


// -----
// Tensor of integer: tensor<8xi32>
// -----
// CHECK-LABEL: func.func @test_mul_tensor_i32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>, %[[ARG1:.*]]: tensor<8xi32>) -> tensor<8xi32>
func.func @test_mul_tensor_i32(%a: tensor<8xi32>, %b: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : tensor<8xi32>
  %0 = aadesh.mul %a, %b : tensor<8xi32>
  return %0 : tensor<8xi32>
}


// -----
// Dynamic-shape tensor: tensor<?xf32>
// -----
// CHECK-LABEL: func.func @test_mul_tensor_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>) -> tensor<?xf32>
func.func @test_mul_tensor_dynamic(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[RES:.*]] = aadesh.mul %[[ARG0]], %[[ARG1]] : tensor<?xf32>
  %0 = aadesh.mul %a, %b : tensor<?xf32>
  return %0 : tensor<?xf32>
}


// =======================================================================
// ReluOp tests — single-operand analogue of AddOp/MulOp coverage
// =======================================================================

// Basic round-trip: parse, print, and check the custom assembly form survives.
// CHECK-LABEL: func.func @test_relu
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
func.func @test_relu(%a: f32) -> f32 {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : f32
  %0 = aadesh.relu %a : f32
  // CHECK: return %[[RES]] : f32
  return %0 : f32
}


// Confirm the generic form also parses correctly (i.e. the op registered
// properly and isn't only reachable via the custom assemblyFormat).
// CHECK-LABEL: func.func @test_relu_generic
func.func @test_relu_generic(%a: f32) -> f32 {
  // CHECK: aadesh.relu %{{.*}} : f32
  %0 = "aadesh.relu"(%a) : (f32) -> f32
  return %0 : f32
}


// Verify the op is usable inside a larger, more realistic function body,
// chained with itself and with a standard dialect op.
// CHECK-LABEL: func.func @test_relu_chain
func.func @test_relu_chain(%a: f32, %b: f32) -> f32 {
  // CHECK: %[[SUM:.*]] = aadesh.add
  %0 = aadesh.add %a, %b : f32
  // CHECK: %[[RES:.*]] = aadesh.relu %[[SUM]]
  %1 = aadesh.relu %0 : f32
  return %1 : f32
}


// -----
// Additional scalar float width: f64
// -----
// CHECK-LABEL: func.func @test_relu_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64) -> f64
func.func @test_relu_f64(%a: f64) -> f64 {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : f64
  %0 = aadesh.relu %a : f64
  return %0 : f64
}


// -----
// Signless integer: i32
// -----
// CHECK-LABEL: func.func @test_relu_i32
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
func.func @test_relu_i32(%a: i32) -> i32 {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : i32
  %0 = aadesh.relu %a : i32
  return %0 : i32
}


// -----
// Signed integer: si32
// -----
// CHECK-LABEL: func.func @test_relu_si32
// CHECK-SAME: (%[[ARG0:.*]]: si32) -> si32
func.func @test_relu_si32(%a: si32) -> si32 {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : si32
  %0 = aadesh.relu %a : si32
  return %0 : si32
}


// -----
// Unsigned integer: ui32
// -----
// CHECK-LABEL: func.func @test_relu_ui32
// CHECK-SAME: (%[[ARG0:.*]]: ui32) -> ui32
func.func @test_relu_ui32(%a: ui32) -> ui32 {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : ui32
  %0 = aadesh.relu %a : ui32
  return %0 : ui32
}


// -----
// Tensor of float: tensor<4xf32>
// -----
// CHECK-LABEL: func.func @test_relu_tensor_f32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<4xf32>
func.func @test_relu_tensor_f32(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : tensor<4xf32>
  %0 = aadesh.relu %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}


// -----
// Tensor of integer: tensor<8xi32>
// -----
// CHECK-LABEL: func.func @test_relu_tensor_i32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>) -> tensor<8xi32>
func.func @test_relu_tensor_i32(%a: tensor<8xi32>) -> tensor<8xi32> {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : tensor<8xi32>
  %0 = aadesh.relu %a : tensor<8xi32>
  return %0 : tensor<8xi32>
}


// -----
// Dynamic-shape tensor: tensor<?xf32>
// -----
// CHECK-LABEL: func.func @test_relu_tensor_dynamic
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf32>
func.func @test_relu_tensor_dynamic(%a: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[RES:.*]] = aadesh.relu %[[ARG0]] : tensor<?xf32>
  %0 = aadesh.relu %a : tensor<?xf32>
  return %0 : tensor<?xf32>
}
