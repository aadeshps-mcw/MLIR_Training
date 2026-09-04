// RUN: aadesh-opt %s | aadesh-opt | FileCheck %s

// ---------------------------------------------------------------------
// Required operand only (input2 absent): float input, float matrix.
// ---------------------------------------------------------------------
// CHECK-LABEL: func.func @test_apply_matrix_single_input
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>) -> tensor<2x3x4xf32>
func.func @test_apply_matrix_single_input(%a: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.apply_matrix %[[ARG0]]
  // CHECK-SAME: matrix = dense<{{\[}}[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]{{\]}}> : tensor<2x2xf32>
  // CHECK-SAME: : (tensor<4xf32>) -> tensor<2x3x4xf32>
  %0 = aadesh.apply_matrix %a { matrix = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32> } : (tensor<4xf32>) -> tensor<2x3x4xf32>
  // CHECK: return %[[RES]] : tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}


// ---------------------------------------------------------------------
// Both operands present (input1 + optional input2), integer inputs.
// ---------------------------------------------------------------------
// CHECK-LABEL: func.func @test_apply_matrix_both_inputs
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xi32>, %[[ARG1:.*]]: tensor<4xi32>) -> tensor<2x2x4xf32>
func.func @test_apply_matrix_both_inputs(%a: tensor<4xi32>, %b: tensor<4xi32>) -> tensor<2x2x4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.apply_matrix %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: : (tensor<4xi32>, tensor<4xi32>) -> tensor<2x2x4xf32>
  %0 = aadesh.apply_matrix %a, %b { matrix = dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32> } : (tensor<4xi32>, tensor<4xi32>) -> tensor<2x2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x2x4xf32>
  return %0 : tensor<2x2x4xf32>
}


// ---------------------------------------------------------------------
// Confirm the generic form also parses correctly.
// ---------------------------------------------------------------------
// CHECK-LABEL: func.func @test_apply_matrix_generic
func.func @test_apply_matrix_generic(%a: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: aadesh.apply_matrix %{{.*}}
  %0 = "aadesh.apply_matrix"(%a) { matrix = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32> } : (tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}


// ---------------------------------------------------------------------
// Mixed float/integer inputs (input1 float, input2 integer) — allowed,
// since each operand's type is checked independently (no
// SameOperandsAndResultType on this op).
// ---------------------------------------------------------------------
// CHECK-LABEL: func.func @test_apply_matrix_mixed_input_types
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xi64>) -> tensor<1x1x4xf32>
func.func @test_apply_matrix_mixed_input_types(%a: tensor<4xf32>, %b: tensor<4xi64>) -> tensor<1x1x4xf32> {
  // CHECK: %[[RES:.*]] = aadesh.apply_matrix %[[ARG0]], %[[ARG1]]
  %0 = aadesh.apply_matrix %a, %b { matrix = dense<[[5.0]]> : tensor<1x1xf32> } : (tensor<4xf32>, tensor<4xi64>) -> tensor<1x1x4xf32>
  return %0 : tensor<1x1x4xf32>
}


// ---------------------------------------------------------------------
// Chained usage in a larger function body.
// ---------------------------------------------------------------------
// CHECK-LABEL: func.func @test_apply_matrix_chain
func.func @test_apply_matrix_chain(%a: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: %[[R1:.*]] = aadesh.apply_matrix %{{.*}}
  %0 = aadesh.apply_matrix %a { matrix = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32> } : (tensor<4xf32>) -> tensor<2x3x4xf32>
  // CHECK: %[[R2:.*]] = aadesh.add %[[R1]], %[[R1]]
  %1 = aadesh.add %0, %0 : tensor<2x3x4xf32>
  return %1 : tensor<2x3x4xf32>
}
