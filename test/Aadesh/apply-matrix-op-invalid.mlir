// RUN: not aadesh-opt --split-input-file %s 2>&1 | FileCheck %s

// Rank-1 matrix (not 2D): should be rejected by the verifier's rank check.
func.func @test_matrix_wrong_rank(%a: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: error: 'aadesh.apply_matrix' op matrix attribute must be a rank-2 (2D) tensor, got rank 1
  %0 = aadesh.apply_matrix %a { matrix = dense<[1.0, 2.0, 3.0]> : tensor<3xf32> } : (tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

// Result is not rank-3: should be rejected by the TensorRankOf<[...], [3]>
// constraint on the result.
func.func @test_result_wrong_rank(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: error: 'aadesh.apply_matrix' op result #0 must be 3D tensor of floating-point values, but got 'tensor<4xf32>'
  %0 = aadesh.apply_matrix %a { matrix = dense<[[1.0, 2.0]]> : tensor<1x2xf32> } : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Result element type is not a float: should be rejected since
// TensorRankOf<[AnyFloat], [3]> only allows float element types.
func.func @test_result_wrong_element_type(%a: tensor<4xf32>) -> tensor<1x1x4xi32> {
  // CHECK: error: 'aadesh.apply_matrix' op result #0 must be 3D tensor of floating-point values, but got 'tensor<1x1x4xi32>'
  %0 = aadesh.apply_matrix %a { matrix = dense<[[1.0]]> : tensor<1x1xf32> } : (tensor<4xf32>) -> tensor<1x1x4xi32>
  return %0 : tensor<1x1x4xi32>
}

// -----

// Missing required attribute `matrix`: should fail to parse/verify.
func.func @test_missing_matrix_attr(%a: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: error: 'aadesh.apply_matrix' op requires attribute 'matrix'
  %0 = aadesh.apply_matrix %a : (tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}
