// Sample Aadesh IR to exercise the lower-aadesh-to-tosa pass.
// Run with:
//   aadesh-opt demo.mlir -lower-aadesh-to-tosa

module {
  // Tensor case: relu should lower to tosa.clamp
  func.func @relu_tensor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = aadesh.relu %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // Scalar case: relu should lower to arith.constant + arith.maximumf
  func.func @relu_scalar(%arg0: f32) -> f32 {
    %0 = aadesh.relu %arg0 : f32
    return %0 : f32
  }

  // Integer tensor case: relu should lower to tosa.clamp with correct i32 bounds
  func.func @relu_tensor_i32(%arg0: tensor<8xi32>) -> tensor<8xi32> {
    %0 = aadesh.relu %arg0 : tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}
