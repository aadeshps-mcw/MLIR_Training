func.func @relu_tensor_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = aadesh.relu %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
