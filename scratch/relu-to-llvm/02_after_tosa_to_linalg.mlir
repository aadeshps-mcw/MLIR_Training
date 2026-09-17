#map = affine_map<(d0) -> (d0)>
module {
  func.func @relu_tensor_f32(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tensor.empty() : tensor<4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 0x7F800000 : f32
      %2 = arith.minimumf %in, %cst_0 : f32
      %3 = arith.maximumf %2, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

