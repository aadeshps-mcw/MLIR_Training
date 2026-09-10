#map = affine_map<(d0) -> (d0)>
module {
  func.func @relu_tensor_f32(%arg0: memref<4xf32, strided<[?], offset: ?>>) -> memref<4xf32> {
    %alloc = memref.alloc() alignment = 64 : memref<4xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<4xf32, strided<[?], offset: ?>>) outs(%alloc : memref<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 0x7F800000 : f32
      %0 = arith.minimumf %in, %cst_0 : f32
      %1 = arith.maximumf %0, %cst : f32
      linalg.yield %1 : f32
    }
    %cast = memref.cast %alloc : memref<4xf32> to memref<4xf32, strided<[?], offset: ?>>
    return %alloc : memref<4xf32>
  }
}

