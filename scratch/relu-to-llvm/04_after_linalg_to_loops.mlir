module {
  func.func @relu_tensor_f32(%arg0: memref<4xf32, strided<[?], offset: ?>>) -> memref<4xf32> {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() alignment = 64 : memref<4xf32>
    scf.for %arg1 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg0[%arg1] : memref<4xf32, strided<[?], offset: ?>>
      %1 = arith.maximumf %0, %cst : f32
      memref.store %1, %alloc[%arg1] : memref<4xf32>
    }
    return %alloc : memref<4xf32>
  }
}

