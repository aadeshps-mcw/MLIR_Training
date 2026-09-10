module {
  func.func @relu_tensor_f32(%arg0: memref<4xf32, strided<[?], offset: ?>>) -> memref<4xf32> {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() alignment = 64 : memref<4xf32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<4xf32, strided<[?], offset: ?>>
    %3 = arith.maximumf %2, %cst : f32
    memref.store %3, %alloc[%0] : memref<4xf32>
    %4 = arith.addi %0, %c1 : index
    cf.br ^bb1(%4 : index)
  ^bb3:  // pred: ^bb1
    return %alloc : memref<4xf32>
  }
}

