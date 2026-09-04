// RUN: not aadesh-opt --split-input-file %s 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL

func.func @test_add_type_mismatch(%a: f32, %b: i32) -> f32 {
  // CHECK-FAIL: 'aadesh.add' op requires the same type for all operands and results
  %0 = "aadesh.add"(%a, %b) : (f32, i32) -> f32
  return %0 : f32
}

// -----

func.func @test_mul_type_mismatch(%a: f32, %b: i32) -> f32 {
  // CHECK-FAIL: 'aadesh.mul' op requires the same type for all operands and results
  %0 = "aadesh.mul"(%a, %b) : (f32, i32) -> f32
  return %0 : f32
}

// -----

func.func @test_relu_type_mismatch(%a: f32) -> i32 {
  // CHECK-FAIL: 'aadesh.relu' op requires the same type for all operands and results
  %0 = "aadesh.relu"(%a) : (f32) -> i32
  return %0 : i32
}
