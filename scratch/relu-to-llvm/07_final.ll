; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @relu_tensor_f32(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = call ptr @malloc(i64 80)
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 63
  %14 = urem i64 %13, 64
  %15 = sub i64 %13, %14
  %16 = inttoptr i64 %15 to ptr
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %11, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %16, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 0, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 4, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 1, 4, 0
  br label %22

22:                                               ; preds = %25, %5
  %23 = phi i64 [ %36, %25 ], [ 0, %5 ]
  %24 = icmp slt i64 %23, 4
  br i1 %24, label %25, label %37

25:                                               ; preds = %22
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %28 = getelementptr float, ptr %26, i64 %27
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %30 = mul nsw i64 %23, %29
  %31 = getelementptr inbounds float, ptr %28, i64 %30
  %32 = load float, ptr %31, align 4
  %33 = call float @llvm.maximum.f32(float %32, float 0.000000e+00)
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 1
  %35 = getelementptr inbounds nuw float, ptr %34, i64 %23
  store float %33, ptr %35, align 4
  %36 = add i64 %23, 1
  br label %22

37:                                               ; preds = %22
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %21
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
