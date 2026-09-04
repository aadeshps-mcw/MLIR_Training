add_llvm_executable(aadesh-opt
  aadesh-opt.cpp
)

target_link_libraries(aadesh-opt
  PRIVATE
  MLIRAadesh
  MLIRIR
  AadeshToTosa
  MLIRMlirOptMain
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRBytecodeOpInterface   
)
