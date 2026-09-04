# lib/Dialect/CMakeLists.txt
add_mlir_dialect_library(MLIRAadesh
  AadeshDialect.cpp
  AadeshOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/Aadesh

  DEPENDS
  AadeshOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
