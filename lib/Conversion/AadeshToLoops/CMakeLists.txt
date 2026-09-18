add_mlir_conversion_library(AadeshToLoops
  LowerPowToLoops.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/Aadesh

  DEPENDS
  AadeshOpsIncGen
  AadeshPassIncGen

  LINK_LIBS PUBLIC
  MLIRAadesh
  MLIRArithDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRControlFlowDialect
)