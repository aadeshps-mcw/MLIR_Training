add_mlir_conversion_library(AadeshToTosa
  LowerToTosa.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/Aadesh

  DEPENDS
  AadeshOpsIncGen
  AadeshPassIncGen

  LINK_LIBS PUBLIC
  MLIRAadesh
  MLIRArithDialect
  MLIRTosaDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
