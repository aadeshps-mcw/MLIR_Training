add_mlir_public_c_api_library(AadeshCAPI
  Dialects.cpp
  Passes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAadesh
  AadeshToTosa
  AadeshToLoops
)
