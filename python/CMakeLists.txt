include(AddMLIRPython)
 
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_aadesh.")
 
declare_mlir_python_sources(AadeshPythonSources)
 
declare_mlir_python_sources(AadeshPythonSources.Dialects
  ADD_TO_PARENT AadeshPythonSources
)
 
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AadeshPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_aadesh"
  TD_FILE dialects/AadeshOps.td
  SOURCES
    dialects/aadesh.py
  DIALECT_NAME aadesh
)
 
declare_mlir_python_extension(AadeshPythonSources.Extension
  MODULE_NAME _aadeshDialects
  ADD_TO_PARENT AadeshPythonSources
  SOURCES
    AadeshExtension.cpp
  EMBED_CAPI_LINK_LIBS
    AadeshCAPI
)
 
add_mlir_python_common_capi_library(AadeshPythonCAPI
  INSTALL_COMPONENT AadeshPythonModules
  INSTALL_DESTINATION python_packages/aadesh/mlir_aadesh/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/aadesh/mlir_aadesh/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    AadeshPythonSources
    MLIRPythonSources.Core
    MLIRPythonExtension.RegisterEverything
)
 
target_link_libraries(AadeshPythonCAPI PUBLIC
  MLIRCAPITransforms
  MLIRCAPIConversion
  MLIRCAPIRegisterEverything
  -Wl,--whole-archive
  MLIRCAPIExecutionEngine
  -Wl,--no-whole-archive
)
 
add_mlir_python_modules(AadeshPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/aadesh/mlir_aadesh"
  INSTALL_PREFIX "python_packages/aadesh/mlir_aadesh"
  DECLARED_SOURCES
    AadeshPythonSources
    MLIRPythonSources
    MLIRPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    AadeshPythonCAPI
)
