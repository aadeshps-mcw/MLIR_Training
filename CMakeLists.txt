cmake_minimum_required(VERSION 3.20)
project(aadesh-dialect CXX C)
set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_BINARY_DIR}/lib)
add_definitions(${LLVM_DEFINITIONS})

list(APPEND MLIR_TABLEGEN_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/include")

find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit.py lit)
if(NOT LLVM_EXTERNAL_LIT)
  message(FATAL_ERROR "llvm-lit not found — pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit")
endif()

add_subdirectory(include/Aadesh)   # generates AadeshOpsIncGen + AadeshPassIncGen
add_subdirectory(lib/Dialect)      # builds MLIRAadesh
add_subdirectory(lib/Conversion)   # builds AadeshToTosa
add_subdirectory(tools)
add_subdirectory(test)
