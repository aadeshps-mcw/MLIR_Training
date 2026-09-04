#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

#include "Aadesh/AadeshDialect.cpp.inc"

namespace mlir::aadesh {
void AadeshDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Aadesh/AadeshOps.cpp.inc"
      >();
}
} // namespace mlir::aadesh
