#include "Aadesh-c/Passes.h"
#include "Aadesh/AadeshPasses.h"

void mlirRegisterAadeshPasses(void) {
  mlir::aadesh::registerAadeshPasses();
}
