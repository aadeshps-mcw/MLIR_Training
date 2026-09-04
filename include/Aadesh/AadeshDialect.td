#ifndef AADESH_DIALECT_TD
#define AADESH_DIALECT_TD

include "mlir/IR/OpBase.td"

def Aadesh_Dialect : Dialect {
  let name = "aadesh";
  let cppNamespace = "::mlir::aadesh";
  let summary = "The Aadesh Dialect";
  let description = [{
    A custom playground dialect for testing MLIR operations.
  }];
}

#endif // AADESH_DIALECT_TD
