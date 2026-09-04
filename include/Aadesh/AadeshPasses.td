#ifndef AADESH_PASSES_TD
#define AADESH_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerAadeshToTosa Pass
//===----------------------------------------------------------------------===//

def LowerAadeshToTosa : Pass<"lower-aadesh-to-tosa", "mlir::ModuleOp"> {
  let summary = "Lower Aadesh dialect operations to TOSA and Arith dialects";
  let description = [{
    Lowers `aadesh` ops (`aadesh.relu`, `aadesh.add`, `aadesh.mul`) into TOSA 
    dialect operations for tensor types (e.g. `tosa.clamp`, `tosa.add`, `tosa.mul`) 
    and Arith dialect operations for scalar types.
  }];
  
  let constructor = "aadesh::createLowerAadeshToTosaPass()";
  
  // Dependent dialects that may be created dynamically during conversion
  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // AADESH_PASSES_TD
