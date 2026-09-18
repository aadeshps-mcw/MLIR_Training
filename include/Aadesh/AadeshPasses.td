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

def LowerAadeshPowToLoops : Pass<"lower-aadesh-pow-to-loops", "mlir::ModuleOp"> {
  let summary = "Lower aadesh.pow using exponentiation-by-squaring loops";
  let description = [{
    Lowers `aadesh.pow` into an `scf.for`-based exponentiation-by-squaring
    loop over `arith` ops, for integer or float base with an integer-typed
    exponent (assumed non-negative). Tensor operands are handled via a
    `linalg.generic` whose body performs the same loop.
  }];
  let constructor = "aadesh::createLowerAadeshPowToLoopsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

#endif // AADESH_PASSES_TD
