#ifndef AADESH_PASSES_TD
#define AADESH_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// LowerAadeshToTosa Pass
//===----------------------------------------------------------------------===//

def LowerAadeshToTosa : Pass<"lower-aadesh-to-tosa", "mlir::ModuleOp"> {
  let summary = "Lower Aadesh dialect operations to TOSA, Linalg, SCF, Math, "
                "Arith and ControlFlow dialects";
  let description = [{
    Lowers `aadesh` ops into lower-level dialects:
    - `aadesh.relu`, `aadesh.add`, `aadesh.mul` lower to TOSA dialect ops
      (`tosa.clamp`, `tosa.add`, `tosa.mul`) for tensor types, and Arith
      dialect ops for scalar types, as before.
    - `aadesh.pow` lowers to a `linalg.generic`. Float element types call
      `math.powf`. Integer element types compute the result via an unrolled
      exponentiation-by-squaring loop (`scf.for` over `arith` ops), guarded
      by a `cf.assert` against negative exponents on signed integer types.
      

    Note: the pass name/CLI flag (`lower-aadesh-to-tosa`) is kept as-is
    for compatibility even though `aadesh.pow` no longer lowers through
    TOSA; worth renaming to `lower-aadesh` if more non-TOSA targets
    get added later.
  }];

  let constructor = "aadesh::createLowerAadeshToTosaPass()";

  // Dependent dialects that may be created dynamically during conversion
  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

#endif // AADESH_PASSES_TD
