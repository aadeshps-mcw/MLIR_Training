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

def LowerAadeshArgMaxToLoops : Pass<"lower-aadesh-argmax-to-loops", "mlir::ModuleOp"> {
  let summary = "Lower aadesh.argmax to scf loops";
  let description = [{
    Lowers `aadesh.argmax` into `scf.for`-based reduction loops over
    `tensor.extract`/`tensor.insert`. When `dim` is absent, the input is
    flattened and reduced over all elements. When `dim` is present, one
    `scf.for` per non-reduced dimension is nested around the reduction
    loop, with `keep_dim` controlling whether the reduced axis survives
    (size 1) in the result.
  }];
  let constructor = "aadesh::createLowerAadeshArgMaxToLoopsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // AADESH_PASSES_TD
