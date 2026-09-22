#include "Aadesh/AadeshDialect.h"
#include "Aadesh/AadeshOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "Aadesh/AadeshDialect.cpp.inc"

namespace mlir::aadesh {
void AadeshDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Aadesh/AadeshOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AadeshDialect::materializeConstant
//===----------------------------------------------------------------------===//
//
// Called by the canonicalizer (and anything else driving folding) whenever a
// `fold()` hook returns an Attribute instead of an existing Value. This hook
// is responsible for turning that Attribute back into a real, verifiable
// operation in the IR. Without it, folds that produce new constant values
// (as opposed to just forwarding an existing operand) are silently dropped.
mlir::Operation *AadeshDialect::materializeConstant(mlir::OpBuilder &builder,
                                                     mlir::Attribute value,
                                                     mlir::Type type,
                                                     mlir::Location loc) {
  // arith.constant covers scalar integer/float constants, and also dense
  // element constants (ElementsAttr, the common base of DenseElementsAttr)
  // for tensor-typed folds — which is everything ReluOp::fold (and friends)
  // currently produce.
  if (llvm::isa<mlir::IntegerAttr, mlir::FloatAttr, mlir::ElementsAttr>(value))
    return mlir::arith::ConstantOp::create(
      builder, loc, type, llvm::cast<mlir::TypedAttr>(value));

  return nullptr;
}

} // namespace mlir::aadesh
