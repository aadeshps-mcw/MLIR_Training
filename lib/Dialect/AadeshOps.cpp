#include "Aadesh/AadeshOps.h"
#include "Aadesh/AadeshDialect.h"

#define GET_OP_CLASSES
#include "Aadesh/AadeshOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ApplyMatrixOp::verify()
//===----------------------------------------------------------------------===//
//
// ODS (hasVerifier = 1) only declares this function in the generated
// header; it does not implement it. `matrix` is a DenseFPElementsAttr
// (F32ElementsAttr), which is backed by a RankedTensorType — so unlike
// the old nested ArrayAttr representation, ragged rows are impossible
// by construction. The only thing ODS can't express is rank, so that's
// checked here.

::llvm::LogicalResult mlir::aadesh::ApplyMatrixOp::verify() {
  mlir::DenseElementsAttr matrix = getMatrix();
  mlir::ShapedType matrixType = matrix.getType();

  if (matrixType.getRank() != 2)
    return emitOpError("matrix attribute must be a rank-2 (2D) tensor, got rank ")
           << matrixType.getRank();

  return ::mlir::success();
}
