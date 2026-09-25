#ifndef AADESH_AADESHPASSES_H
#define AADESH_AADESHPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"          // for mlir::ModuleOp
#include "mlir/Dialect/Tosa/IR/TosaOps.h" // for mlir::tosa::TosaDialect
#include "mlir/Dialect/Arith/IR/Arith.h"  // for mlir::arith::ArithDialect
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include <memory>

namespace mlir {
namespace aadesh {

#define GEN_PASS_DECL
#include "Aadesh/AadeshPasses.h.inc"

std::unique_ptr<Pass> createLowerAadeshToTosaPass();
std::unique_ptr<Pass> createLowerAadeshPowToLoopsPass();
std::unique_ptr<Pass> createLowerAadeshArgMaxToLoopsPass();

#define GEN_PASS_REGISTRATION
#include "Aadesh/AadeshPasses.h.inc"

} // namespace aadesh
} // namespace mlir

#endif // AADESH_AADESHPASSES_H
