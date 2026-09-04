#ifndef AADESH_OPS_H
#define AADESH_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "Aadesh/AadeshDialect.h"

#define GET_OP_CLASSES
#include "Aadesh/AadeshOps.h.inc"

#endif // AADESH_OPS_H
