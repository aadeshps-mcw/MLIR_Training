#include "Aadesh-c/Dialects.h"
#include "Aadesh/AadeshDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Aadesh, aadesh,
                                       mlir::aadesh::AadeshDialect)
