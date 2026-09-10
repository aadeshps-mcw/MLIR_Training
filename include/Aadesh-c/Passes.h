#ifndef AADESH_C_PASSES_H
#define AADESH_C_PASSES_H

#include "mlir-c/Support.h"   // for MLIR_CAPI_EXPORTED

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterAadeshPasses(void);

#ifdef __cplusplus
}
#endif
#endif // AADESH_C_PASSES_H
