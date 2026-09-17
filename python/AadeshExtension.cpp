#include "Aadesh-c/Dialects.h"
#include "Aadesh-c/Passes.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_aadeshDialects, m) {
  auto aadeshM = m.def_submodule("aadesh");

  aadeshM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__aadesh__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);

  aadeshM.def("register_passes", []() {
    mlirRegisterAadeshPasses();
  });
}
