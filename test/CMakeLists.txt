configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(AADESH_TEST_DEPENDS aadesh-opt FileCheck)

add_lit_testsuite(check-aadesh "Running the Aadesh regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AADESH_TEST_DEPENDS}
)
