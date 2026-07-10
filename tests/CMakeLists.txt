FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
  SYSTEM
)
FetchContent_MakeAvailable(googletest)

add_executable(pde_core_tests eigen_smoke_test.cpp)
target_link_libraries(pde_core_tests PRIVATE pde::core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(pde_core_tests)
