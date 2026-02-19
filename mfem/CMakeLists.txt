# mfem/CMakeLists.txt

set(MFEM_PREFIX "" CACHE PATH "MFEM install prefix (contains include/ and lib/)")
set(HYPRE_PREFIX "" CACHE PATH "HYPRE install prefix (contains include/ and lib/)")
set(METIS_PREFIX "" CACHE PATH "METIS install prefix (contains include/ and lib/)")

foreach(var MFEM_PREFIX HYPRE_PREFIX METIS_PREFIX)
  if(NOT ${var})
    message(FATAL_ERROR "${var} not set. Pass -D${var}=... when configuring.")
  endif()
endforeach()

# MFEM library
find_library(MFEM_LIB mfem PATHS "${MFEM_PREFIX}/lib" REQUIRED)

add_library(MFEM::mfem UNKNOWN IMPORTED)
set_target_properties(MFEM::mfem PROPERTIES
  IMPORTED_LOCATION "${MFEM_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${MFEM_PREFIX}/include"
)

# HYPRE + METIS libs (static in your case)
find_library(HYPRE_LIB HYPRE PATHS "${HYPRE_PREFIX}/lib" REQUIRED)
find_library(METIS_LIB metis PATHS "${METIS_PREFIX}/lib" REQUIRED)

target_include_directories(MFEM::mfem INTERFACE
  "${HYPRE_PREFIX}/include"
  "${METIS_PREFIX}/include"
)

# MFEM dependencies
target_link_libraries(MFEM::mfem INTERFACE
  "${HYPRE_LIB}"
  "${METIS_LIB}"
  MPI::MPI_CXX
  m
  rt
)

add_executable(mfem_smoke smoke.cpp)
target_link_libraries(mfem_smoke PRIVATE MFEM::mfem MPI::MPI_CXX project_warnings)