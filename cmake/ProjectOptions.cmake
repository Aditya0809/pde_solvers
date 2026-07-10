add_library(pde_project_options INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(
    pde_project_options
    INTERFACE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wshadow
      $<$<BOOL:${PDE_WARNINGS_AS_ERRORS}>:-Werror>
  )
elseif(MSVC)
  target_compile_options(
    pde_project_options
    INTERFACE /W4 $<$<BOOL:${PDE_WARNINGS_AS_ERRORS}>:/WX>
  )
endif()

if(PDE_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(pde_project_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(pde_project_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  else()
    message(WARNING "The configured compiler does not support this project's sanitizer flags")
  endif()
endif()
