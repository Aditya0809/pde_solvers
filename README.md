# PDE Solvers

A C++20 learning project for verified finite-difference/finite-element PDE solvers and
time-integration methods.

## Prerequisites (macOS)

Install the build and developer tools with Homebrew:

```sh
brew install cmake ninja llvm
```

Eigen and GoogleTest are pinned and downloaded by CMake, so they do not need to be installed
globally. Add Homebrew LLVM to your shell path to use `clang-format` and `clang-tidy`:

```sh
echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

MPI is optional. To build the existing MPI prototype, also run `brew install open-mpi` and
configure with `-DPDE_ENABLE_MPI=ON`.

## Configure, build, and test

Debug builds enable AddressSanitizer and UndefinedBehaviorSanitizer:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

For an optimized build:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Format and statically analyze source files with:

```sh
clang-format -i include/**/*.hpp tests/*.cpp common/**/*.{cpp,hpp} fd/**/*.cpp
clang-tidy -p build/debug tests/eigen_smoke_test.cpp
```

## Dependency policy

- Eigen and GoogleTest are project-local, version-pinned CMake dependencies.
- CMake, Ninja, clang-format, and clang-tidy are developer tools installed on the machine.
- The first CMake configure requires internet access to download Eigen and GoogleTest.
- Start by implementing time integrators directly against Eigen vectors. Add Boost.Odeint or
  SUNDIALS only if a later requirement calls for adaptive/reference integrators or stiff systems.
