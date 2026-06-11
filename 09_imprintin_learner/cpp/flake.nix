{
  description = "C++26 linear algebra playground backed by OpenBLAS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        nativeBuildInputs = with pkgs; [
          cmake
          ninja
          pkg-config
        ];

        # OpenBLAS, GoogleTest and Google Benchmark are provided by the shell,
        # so CMake's find_package short-circuits FetchContent for reproducible,
        # network-free builds.
        buildInputs = with pkgs; [
          openblas
          gtest
          gbenchmark
        ];
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "imprinting-learner";
          version = "0.1.0";
          src = ./.;
          inherit nativeBuildInputs buildInputs;
          cmakeFlags = [ "-DCMAKE_BUILD_TYPE=Release" ];
          doCheck = true;
          checkPhase = "ctest --output-on-failure";
        };

        devShells.default = pkgs.mkShell {
          packages = nativeBuildInputs ++ buildInputs ++ (with pkgs; [
            gcc          # recent GCC for C++26 (<print>, std::span, ...)
            clang-tools  # clangd + clang-format for editor tooling
          ]);

          shellHook = ''
            echo "imprinting-learner dev shell ready"
            echo "  configure : cmake --preset default"
            echo "  build     : cmake --build build/default"
            echo "  test      : ctest --preset default"
            echo "  benchmark : ./build/default/benchmarks/il_benchmarks"
          '';
        };
      });
}
