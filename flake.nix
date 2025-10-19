{
  description = "Graphic";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=release-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        devTools = with pkgs; [
          
        ];

        nativeBuildInputs = with pkgs; [
          git
        ];
        
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          tqdm
          numpy
          pillow
          matplotlib
          ps.pip
        ]);
      in
      rec {
        devShells.default = pkgs.mkShell {
          name = "Graphic";
          packages = devTools ++ nativeBuildInputs ++ [pythonEnv];
        };
      });
}