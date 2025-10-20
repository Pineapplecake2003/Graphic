{
  description = "Graphic";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=release-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nixgl.overlays.default ];
        };

        nix_pkgs = import nixpkgs {
          system = "x86_64-linux";
          overlays = [ nixgl.overlay ];
        };
        
        devTools = with pkgs; [
        ];

        nativeBuildInputs = with pkgs; [
          git
        ];

        mesaPackages = with pkgs; [
          mesa
          libGL
          libGLU
          xorg.libXxf86vm
          xorg.libXi
          xorg.libXrandr
        ];

        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          tqdm
          numpy
          pillow
          matplotlib
          (ps.buildPythonPackage rec {
            pname = "vedo";
            version = "2025.5.4";
            format = "pyproject";
            src = ps.fetchPypi {
              inherit pname version;
              sha256 = "192kjaiw77idma08migqqbpxwfsrim0qkrlh526lzx4r5dw8bdgy";
            };
            nativeBuildInputs = [ ps.setuptools ps.wheel ps.build ];
            propagatedBuildInputs = [
              numpy
              matplotlib
              pillow
              ps.vtk
              ps.typing-extensions
              ps.pygments
            ];
          })
          ps.pip
        ]);

        x11Packages = with pkgs; [
          libGL
          libGLU
          xorg.xhost
          xorg.xauth
          xorg.libX11
          xorg.libXrandr
          xorg.libXi
          xorg.libXcursor
          xorg.libXinerama
          xorg.libXrender
          xorg.libXfixes
          xorg.libXdamage
          xorg.libXcomposite
          xorg.libXt
          xorg.libSM
          xorg.libICE
        ];
      in
      rec {
        devShells.default = pkgs.mkShell {
          name = "Graphic";
          shellHook = ''
            export LIBGL_ALWAYS_SOFTWARE=1
          '';
          packages = devTools 
            ++ nativeBuildInputs 
            ++ [ pythonEnv (nixgl.packages.${system}.nixGLIntel) ] 
            ++ x11Packages 
            ++ mesaPackages;
        };
      });
}
