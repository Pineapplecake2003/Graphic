# Graphic

## Enter develop environment
```bash
nix develop
```

### Execute python
```bash
make
```

### Testing for check the stability of floating number
```bash
for i in $(seq 1 100); do echo "=== Run #$i ==="; if ! make; then echo "make failed at iteration $i"; exit 1; fi; done; echo "All 100 runs succeeded!"
```

### Clean generated files
```bash
make clean
```

#### 3DModel source
[Rushia model](https://sketchfab.com/3d-models/uruha-rushia-hololive-vtuber-99edec1aeb67428cb40113646428bf38)

[teapot](https://sketchfab.com/3d-models/utah-teapot-92f31e2028244c4b8ef6cbc07738aee5)