# Hrlrvo

Source code, configuration, tests, and documentation for the current Hrlrvo
workspace, published under the repository's `Hrlrvo/` directory.

Included project trees:

- `pipeline/`
- `pipeline_8K_to_16K/`
- `VideoOutpainting/`
- `Upscaler/`
- `monitor_Web/`
- `experiments/` (source, configuration, and documentation)
- `tools/`

Clone and enter the project directory:

```bash
git clone git@github.com:ccchoi7171/Hrlrvo.git
cd Hrlrvo/Hrlrvo
```

Videos, raster images, model weights/checkpoints, tensor/array dumps, datasets,
runtime outputs/logs, caches, environments, and compiled binaries are excluded.
Small model configuration and tokenizer files are included where available.
Obtain the required model weights and input data separately before running the
pipelines; consult the documentation in each project tree for setup details.

Third-party implementations are included as ordinary source files with their
existing license files. Nested Git histories and submodule declarations are
omitted so locally modified implementation files are present in the snapshot.
See [VENDORED_SOURCES.md](VENDORED_SOURCES.md) for upstream repository revisions.
