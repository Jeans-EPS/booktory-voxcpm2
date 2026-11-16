# booktory build of voxcpm2-cli

Private mirror of [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) that builds
`voxcpm2-cli` for the booktory plugin. booktory's installer downloads the Release assets with the
user's GitHub login and pins them by SHA-256 in `tools/lock/voxcpm2-cli.json`.

## Upstream pin

- Upstream: `tc-mb/llama.cpp-omni` `master` at `350a835d5f8215145aa210a7df0f67e8877b5d6e`
  (2026-11-13), kept as the `upstream` branch. The `booktory` branch is that commit plus the
  patches below, one commit each. To move the pin: fast-forward `upstream`, rebase `booktory`
  onto it, bump the tag (`voxcpm2-vN`).

## Patches

1. **MSVC build fixes**: `#include <string>` in `src/llama-mmap.h`; an `iovec`/`ssize_t` shim for
   Windows in `src/llama-moe-slot.cpp`.
2. **voxcpm2-cli**:
   - UTF-8 everywhere: on Windows argv is rebuilt from the wide command line and files are opened
     through UTF-8 paths; `--text-file PATH` reads the text from a UTF-8 file (BOM dropped).
   - `--batch PATH`: one model load for many clips. Each line is
     `<output.wav> TAB <cfg> TAB <text>` (empty cfg = `--cfg`). The reference/prompt WAV is loaded
     once; every clip uses the same `--seed`. One JSON line per clip on stdout
     (`index`, `output`, `ok`, `seconds`, `elapsed`, `sampleRate`); exit 2 if any clip failed.
   - `--threads N`, default the physical core count (upstream used ggml's default of 4).
   - `--version`: prints `voxcpm2-cli <tag>` and the compute devices, exit 0. The CUDA build exits
     3 when no usable GPU is found, so an installer can fall back to the CPU build.
3. **Speed and memory** (booktory #28):
   - Contexts: BaseLM `n_ctx` and ResidualLM `max_length` were a fixed 32768 (about 1.5 GB of KV);
     they default to 4096 and are set with `--lm-ctx N` / `--residual-ctx N`. A prompt that does
     not fit fails the clip with a message instead of asserting.
   - Reuse across the clips of a batch (`--no-prefix-cache` turns it off): the AudioVAE encoding of
     an unchanged reference WAV, the LocEnc embedding of unchanged audio positions, and the
     BaseLM/ResidualLM KV caches of the longest prefill prefix shared with the previous clip. With
     `-r` the whole reference prefix is reused; with `--prompt-wav`/`--prompt-text` the prompt-text
     tokens are (the audio follows the target text there, so it is prefilled again).
   - Graph size: LocEnc over the prefill was one graph unrolled per position, so a prompt
     transcript overflowed it (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`). It now runs only
     on audio positions, 32 patches per graph.
   - Per-clip decode cap: a batch line may be `<output.wav> TAB <cfg> TAB <steps> TAB <text>`.
   - Each JSON result line adds `steps`, `maxSteps`, `capped` (the cap was hit before the stop
     predictor fired), `promptPositions`, `reusedPositions` and `stages`: seconds per stage
     (`refEncode`, `locEnc`, `lmPrefill`, `resPrefill`, `cfm`, `lmStep`, `resStep`, `vaeDecode`).
4. **CI**: upstream workflows are removed; `.github/workflows/booktory-release.yml` builds on a
   `voxcpm2-v*` tag. CUDA builds use `CMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;89-real` (SASS
   for Ampere and Ada, PTX for Turing that newer GPUs JIT-compile; GPUs below compute capability
   7.5 use the CPU build) and `GGML_CUDA_FA=OFF` (VoxCPM2 does not use flash attention). The Linux
   jobs cache compiler output with ccache. A manual dispatch with `linux_only` builds only the
   Linux assets as run artifacts and publishes nothing: from `voxcpm2-v2` on, the Windows assets
   are built locally (MSVC 2022 Build Tools; CUDA from NVIDIA's 12.4.1 redist archives, no
   installer) with the same flags, and the release is assembled with `gh release create`.

## Release assets

| Asset | Contents |
|---|---|
| `voxcpm2-cli-win-x64-cpu.zip` | `voxcpm2-cli.exe`, AVX2/FMA/F16C/BMI2, static MSVC runtime |
| `voxcpm2-cli-win-x64-cuda12.zip` | `voxcpm2-cli.exe` with CUDA 12.4, compute capability 7.5+ (needs the cudart zip and an NVIDIA driver >= 527.41) |
| `cudart-win-x64-cuda12.zip` | `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` |
| `voxcpm2-cli-linux-x64-cpu.tar.gz` | `voxcpm2-cli`, glibc 2.35+ (Ubuntu 22.04), static libstdc++ |
| `voxcpm2-cli-linux-x64-cuda12.tar.gz` | `voxcpm2-cli` with CUDA 12.4, compute capability 7.5+, RPATH `$ORIGIN` (needs the cudart tarball and driver >= 525.60.13) |
| `cudart-linux-x64-cuda12.tar.gz` | `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` |
| `SHA256SUMS` | SHA-256 of every asset |

All builds: `GGML_NATIVE=OFF`, AVX2 baseline, no AVX-512, `GGML_OPENMP=OFF` (no OpenMP runtime
to ship). CI starts the Linux builds (`--version`); the CUDA one runs against the toolkit's stub
driver and must load and report no usable GPU.
