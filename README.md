# Scriptum

[![CI](https://github.com/RKeelan/Scriptum/actions/workflows/ci.yml/badge.svg)](https://github.com/RKeelan/Scriptum/actions/workflows/ci.yml)

Browser-based handwriting synthesis using Alex Graves' recurrent neural network model. Given input text, Scriptum generates realistic handwriting as SVG paths.

## Setup

Model weights are stored with [Git LFS](https://git-lfs.com/). Install it before cloning:

```powershell
git lfs install    # One-time setup
git clone <repo>   # LFS files are fetched automatically
```

If you've already cloned without LFS, fetch the weights with:

```powershell
git lfs pull
```

## Commands

```powershell
bun install        # Install dependencies
bun run build      # Compile TypeScript
bun run check      # Run linter and formatter checks
bun run fmt        # Auto-fix lint and format issues
bun test           # Run tests
bun run dev        # Serve the demo page locally
```
