# ComfyUI_UltimateSGUpscale

Tiled upscaling for ComfyUI using built-in nodes. Replicates the core features of [UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) as a transparent workflow you can inspect and modify.

## Requirements

- ComfyUI with `SplitImageToTileList` and `ImageMergeTileList` nodes (added in [PR #12599](https://github.com/comfyanonymous/ComfyUI/pull/12599))

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI_UltimateSGUpscale.git
```

This installs one custom node (`Generate Seam Mask`) and provides an example workflow.

## What's Included

### Example Workflow

`example_workflows/tiled-upscale-builtin-nodes.json` — a two-pass tiled upscaling workflow:

**Pass 1 — Tiled Redraw:** Upscales the image with a model (e.g. 4x-UltraSharp), splits it into overlapping tiles, runs each tile through KSampler, then merges them back with sine-based blending.

**Pass 2 — Seam Fix (optional):** Generates a mask targeting only the seam regions between tiles, then runs a second tiled denoise pass restricted to those seam bands via `SetLatentNoiseMask`. Mute or bypass the "Seam Fix" group to skip this pass.

### Generate Seam Mask Node

A small helper node that creates a binary mask image with white bands at tile seam positions. It replicates `SplitImageToTileList`'s tiling logic to place bands at the exact center of each overlap region.

**Inputs:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| image_width | 2048 | Image width (connect from GetImageSize) |
| image_height | 2048 | Image height (connect from GetImageSize) |
| tile_width | 1024 | Tile width matching Pass 1 |
| tile_height | 1024 | Tile height matching Pass 1 |
| overlap | 128 | Overlap matching Pass 1 |
| seam_width | 64 | Width of seam bands in pixels |

**Output:** `IMAGE` — a mask with white bands at seam positions, black elsewhere.

## How It Works

The workflow chains standard ComfyUI nodes together. `SplitImageToTileList` outputs a list, and ComfyUI's auto-iteration runs all downstream nodes (VAEEncode, KSampler, VAEDecode) once per tile automatically. Scalar inputs (model, conditioning, VAE) are reused across tiles. `ImageMergeTileList` reassembles tiles using sine-weighted blending for smooth overlap transitions.

The seam fix pass uses `SetLatentNoiseMask` to restrict denoising to only the masked seam regions, leaving the rest of the image untouched.

## License

MIT
