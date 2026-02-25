# Differential Diffusion Seam Fix

## Problem

The current seam fix pass uses binary masks (1.0/0.0) with `SetLatentNoiseMask`. This creates hard transitions at band edges that can themselves become visible artifacts. Differential diffusion allows gradient masks where the value controls per-pixel denoise intensity, producing smoother seam repairs.

## Design

### GenerateSeamMask Node Changes

Add a `mode` combo input:

- **`binary`** (default): Current behavior. Output is 1.0 inside seam bands, 0.0 outside.
- **`gradient`**: Linear falloff from 1.0 at seam center to 0.0 at band edge. Value at distance `d` from center: `max(0, 1.0 - d / half_w)`. Where horizontal and vertical bands overlap (grid intersections), take `max` of both values.

The `seam_width` parameter keeps the same meaning in both modes.

### Workflow Changes

Add one `DifferentialDiffusion` node (node 24) inside the Seam Fix group. It wraps the model before it reaches the seam fix KSampler:

- Checkpoint → DifferentialDiffusion → Seam Fix KSampler (replaces direct Checkpoint → KSampler link)
- All other wiring unchanged. `SetLatentNoiseMask` still passes the mask to the latent.

### Tests

- Existing binary tests pass with explicit `mode="binary"`
- Gradient tests: center=1.0, edge=0.0, midpoint~0.5, intersection uses max
