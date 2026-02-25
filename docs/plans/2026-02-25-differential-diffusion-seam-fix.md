# Differential Diffusion Seam Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add gradient mask mode to GenerateSeamMask and wire DifferentialDiffusion into the seam fix workflow pass.

**Architecture:** Add a `mode` combo input to GenerateSeamMask. In `gradient` mode, paint linear falloff bands instead of binary ones. In the workflow, insert a DifferentialDiffusion node wrapping the model before the seam fix KSampler.

**Tech Stack:** Python, PyTorch, ComfyUI workflow JSON

---

### Task 1: Add gradient mode tests

**Files:**
- Modify: `tests/test_seam_mask.py`

**Step 1: Write failing gradient tests**

Add these tests after the existing tests in `tests/test_seam_mask.py`:

```python
def test_binary_mode_explicit():
    """Existing behavior works when mode='binary' is passed explicitly."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="binary")
    mask = result[0]
    unique = mask.unique()
    assert len(unique) <= 2, f"Binary mode should only have 0.0 and 1.0, got {unique}"
    assert mask[0, 0, 960, 0].item() == 1.0, "Center should be white"


def test_gradient_center_is_one():
    """In gradient mode, the seam center should be 1.0."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    # Seam center at x=960
    assert mask[0, 0, 960, 0].item() == 1.0, "Gradient center should be 1.0"


def test_gradient_edge_is_zero():
    """In gradient mode, the band edge should be 0.0."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    # Seam center=960, half_w=32, band=[928,992)
    # Pixel 928 is at distance 32 from center -> value = 1 - 32/32 = 0.0
    assert mask[0, 0, 928, 0].item() == 0.0, "Band edge should be 0.0"
    assert mask[0, 0, 927, 0].item() == 0.0, "Outside band should be 0.0"


def test_gradient_midpoint():
    """Halfway between center and edge should be ~0.5."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    # Center=960, half_w=32. Pixel at 960-16=944 -> distance=16 -> value=1-16/32=0.5
    val = mask[0, 0, 944, 0].item()
    assert abs(val - 0.5) < 0.01, f"Midpoint should be ~0.5, got {val}"


def test_gradient_intersection_uses_max():
    """Where H and V seam bands cross, the value should be the max of both."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    # Both seams cross at (960, 960) — both are centers, so value should be 1.0
    assert mask[0, 960, 960, 0].item() == 1.0, "Intersection of two centers should be 1.0"
    # At (960, 944): vertical seam center (1.0), horizontal seam at distance 16 (0.5)
    # max(1.0, 0.5) = 1.0
    assert mask[0, 944, 960, 0].item() == 1.0, "On vertical center line, should be 1.0"


def test_gradient_no_seams_single_tile():
    """Gradient mode with single tile should also produce all zeros."""
    node = GenerateSeamMask()
    result = node.generate(image_width=512, image_height=512,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    assert mask.sum().item() == 0.0, "Single tile should have no seams in gradient mode"
```

Also update the `__main__` block to include the new tests, and update `test_values_are_binary` to pass `mode="binary"` explicitly.

**Step 2: Run tests to verify they fail**

Run: `cd /media/p5/ComfyUI_UltimateSGUpscale && python -m pytest tests/test_seam_mask.py -v`

Expected: New tests FAIL with `TypeError: generate() got an unexpected keyword argument 'mode'`. Existing tests still PASS (they don't pass `mode`).

**Step 3: Commit**

```bash
git add tests/test_seam_mask.py
git commit -m "test: add gradient mode tests for GenerateSeamMask"
```

---

### Task 2: Add mode parameter and gradient logic to GenerateSeamMask

**Files:**
- Modify: `seam_mask_node.py:6-21` (INPUT_TYPES — add mode combo)
- Modify: `seam_mask_node.py:44-70` (generate method — add mode parameter, gradient logic)

**Step 1: Add `mode` combo to INPUT_TYPES**

In `seam_mask_node.py`, add after the `seam_width` input (line 20), before the closing `}`:

```python
                "mode": (["binary", "gradient"], {"default": "binary",
                    "tooltip": "binary: hard 0/1 mask. gradient: linear falloff for use with Differential Diffusion."}),
```

**Step 2: Update the generate method**

Replace the `generate` method (lines 44-70) with:

```python
    def generate(self, image_width, image_height, tile_width, tile_height, overlap, seam_width, mode="binary"):
        mask = torch.zeros(1, image_height, image_width, 3)
        half_w = seam_width // 2

        # Compute actual tile grids (same logic as SplitImageToTileList)
        x_tiles = self._get_tile_positions(image_width, tile_width, overlap)
        y_tiles = self._get_tile_positions(image_height, tile_height, overlap)

        if mode == "gradient":
            # Build 1D linear ramps for each seam, then take max across all bands
            # Vertical seam bands
            for i in range(len(x_tiles) - 1):
                ovl_start = max(x_tiles[i][0], x_tiles[i + 1][0])
                ovl_end = min(x_tiles[i][1], x_tiles[i + 1][1])
                center = (ovl_start + ovl_end) // 2
                x_start = max(0, center - half_w)
                x_end = min(image_width, center + half_w)
                for x in range(x_start, x_end):
                    val = 1.0 - abs(x - center) / half_w
                    mask[:, :, x, :] = torch.max(mask[:, :, x, :], torch.tensor(val))

            # Horizontal seam bands
            for i in range(len(y_tiles) - 1):
                ovl_start = max(y_tiles[i][0], y_tiles[i + 1][0])
                ovl_end = min(y_tiles[i][1], y_tiles[i + 1][1])
                center = (ovl_start + ovl_end) // 2
                y_start = max(0, center - half_w)
                y_end = min(image_height, center + half_w)
                for y in range(y_start, y_end):
                    val = 1.0 - abs(y - center) / half_w
                    mask[:, y, :, :] = torch.max(mask[:, y, :, :], torch.tensor(val))
        else:
            # Binary mode (original behavior)
            for i in range(len(x_tiles) - 1):
                ovl_start = max(x_tiles[i][0], x_tiles[i + 1][0])
                ovl_end = min(x_tiles[i][1], x_tiles[i + 1][1])
                center = (ovl_start + ovl_end) // 2
                x_start = max(0, center - half_w)
                x_end = min(image_width, center + half_w)
                mask[:, :, x_start:x_end, :] = 1.0

            for i in range(len(y_tiles) - 1):
                ovl_start = max(y_tiles[i][0], y_tiles[i + 1][0])
                ovl_end = min(y_tiles[i][1], y_tiles[i + 1][1])
                center = (ovl_start + ovl_end) // 2
                y_start = max(0, center - half_w)
                y_end = min(image_height, center + half_w)
                mask[:, y_start:y_end, :, :] = 1.0

        return (mask,)
```

**Step 3: Run all tests**

Run: `cd /media/p5/ComfyUI_UltimateSGUpscale && python -m pytest tests/test_seam_mask.py -v`

Expected: ALL tests PASS (both old binary tests and new gradient tests).

**Step 4: Commit**

```bash
git add seam_mask_node.py
git commit -m "feat: add gradient mode to GenerateSeamMask for differential diffusion"
```

---

### Task 3: Update workflow JSON with DifferentialDiffusion node

**Files:**
- Modify: `example_workflows/tiled-upscale-builtin-nodes.json`

**Step 1: Add DifferentialDiffusion node and update wiring**

Changes to the workflow JSON:

1. Update `last_node_id` from 23 to 24
2. Update `last_link_id` from 37 to 39
3. In node 1 (CheckpointLoaderSimple), change MODEL output links from `[1, 2]` to `[1, 38]`
4. Add new node 24 (DifferentialDiffusion) positioned at `[2560, 160]` inside the Seam Fix group:

```json
{
    "id": 24,
    "type": "DifferentialDiffusion",
    "pos": [2560, 160],
    "size": [250, 46],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 38}
    ],
    "outputs": [
        {"name": "MODEL", "type": "MODEL", "slot_index": 0, "links": [39]}
    ],
    "properties": {"Node name for S&R": "DifferentialDiffusion"},
    "widgets_values": []
}
```

5. In node 19 (seam fix KSampler), change model input link from `2` to `39`
6. In node 13 (GenerateSeamMask), update `widgets_values` from `[2048, 2048, 1024, 1024, 128, 64]` to `[2048, 2048, 1024, 1024, 128, 64, "gradient"]`
7. Replace link `[2, 1, 0, 19, 0, "MODEL"]` with two new links:
   - `[38, 1, 0, 24, 0, "MODEL"]` (Checkpoint → DD)
   - `[39, 24, 0, 19, 0, "MODEL"]` (DD → Seam KSampler)
8. Increment `order` by 1 for all nodes whose current order >= 12 (to make room for DD at order 12)

**Step 2: Validate workflow JSON**

Run: `cd /media/p5/ComfyUI_UltimateSGUpscale && python3 -c "import json; json.load(open('example_workflows/tiled-upscale-builtin-nodes.json')); print('Valid JSON')"`

**Step 3: Verify no group overlap issues**

Run the group membership check script from the previous session to confirm node 24 is inside Group 5 only.

**Step 4: Commit**

```bash
git add example_workflows/tiled-upscale-builtin-nodes.json
git commit -m "feat: add DifferentialDiffusion node to seam fix workflow pass"
```

---

### Task 4: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update documentation**

Add a note about the gradient mode and differential diffusion in the GenerateSeamMask section:

- Add `mode` parameter to the inputs table: `mode | binary | binary: hard mask. gradient: linear falloff for Differential Diffusion.`
- Mention that the example workflow uses gradient mode with DifferentialDiffusion for smoother seam repairs.

**Step 2: Commit and push**

```bash
git add README.md
git commit -m "docs: document gradient mode and differential diffusion"
git push origin main
```
