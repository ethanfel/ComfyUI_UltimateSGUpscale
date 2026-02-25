import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seam_mask_node import GenerateSeamMask


def test_output_shape():
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    assert mask.shape == (1, 2048, 2048, 3), f"Expected (1, 2048, 2048, 3), got {mask.shape}"


def test_seam_positions():
    """Seam bands should be centered at the overlap midpoint between adjacent tiles."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    # Tiles: [0,1024), [896,1920), [1024,2048)
    # Overlap between tile 0 and 1: [896, 1024), center=960
    # Band should be at [928, 992)
    assert mask[0, 0, 960, 0].item() == 1.0, "Center of overlap (960) should be white"
    assert mask[0, 960, 0, 0].item() == 1.0, "Horizontal seam center should be white"
    assert mask[0, 0, 400, 0].item() == 0.0, "Far from any seam should be black"
    # Old wrong position (stride=896) should NOT be in the band
    assert mask[0, 0, 896, 0].item() == 0.0, "Start of overlap (896) should be outside the band"


def test_no_seams_single_tile():
    """If image fits in one tile, no seams should exist."""
    node = GenerateSeamMask()
    result = node.generate(image_width=512, image_height=512,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    assert mask.sum().item() == 0.0, "Single tile image should have no seams"


def test_seam_band_width_no_overlap():
    """With overlap=0, seam center is at tile boundary."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=0, seam_width=64)
    mask = result[0]
    # Tiles: [0,1024), [1024,2048). Overlap: [1024,1024) = empty.
    # Center at 1024, band [992, 1056)
    assert mask[0, 0, 1024, 0].item() == 1.0, "Seam center should be white"
    assert mask[0, 0, 991, 0].item() == 0.0, "Outside band should be black"


def test_no_spurious_bands():
    """Should not generate bands beyond the actual tile grid."""
    node = GenerateSeamMask()
    # 2816px with 1024 tiles, stride=896: 3 tiles, 2 seams
    result = node.generate(image_width=2816, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    # Tiles: [0,1024), [896,1920), [1792,2816) — 3 tiles, 2 vertical seams
    # Seam 0-1: overlap [896,1024), center=960
    # Seam 1-2: overlap [1792,1920), center=1856
    assert mask[0, 0, 960, 0].item() == 1.0, "Seam 0-1 center should be white"
    assert mask[0, 0, 1856, 0].item() == 1.0, "Seam 1-2 center should be white"
    # x=2688 was a spurious band in the old code — should be black now
    assert mask[0, 0, 2688, 0].item() == 0.0, "No spurious band beyond tile grid"


def test_edge_tile_seam_position():
    """Edge tile seam should be at the actual overlap center, not at n*stride."""
    node = GenerateSeamMask()
    # 2048px: tiles [0,1024), [896,1920), [1024,2048)
    # Edge seam between tile 1 and 2: overlap [1024,1920), center=1472
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    assert mask[0, 0, 1472, 0].item() == 1.0, "Edge tile seam center (1472) should be white"
    # Old wrong position
    assert mask[0, 0, 1792, 0].item() == 0.0, "Old position (1792) should be black"


def test_values_are_binary():
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="binary")
    mask = result[0]
    unique = mask.unique()
    assert len(unique) <= 2, f"Mask should only contain 0.0 and 1.0, got {unique}"


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
    assert mask[0, 0, 960, 0].item() == 1.0, "Gradient center should be 1.0"


def test_gradient_edge_is_zero():
    """In gradient mode, the band edge should be 0.0."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    assert mask[0, 0, 928, 0].item() == 0.0, "Band edge should be 0.0"
    assert mask[0, 0, 927, 0].item() == 0.0, "Outside band should be 0.0"


def test_gradient_midpoint():
    """Halfway between center and edge should be ~0.5."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    val = mask[0, 0, 944, 0].item()
    assert abs(val - 0.5) < 0.01, f"Midpoint should be ~0.5, got {val}"


def test_gradient_intersection_uses_max():
    """Where H and V seam bands cross, the value should be the max of both."""
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    assert mask[0, 960, 960, 0].item() == 1.0, "Intersection of two centers should be 1.0"
    assert mask[0, 944, 960, 0].item() == 1.0, "On vertical center line, should be 1.0"


def test_gradient_no_seams_single_tile():
    """Gradient mode with single tile should also produce all zeros."""
    node = GenerateSeamMask()
    result = node.generate(image_width=512, image_height=512,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64, mode="gradient")
    mask = result[0]
    assert mask.sum().item() == 0.0, "Single tile should have no seams in gradient mode"


if __name__ == "__main__":
    test_output_shape()
    test_seam_positions()
    test_no_seams_single_tile()
    test_seam_band_width_no_overlap()
    test_no_spurious_bands()
    test_edge_tile_seam_position()
    test_values_are_binary()
    test_binary_mode_explicit()
    test_gradient_center_is_one()
    test_gradient_edge_is_zero()
    test_gradient_midpoint()
    test_gradient_intersection_uses_max()
    test_gradient_no_seams_single_tile()
    print("All tests passed!")
