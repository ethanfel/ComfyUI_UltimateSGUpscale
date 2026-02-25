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
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    # Stride = 1024 - 128 = 896
    # Seams at x=896, 1792 and y=896, 1792
    assert mask[0, 0, 896, 0].item() == 1.0, "Center of vertical seam should be white"
    assert mask[0, 896, 0, 0].item() == 1.0, "Center of horizontal seam should be white"
    assert mask[0, 0, 400, 0].item() == 0.0, "Far from any seam should be black"


def test_no_seams_single_tile():
    """If image fits in one tile, no seams should exist."""
    node = GenerateSeamMask()
    result = node.generate(image_width=512, image_height=512,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    assert mask.sum().item() == 0.0, "Single tile image should have no seams"


def test_seam_band_width():
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=1024,
                           tile_width=1024, tile_height=1024,
                           overlap=0, seam_width=64)
    mask = result[0]
    # Stride = 1024, seam at x=1024, band from 992 to 1056
    assert mask[0, 0, 1023, 0].item() == 1.0, "Inside band should be white"
    assert mask[0, 0, 991, 0].item() == 0.0, "Outside band should be black"


def test_values_are_binary():
    node = GenerateSeamMask()
    result = node.generate(image_width=2048, image_height=2048,
                           tile_width=1024, tile_height=1024,
                           overlap=128, seam_width=64)
    mask = result[0]
    unique = mask.unique()
    assert len(unique) <= 2, f"Mask should only contain 0.0 and 1.0, got {unique}"


if __name__ == "__main__":
    test_output_shape()
    test_seam_positions()
    test_no_seams_single_tile()
    test_seam_band_width()
    test_values_are_binary()
    print("All tests passed!")
