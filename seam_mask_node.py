import torch


class GenerateSeamMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 1,
                    "tooltip": "Width of the image (from GetImageSize)."}),
                "image_height": ("INT", {"default": 2048, "min": 64, "max": 16384, "step": 1,
                    "tooltip": "Height of the image (from GetImageSize)."}),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Tile width used in the main tiled redraw pass."}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Tile height used in the main tiled redraw pass."}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 1,
                    "tooltip": "Overlap used in the main tiled redraw pass."}),
                "seam_width": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8,
                    "tooltip": "Width of the seam bands to fix (in pixels)."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Generates a mask image with white bands at tile seam positions. Used for targeted seam fix denoising."

    @staticmethod
    def _get_tile_positions(length, tile_size, overlap):
        """Compute 1D tile start/end positions, matching SplitImageToTileList's get_grid_coords."""
        stride = max(1, tile_size - overlap)
        positions = []
        p = 0
        while p < length:
            p_end = min(p + tile_size, length)
            p_start = max(0, p_end - tile_size)
            positions.append((p_start, p_end))
            if p_end >= length:
                break
            p += stride
        return positions

    def generate(self, image_width, image_height, tile_width, tile_height, overlap, seam_width):
        mask = torch.zeros(1, image_height, image_width, 3)
        half_w = seam_width // 2

        # Compute actual tile grids (same logic as SplitImageToTileList)
        x_tiles = self._get_tile_positions(image_width, tile_width, overlap)
        y_tiles = self._get_tile_positions(image_height, tile_height, overlap)

        # Vertical seam bands (between horizontally adjacent tiles)
        for i in range(len(x_tiles) - 1):
            ovl_start = max(x_tiles[i][0], x_tiles[i + 1][0])
            ovl_end = min(x_tiles[i][1], x_tiles[i + 1][1])
            center = (ovl_start + ovl_end) // 2
            x_start = max(0, center - half_w)
            x_end = min(image_width, center + half_w)
            mask[:, :, x_start:x_end, :] = 1.0

        # Horizontal seam bands (between vertically adjacent tiles)
        for i in range(len(y_tiles) - 1):
            ovl_start = max(y_tiles[i][0], y_tiles[i + 1][0])
            ovl_end = min(y_tiles[i][1], y_tiles[i + 1][1])
            center = (ovl_start + ovl_end) // 2
            y_start = max(0, center - half_w)
            y_end = min(image_height, center + half_w)
            mask[:, y_start:y_end, :, :] = 1.0

        return (mask,)
