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

    def generate(self, image_width, image_height, tile_width, tile_height, overlap, seam_width):
        mask = torch.zeros(1, image_height, image_width, 3)

        stride_x = max(1, tile_width - overlap)
        stride_y = max(1, tile_height - overlap)
        half_w = seam_width // 2

        # Vertical seam bands
        x = stride_x
        while x < image_width:
            x_start = max(0, x - half_w)
            x_end = min(image_width, x + half_w)
            mask[:, :, x_start:x_end, :] = 1.0
            x += stride_x

        # Horizontal seam bands
        y = stride_y
        while y < image_height:
            y_start = max(0, y - half_w)
            y_end = min(image_height, y + half_w)
            mask[:, y_start:y_end, :, :] = 1.0
            y += stride_y

        return (mask,)
