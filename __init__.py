from .seam_mask_node import GenerateSeamMask

NODE_CLASS_MAPPINGS = {
    "GenerateSeamMask": GenerateSeamMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateSeamMask": "Generate Seam Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
