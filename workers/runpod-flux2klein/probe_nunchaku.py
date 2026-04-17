#!/usr/bin/env python3
"""Verbatim reproduction of the nunchaku official example to isolate env issues."""
import torch
import time
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.models.transformers.transformer_zimage import NunchakuZImageRopeHook
from nunchaku.utils import get_precision, is_turing


# Patch: nunchaku 1.3.0dev calls super().forward positionally against a signature
# that moved args. Force kwarg-based call to match current diffusers.
def _fixed_forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1, return_dict=True, **kw):
    rope_hook = NunchakuZImageRopeHook()
    self.register_rope_hook(rope_hook)
    try:
        return ZImageTransformer2DModel.forward(
            self, x, t, cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
    finally:
        self.unregister_rope_hook()
        del rope_hook

NunchakuZImageTransformer2DModel.forward = _fixed_forward

precision = get_precision()
rank = 128
dtype = torch.float16 if is_turing() else torch.bfloat16
print(f"precision={precision}, rank={rank}, dtype={dtype}", flush=True)

print("[load] transformer", flush=True)
t0 = time.perf_counter()
transformer = NunchakuZImageTransformer2DModel.from_pretrained(
    f"nunchaku-ai/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",
    torch_dtype=dtype,
)
print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

print("[load] pipe", flush=True)
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=dtype, low_cpu_mem_usage=False,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"vram={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

print("[run] 1024x1024 8 steps", flush=True)
t0 = time.perf_counter()
image = pipe(
    prompt="a bright white lightning bolt against a pitch black night sky, dramatic",
    height=1024, width=1024,
    num_inference_steps=8, guidance_scale=0.0,
    generator=torch.Generator().manual_seed(12345),
).images[0]
dt = (time.perf_counter() - t0) * 1000
image.save(f"/workspace/probe_nunchaku_{precision}_r{rank}.png")
print(f"[run] SUCCESS in {dt:.0f}ms", flush=True)
