"""TeaCache port for FLUX.2 Klein (KV-cached variant in diffusers).

Adapted from the upstream FLUX.1 reference at
https://raw.githubusercontent.com/ali-vilab/TeaCache/main/TeaCache4FLUX/teacache_flux.py

The core idea is unchanged: estimate, via a polynomial fit on the relative-L1
distance of a "modulated input" probe, whether the residual that the block
stack would produce this step is close enough to the residual produced last
step that we can simply reuse the previous residual instead of running the
blocks. The first and last step of every batch are always force-computed.

================================================================================
WHAT IS KLEIN-SPECIFIC
================================================================================

1. CLASS NAME — `Flux2Transformer2DModel`, lives in
   `diffusers.models.transformers.transformer_flux2`. Verified against
   diffusers commit 160852de680d36117e0a787f7f8b718232539abb.

2. FORWARD SIGNATURE — Klein has NO `pooled_projections` and NO
   `controlnet_*_samples`. It DOES have `kv_cache`, `kv_cache_mode`,
   `num_ref_tokens`, `ref_fixed_timestep` for the reference-image KV cache.
   Time embedding is `time_guidance_embed(timestep, guidance)` (no pooled).

3. MODULATION FLOW — Klein computes three separate streams up front:
       double_stream_mod_img = self.double_stream_modulation_img(temb)
       double_stream_mod_txt = self.double_stream_modulation_txt(temb)
       single_stream_mod     = self.single_stream_modulation(temb)
   and passes them to blocks as `temb_mod_img`, `temb_mod_txt`, `temb_mod`.
   FLUX.1 just passes raw `temb` and modulation happens inside each block.

4. PROBE — In FLUX.1, `transformer_blocks[0].norm1` is `AdaLayerNormZero`,
   which returns 5-tuple `(modulated_inp, gate_msa, shift_mlp, scale_mlp,
   gate_mlp)`. In Klein, `norm1` is plain `nn.LayerNorm` (no modulation).
   The block itself does `(1 + scale_msa) * norm1(x) + shift_msa`. We
   reproduce that formula inline to get an equivalent "modulated input"
   probe — capturing the post-AdaLN signal that drives the rest of the
   stack. Coefficients are kept FLUX-tuned per the brief; if they prove
   off, re-fit on Klein's actual L1 trajectory.

5. BLOCK SIGNATURES — different. See block-loop below.

6. POSITIONAL EMBEDDINGS — Klein computes `pos_embed(img_ids)` and
   `pos_embed(txt_ids)` separately, then concatenates the (cos, sin)
   tuples. FLUX.1 cats the ids first and embeds once.

7. OUTPUT STRIP — Klein uses `hidden_states[:, num_txt_tokens:]`
   (plus `+ num_ref_tokens` in extract mode). FLUX.1 strips by
   `encoder_hidden_states.shape[1]`.

================================================================================
KV-CACHE SAFETY (THIS IS THE CRITICAL PART)
================================================================================

`Flux2KleinKVPipeline` extracts reference-image K/V on the first denoising
step (`kv_cache_mode="extract"`) and reads from it on every subsequent step
(`kv_cache_mode="cached"`). The KV WRITE happens INSIDE the inner block
loop, deep in `Flux2KVAttnProcessor.__call__` / `Flux2KVParallelSelfAttn-
Processor.__call__`, gated on `kv_cache_mode == "extract" and num_ref_tokens > 0`.

CRUCIAL OBSERVATION: the WRITE only fires on the extract step. Cached-mode
steps (the ones where TeaCache might skip) only READ from the cache —
they never mutate it.

Because TeaCache force-recomputes on `cnt == 0`, the extract step is
always a full forward and the KV cache is always populated correctly.
For cached-mode steps, skipping the block loop means we don't perform the
attention reads either, but those are SIDE-EFFECT-FREE: their only output
goes into `hidden_states`, which we substitute with `previous_residual +
hidden_states`. So skipping is safe.

Sanity check: if a future Klein variant ever started writing KV during
cached-mode steps (e.g. some accumulating cache), this assumption breaks
and we would need to call into the blocks anyway. We guard against that
by asserting `kv_cache_mode != "extract"` whenever we take the skip
branch. If extract ever fires past step 0, raise.

================================================================================
COMPOSITION WITH torch.compile AND torchao fp8
================================================================================

torch.compile: TeaCache has data-dependent control flow:
    - `if not should_calc:` branches on a Python bool that depends on
      `(modulated_inp - prev).abs().mean() / prev.abs().mean()).cpu().item()`
      which forces a CPU sync per step.
    - `should_calc` flips per step based on accumulated state.

This is HOSTILE to graph capture. Two options:

  (A) RECOMMENDED for now: skip torch.compile entirely on the transformer
      and use TeaCache alone. TeaCache's residual-skip gain (~1.8x at
      thresh=0.4) often dominates compile gain at 4 steps where compile
      can only amortize 4 GPU graphs anyway. Keep compile on
      vae.encoder/vae.decoder which have no data-dependent control.

  (B) ADVANCED: compile only the inner block-loop sub-graph (compile each
      Flux2TransformerBlock and Flux2SingleTransformerBlock individually,
      NOT the wrapping forward). The patched forward stays uncompiled
      Python and dispatches into compiled blocks. This requires moving
      compile calls from `pipe.transformer = torch.compile(...)` to a
      per-block loop AFTER `apply_teacache_to_klein(pipe)`. Untested.

The __main__ block below uses option (A): no compile on transformer.

torchao fp8 (Float8DynamicActivationFloat8WeightConfig):
  - Quantize FIRST, patch SECOND. quantize_() walks linear layers and
    swaps in fp8 wrappers, which works fine because we don't change
    layer types — we only override the class-level forward method and
    set instance attrs. The patched forward calls `self.x_embedder`,
    `self.transformer_blocks[i]`, etc. which are all unmodified module
    references: whatever quantize_ swapped in is what runs.
  - If you patch first and quantize second, also fine. Order doesn't
    actually matter here because the patch touches forward() not the
    submodule registry. The __main__ block does quantize -> patch.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

# Import the actual Klein transformer class so we can monkey-patch it.
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    Flux2KVCache,
    Flux2Modulation,
    _blend_double_block_mods,
    _blend_single_block_mods,
)


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------

def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
    kv_cache: "Flux2KVCache | None" = None,
    kv_cache_mode: str | None = None,
    num_ref_tokens: int = 0,
    ref_fixed_timestep: float = 0.0,
) -> torch.Tensor | Flux2Transformer2DModelOutput:
    """TeaCache-augmented forward for Flux2Transformer2DModel.

    Mirrors the upstream Flux2Transformer2DModel.forward (commit
    160852de) but wraps the block stack in a residual-cache check.
    """
    # ----- Klein-specific input shape: no pooled_projections -----
    # FLUX.1 has `pooled_projections`; Klein folds everything into temb
    # via time_guidance_embed.
    num_txt_tokens = encoder_hidden_states.shape[1]

    # ----- Step 1: temb + modulation streams (Klein-specific) -----
    # FLUX.1 does: temb = time_text_embed(timestep, [guidance,] pooled)
    # Klein does: temb = time_guidance_embed(timestep, guidance)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    # Klein-specific: three separate modulation streams up front.
    # In FLUX.1 the raw `temb` flows into each block and modulation
    # happens internally via AdaLayerNormZero.
    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    # ----- KV-cache extract setup (Klein-specific) -----
    # Replicated verbatim from upstream Flux2Transformer2DModel.forward
    # because the block loop depends on these blended modulations and on
    # `kv_cache` being populated. We MUST run this before the TeaCache
    # decision so that the cache exists if extract mode is requested.
    ref_single_mod = None  # populated below if extract
    if kv_cache_mode == "extract" and num_ref_tokens > 0:
        num_img_tokens = hidden_states.shape[1]  # includes ref tokens

        kv_cache = Flux2KVCache(
            num_double_layers=len(self.transformer_blocks),
            num_single_layers=len(self.single_transformer_blocks),
        )
        kv_cache.num_ref_tokens = num_ref_tokens

        ref_timestep = torch.full_like(timestep, ref_fixed_timestep * 1000)
        ref_temb = self.time_guidance_embed(ref_timestep, guidance)

        ref_double_mod_img = self.double_stream_modulation_img(ref_temb)
        ref_single_mod = self.single_stream_modulation(ref_temb)

        double_stream_mod_img = _blend_double_block_mods(
            double_stream_mod_img, ref_double_mod_img, num_ref_tokens, num_img_tokens
        )

    # ----- Step 2: input projection (Klein matches FLUX.1) -----
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # ----- Step 3: RoPE (Klein-specific: separate then concat) -----
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    # ----- TeaCache decision (probe + residual cache) -----
    # KEY ADAPTATION FROM FLUX.1:
    # FLUX.1 does:
    #   modulated_inp, *_ = self.transformer_blocks[0].norm1(inp, emb=temb_)
    # because FLUX.1's norm1 is AdaLayerNormZero returning a 5-tuple.
    # Klein's norm1 is plain nn.LayerNorm with NO modulation built-in;
    # the block itself applies (1 + scale_msa) * norm + shift_msa using
    # external Flux2Modulation params. We replicate that here to get
    # a probe that captures the same post-AdaLN signal.
    #
    # SHAPE NOTE: On the extract step, `hidden_states` has shape
    # [B, num_ref + num_img, dim] because the pipeline prepends ref
    # tokens. On all subsequent (cached) steps it's [B, num_img, dim].
    # We compare probes across steps, so we slice to image-only tokens
    # to keep shapes consistent. We do the same for `previous_residual`.
    if self.enable_teacache:
        # Compute the modulated probe the same way the first block will.
        # double_stream_mod_img has 2 param sets (msa + mlp); we need
        # only the msa shift/scale for the probe, matching FLUX.1's
        # use of `modulated_inp` (which was the post-AdaLN, pre-attn signal).
        (shift_msa, scale_msa, _gate_msa), _ = Flux2Modulation.split(
            double_stream_mod_img, 2
        )
        # `self.transformer_blocks[0].norm1` is nn.LayerNorm(elementwise_affine=False)
        norm1 = self.transformer_blocks[0].norm1
        # In extract mode `scale_msa` and `shift_msa` are blended along seq
        # (ref-positions get ref-mod, img-positions get img-mod). After we
        # slice to image-only, we need the corresponding image-only mod
        # slices — but Flux2Modulation.split returns shape [B, 1, dim] for
        # plain (non-blended) mod. For the blended case it's [B, S, dim].
        # Either way, slicing the LAST `num_img_only` positions of both
        # tensor and mod is correct (broadcast handles the [B, 1, dim] case).
        modulated_inp_full = norm1(hidden_states)
        modulated_inp_full = (1 + scale_msa) * modulated_inp_full + shift_msa

        # Slice to image-only tokens for shape stability across steps.
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            modulated_inp = modulated_inp_full[:, num_ref_tokens:, :]
        else:
            modulated_inp = modulated_inp_full

        # KV-cache safety guard (see module docstring for proof of safety).
        # Force-recompute conditions:
        #   - cnt == 0: extract step; full KV cache write must happen.
        #   - cnt == num_steps - 1: last step; FLUX.1 reference behavior.
        #   - cnt == 1: first cached-mode step; the probe from cnt=0 used
        #     extract-mode blended modulation (ref+img), making the L1
        #     comparison apples-to-oranges. Force one cached-mode probe
        #     before allowing skip decisions.
        #   - prev_* is None: just-reset state (between batches).
        if (
            self.cnt == 0
            or self.cnt == 1
            or self.cnt == self.num_steps - 1
            or self.previous_modulated_input is None
            or self.previous_residual is None
        ):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # FLUX-tuned coefficients (kept per the brief; re-fit later).
            coefficients = [4.987e02, -2.838e02, 5.586e01, -3.820e00, 2.642e-01]
            rescale_func = np.poly1d(coefficients)
            rel_l1 = (
                (modulated_inp - self.previous_modulated_input).abs().mean()
                / self.previous_modulated_input.abs().mean()
            ).cpu().item()
            self.accumulated_rel_l1_distance += rescale_func(rel_l1)
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    else:
        should_calc = True

    # ----- Block stack (with optional residual reuse) -----
    if self.enable_teacache and not should_calc:
        # SAFETY: skipping the block loop is only OK if no KV write was
        # about to happen. Klein only writes KV in extract mode (cnt==0,
        # which always force-recomputes), so this branch is only reached
        # in cached mode where `hidden_states` is already image-only.
        if kv_cache_mode == "extract":
            raise RuntimeError(
                "TeaCache asked to skip blocks during kv_cache_mode='extract'. "
                "This would silently drop the KV cache write. cnt=0 must always "
                "force-recompute; check that reset_teacache() was called between "
                "batches and that extract mode only fires on step 0."
            )
        # Reuse: residual was captured on the previous full-compute step
        # in image-only shape, matching `hidden_states` here.
        hidden_states = hidden_states + self.previous_residual
    else:
        # Capture pre-block-stack hidden_states for residual delta. In
        # extract mode this includes leading ref tokens; we'll slice
        # those off when storing the residual so it's shape-compatible
        # with cached-mode skip steps.
        ori_hidden_states = hidden_states.clone()

        # Build joint_attention_kwargs with KV cache info for blocks
        # (verbatim from upstream Klein forward).
        if kv_cache_mode == "extract":
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            kv_attn_kwargs = joint_attention_kwargs

        # ----- Double-stream blocks (Klein signature differs from FLUX.1) -----
        # FLUX.1 block call:
        #   block(hidden_states, encoder_hidden_states, temb, image_rotary_emb,
        #         joint_attention_kwargs)
        # Klein block call:
        #   block(hidden_states, encoder_hidden_states, temb_mod_img,
        #         temb_mod_txt, image_rotary_emb, joint_attention_kwargs)
        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=kv_attn_kwargs,
            )

        # Concat for single-stream phase (Klein-specific layout)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Blend single-block mods if extract (Klein-specific)
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods(
                single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
            )

        if kv_cache_mode is not None:
            kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            kv_attn_kwargs_single = kv_attn_kwargs

        # ----- Single-stream blocks (Klein signature differs from FLUX.1) -----
        # FLUX.1: block(hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
        # Klein:  block(hidden_states, encoder_hidden_states=None,
        #               temb_mod=single_stream_mod, image_rotary_emb,
        #               joint_attention_kwargs)
        for index_block, block in enumerate(self.single_transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=kv_attn_kwargs_single,
            )

        # ----- Strip text (and ref) tokens (Klein-specific indexing) -----
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens :, ...]
        else:
            hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # Cache residual for next step's potential reuse. Make sure
        # both sides are sliced to image-only tokens — `hidden_states`
        # already is (post-strip), and `ori_hidden_states` needs to
        # have its leading ref-token rows dropped on extract steps.
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            ori_image_only = ori_hidden_states[:, num_ref_tokens:, :]
        else:
            ori_image_only = ori_hidden_states
        self.previous_residual = hidden_states - ori_image_only

    # ----- Output layers (Klein matches FLUX.1) -----
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # ----- Return (Klein-specific extract-mode tuple) -----
    if kv_cache_mode == "extract":
        if not return_dict:
            return (output, kv_cache)
        return Flux2Transformer2DModelOutput(sample=output, kv_cache=kv_cache)

    if not return_dict:
        return (output,)
    return Flux2Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_teacache_to_klein(
    pipe,
    rel_l1_thresh: float = 0.4,
    num_steps: int = 4,
) -> None:
    """Patch pipe.transformer in-place to use TeaCache residual reuse.

    Mutates the Flux2Transformer2DModel CLASS forward method (so all
    instances share it) and sets per-instance state on `pipe.transformer`.

    Args:
        pipe: A Flux2KleinKVPipeline (or any pipeline whose `.transformer`
            is a `Flux2Transformer2DModel`).
        rel_l1_thresh: Skip threshold for accumulated rescaled rel-L1.
            FLUX.1 reference values: 0.25 ~ 1.5x speedup, 0.4 ~ 1.8x,
            0.6 ~ 2.0x, 0.8 ~ 2.25x. Klein at 4 steps probably wants
            0.3-0.5; tune empirically.
        num_steps: Number of denoising steps per call. MUST match the
            sigmas length / num_inference_steps you'll pass to the pipe.

    Note: torch.compile and TeaCache compose poorly because TeaCache
    has data-dependent control flow and CPU syncs (`.item()`). Prefer
    NOT compiling the transformer when TeaCache is active. See module
    docstring for advanced compose strategies.

    HONEST CAVEAT FOR LOW STEP COUNTS:
    TeaCache forces recompute on cnt=0 (KV extract), cnt=1 (first
    cached-mode step, probe-shape consistency), and cnt=num_steps-1.
    At num_steps=4 only cnt=2 is eligible to skip — at most 25%
    speedup if it always skips. The expected upside grows fast with
    step count: at 8 steps, cnt=2..6 (5 skippable) → up to ~60%.
    If you measure and the 4-step gain is <10%, TeaCache is the wrong
    tool here — try FORA, Pruna, or simply staying at 2-step.
    """
    transformer = pipe.transformer

    # Verify class. We catch torch.compile-wrapped transformers
    # (OptimizedModule) by looking through `_orig_mod`.
    target = transformer
    if hasattr(target, "_orig_mod"):
        target = target._orig_mod
    if not isinstance(target, Flux2Transformer2DModel):
        raise TypeError(
            f"apply_teacache_to_klein: expected Flux2Transformer2DModel, "
            f"got {type(target).__name__}. If you wrapped with "
            f"torch.compile, see the module docstring — compile composes "
            f"poorly with TeaCache."
        )

    # Class-level forward swap (matches upstream FLUX.1 reference style).
    Flux2Transformer2DModel.forward = teacache_forward

    # Per-instance state.
    transformer.enable_teacache = True
    transformer.cnt = 0
    transformer.num_steps = int(num_steps)
    transformer.rel_l1_thresh = float(rel_l1_thresh)
    transformer.accumulated_rel_l1_distance = 0.0
    transformer.previous_modulated_input = None
    transformer.previous_residual = None


def reset_teacache(pipe) -> None:
    """Reset TeaCache step counter and residual cache between batches.

    `cnt` accumulates across forward calls; without reset, the second
    batch would start at cnt != 0 and skip the force-recompute on its
    own first step (where Klein extracts the KV cache). That would
    raise the safety RuntimeError in `teacache_forward`.

    Call this before each new image-to-image call (i.e. between
    `pipe(...)` invocations).
    """
    transformer = pipe.transformer
    if hasattr(transformer, "_orig_mod"):
        transformer = transformer._orig_mod
    transformer.cnt = 0
    transformer.accumulated_rel_l1_distance = 0.0
    transformer.previous_modulated_input = None
    transformer.previous_residual = None


# ---------------------------------------------------------------------------
# Smoke benchmark — run with `python teacache_klein.py` on RunPod
# ---------------------------------------------------------------------------

def _run_smoke_bench():
    """Load Klein, apply TeaCache + fp8 (no compile on transformer),
    run 5 timed iters at 256² / 4-step and print mean ms."""
    from pathlib import Path

    import diffusers
    from PIL import Image
    from diffusers import AutoencoderKLFlux2, Flux2KleinKVPipeline
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_latents

    KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"
    DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
    WAVE_PATH = Path("/workspace/waveforms/waveform_1.png")
    SIZE = 256
    N_STEPS = 4
    ALPHA = 0.10
    SEED = 42
    PROMPT = (
        "a bright white lightning bolt against a pitch black night sky, "
        "dramatic, photographic, high contrast"
    )
    REL_L1_THRESH = 0.4
    WARMUP = 3
    TIMED = 5

    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[init] diffusers={diffusers.__version__}", flush=True)
    print(f"[init] device={torch.cuda.get_device_name(0)}", flush=True)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[load] {KLEIN_REPO}", flush=True)
    t0 = time.perf_counter()
    pipe = Flux2KleinKVPipeline.from_pretrained(KLEIN_REPO, torch_dtype=torch.bfloat16)
    pipe.vae = AutoencoderKLFlux2.from_pretrained(DECODER_REPO, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ----- torchao fp8 BEFORE TeaCache patch -----
    # quantize_ swaps Linear weights/activations, leaves module structure intact;
    # our patch only overrides class.forward and sets instance attrs, so the
    # two compose without conflict. Order doesn't actually matter — see docstring.
    try:
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        print("[fp8] applying Float8DynamicActivationFloat8WeightConfig to transformer", flush=True)
        quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
        print("[fp8] done", flush=True)
    except ImportError:
        print("[fp8] torchao not available; skipping fp8", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fp8] failed: {type(e).__name__}: {e}; continuing in bf16", flush=True)

    # ----- VAE compile (no data-dependent control flow there) -----
    print("[compile] vae.encoder + vae.decoder (mode=default)", flush=True)
    pipe.vae.encoder = torch.compile(pipe.vae.encoder, mode="default", fullgraph=False, dynamic=False)
    pipe.vae.decoder = torch.compile(pipe.vae.decoder, mode="default", fullgraph=False, dynamic=False)
    # NOTE: deliberately NOT compiling pipe.transformer — TeaCache's CPU sync
    # and data-dependent skip break graph capture. See module docstring.

    # ----- TeaCache patch -----
    print(f"[teacache] applying (rel_l1_thresh={REL_L1_THRESH}, num_steps={N_STEPS})", flush=True)
    apply_teacache_to_klein(pipe, rel_l1_thresh=REL_L1_THRESH, num_steps=N_STEPS)

    # ----- Prompt encode -----
    r = pipe.encode_prompt(prompt=PROMPT, device="cuda", num_images_per_prompt=1, max_sequence_length=64)
    prompt_embeds = r[0] if isinstance(r, tuple) else r

    # ----- Input image -----
    if WAVE_PATH.exists():
        wave_pil = Image.open(WAVE_PATH).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    else:
        print(f"[warn] {WAVE_PATH} missing; using random noise as input", flush=True)
        wave_pil = Image.fromarray(
            (np.random.rand(SIZE, SIZE, 3) * 255).astype(np.uint8)
        )

    def encode_img(img_pil):
        a = np.asarray(img_pil, dtype=np.float32) / 127.5 - 1.0
        t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        raw = retrieve_latents(pipe.vae.encode(t), sample_mode="argmax")
        patch = pipe._patchify_latents(raw)
        m = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(patch.device, patch.dtype)
        s = (pipe.vae.bn.running_var + pipe.vae.bn.eps).sqrt().view(1, -1, 1, 1).to(patch.device, patch.dtype)
        return (patch - m) / s

    def run_one(seed):
        # MUST reset TeaCache between calls — cnt accumulates and the
        # next call must see cnt=0 (force-recompute on extract step).
        reset_teacache(pipe)

        lat = encode_img(wave_pil)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn(lat.shape, generator=gen, dtype=lat.dtype, device="cuda")
        noisy = ALPHA * lat + (1 - ALPHA) * noise
        sigmas = np.linspace(1 - ALPHA, 0.0, N_STEPS).tolist()
        return pipe(
            image=None,
            prompt=None,
            prompt_embeds=prompt_embeds,
            latents=noisy,
            sigmas=sigmas,
            height=SIZE,
            width=SIZE,
            num_inference_steps=N_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]

    # ----- Warmup -----
    print(f"[warmup] {WARMUP} iters", flush=True)
    for w in range(WARMUP):
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = run_one(SEED)
        torch.cuda.synchronize()
        print(f"  warmup {w + 1}/{WARMUP}: {(time.perf_counter() - t) * 1000:.1f}ms", flush=True)

    # ----- Timed -----
    print(f"[timed] {TIMED} iters", flush=True)
    lats = []
    for r in range(TIMED):
        torch.cuda.synchronize()
        t = time.perf_counter()
        _ = run_one(SEED + r)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 1000
        lats.append(ms)
        print(f"  iter {r + 1}/{TIMED}: {ms:.1f}ms", flush=True)

    mean = sum(lats) / len(lats)
    print(
        f"\n[result] mean={mean:.1f}ms ({1000 / mean:.1f} fps) "
        f"min={min(lats):.1f} max={max(lats):.1f} "
        f"@ {SIZE}^2 / {N_STEPS}-step / thresh={REL_L1_THRESH}",
        flush=True,
    )


if __name__ == "__main__":
    _run_smoke_bench()
