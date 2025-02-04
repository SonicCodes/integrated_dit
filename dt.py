import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import click
import torch.nn.functional as F
from einops import rearrange
import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        B, T = t.shape
        t = t.view(B*T, 1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(t.dtype))
        return t_emb.view(B, T, -1)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


def create_cached_causal_temporal_mask(
    cache_frames: int,
    new_frames: int,
    H: int,
    W: int,
    num_heads: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Creates a causal temporal mask for use when caching keys/values.

    Args:
        cache_frames (int): Number of frames already cached.
        new_frames (int): Number of new frames (i.e. queries).
        H (int): Height (in patches).
        W (int): Width (in patches).
        num_heads (int): Number of attention heads.
        device (torch.device, optional): Device on which to create the mask.

    Returns:
        torch.Tensor: A mask of shape [1, num_heads, new_frames*H*W, (cache_frames+new_frames)*H*W]
                    that is True where attention is allowed (i.e. for tokens in the same or earlier frame)
                    and False where it is not.
    """
    total_frames = cache_frames + new_frames
    total_tokens = total_frames * (H * W)
    query_tokens = new_frames * (H * W)

    # Create an index per token corresponding to its frame.
    frame_indices = torch.repeat_interleave(
        torch.arange(total_frames, device=device), H * W
    )  # Shape: (total_tokens,)

    # Build the full causal mask over all tokens.
    # For each token (row), keys with frame index <= token's frame index are allowed.
    full_mask = frame_indices[None, :] <= frame_indices[:, None]  # Shape: (total_tokens, total_tokens)

    # For the new query tokens, select the rows corresponding to the last query_tokens.
    cached_mask = full_mask[-query_tokens:, :]  # Shape: (query_tokens, total_tokens)

    # Expand to [1, num_heads, query_tokens, total_tokens]
    cached_mask = cached_mask.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1, -1)
    return cached_mask

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            resample_ratio: float = 1.0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.resample_ratio = resample_ratio
        self.rope = RotaryEmbedding(
                dim = 16,
                freqs_for = 'pixel',
                max_freq = 256
            )
    def forward(self, x: torch.Tensor, grid_info, mask=None, kv_cache=None) -> torch.Tensor:
        (T, H, W) = grid_info
  
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, H, N, D)
        q, k, v = qkv.unbind(0) # (B, H, N, D)
        q, k = self.q_norm(q), self.k_norm(k)

        using_cache=False
        

        # kT =  k.shape[2] // H*W


        def reshape_to_video(x, H, W):
            return rearrange(x, 'b x (T H W) d -> b x T H W d', H=H, W=W)

        def reshape_to_tokens(x, H, W):
            return rearrange(x, 'b x T H W d -> b x (T H W) d',  H=H, W=W)

        q = reshape_to_video(q, H, W)
        k = reshape_to_video(k, H, W)
        v = reshape_to_video(v, H, W)

        if kv_cache is not None:
            (k_cache, v_cache) = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            using_cache=True

        kv_cache = (k[:, :, :-1], v[:, :, :-1])

        kT = k.shape[2] 

        q_freqs = self.rope.get_axial_freqs(kT, H, W)
        k_freqs = self.rope.get_axial_freqs(kT, H, W)

        # repeat T for q, till it reaches, k's if using cache
        # print("pre q:", q.shape)
        if using_cache:
            q = q.repeat(1, 1, kT, 1, 1, 1)
        q = apply_rotary_emb(q_freqs, q)
        if using_cache:
            q = q[:, :, -1:]
        k = apply_rotary_emb(k_freqs, k)

        q = reshape_to_tokens(q, H, W)
        k = reshape_to_tokens(k, H, W)
        v = reshape_to_tokens(v, H, W)

        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
            attn_mask=mask if not using_cache else None,
            is_causal=False
        )

        x = reshape_to_video(x, H, W)
        # print("x shape: ", x.shape)
        # if using_cache:
        #     x = x[:, :, -1:]

        x = reshape_to_tokens(x, H, W)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv_cache



def resample_tokens(x, grid_info, resample_ratio):
    (T, H, W) = grid_info
    # convert from (B, F*H*W, C) to (B, F, H, W, C)
    x = rearrange(x, 'b (f h w) c -> (b f) c h w', f=T, h=H, w=W)
    # downsample
    x = F.interpolate(x, size=(int(H//resample_ratio), int(W//resample_ratio)), mode='bilinear', align_corners=False)
    x = rearrange(x, '(b f) c h w -> b (f h w) c', f=T, h=int(H//resample_ratio), w=int(W//resample_ratio))
    return x


class SiTBlock(nn.Module):
    """
    A single Transformer block that optionally accepts a skip tensor.
    If skip=True, we learn a linear projection over the concatenation of the current features x and skip.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        skip=False,
        use_checkpoint=False,
        resample_ratio=1.0
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )
        self.norm2 = norm_layer(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
        )

        # For injecting time or label embeddings (AdaLayerNorm style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Skip connection logic
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, c, grid_info, skip=None, mask=None, kv_cache=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c, grid_info, skip, mask, kv_cache)
        else:
            return self._forward(x, c, grid_info, skip, mask, kv_cache)

    def _forward(self, x, c, grid_info, skip=None, mask=None, kv_cache=None):
        # If skip_linear exists, we do "concat + linear" just like the paper
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        # AdaLayerNorm modulations from c
        modulation_params = self.adaLN_modulation(c)#.chunk(6, dim=-1) #(B, T, D*6)

        T, H, W = grid_info

        B = x.shape[0]
        # print(modulation_params.shape)

        modulation_params = modulation_params[:, :, None, :]
        # print(modulation_params.shape)
        modulation_params = modulation_params.tile(1, 1, H * W, 1)
        # print(modulation_params.shape)
        modulation_params = modulation_params.view(B, T*H*W, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_params.chunk(6, dim=-1)

        # --- Attention path ---
        x_attn_normed = modulate(self.norm1(x), shift_msa, scale_msa)
        # print(x_attn_normed.shape,)
        x_attn, kv_cache = self.attn(x_attn_normed, grid_info, mask=mask, kv_cache=kv_cache)
        x = x + gate_msa * x_attn

        # --- MLP path ---
        x_mlp_normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_mlp_normed)
        x = x + gate_mlp * x_mlp

        return x, kv_cache
    

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c, grid_info):
        T, H, W = grid_info
        B = x.shape[0]
        modulation_params = self.adaLN_modulation(c)
        modulation_params = modulation_params[:, :, None, :]
        modulation_params = modulation_params.tile(1, 1, H * W, 1)
        modulation_params = modulation_params.view(B, T*H*W, -1)

        shift, scale = modulation_params.chunk(2, dim=-1)
        # print(shift.shape, scale.shape, x.shape)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_block_info(layer_idx, projection_ratios, direction='down'):
    cumsum = 0
    cumulative_ratios = 1
    for ratio, num_pairs in projection_ratios:
        if layer_idx < cumsum + num_pairs:
            # For down blocks: True if last layer in block
            # For up blocks: True if first layer in block
            is_resample_layer = (direction == 'down' and layer_idx == cumsum + num_pairs - 1) or \
                              (direction == 'up' and layer_idx == cumsum)
            cumulative_ratios *= ratio
            return ratio, is_resample_layer
        cumsum += num_pairs
    return 1, False

class SiT(nn.Module):
    """
    A UViT-like refactor of your SiT model:
      - Split 'depth' into in-blocks, a single mid-block, and out-blocks
      - Skip-connections from in-block outputs to out-blocks
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        in_depth = depth // 2
        out_depth = depth // 2

        self.x_embedder = nn.Linear(in_channels, hidden_size)
        # Timestep + label
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Linear(12, hidden_size)

        self.time_norm = nn.LayerNorm(hidden_size)
        self.action_norm = nn.LayerNorm(hidden_size)

        self.cond_combiner = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size*2, hidden_size)
        )

        # In-blocks (encoder)
        self.in_blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=False)
            for _ in range(in_depth)
        ])
        # Mid-block
        self.mid_block = SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=False)
        # Out-blocks (decoder), each with skip=True
        self.out_blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=True)
            for _ in range(out_depth)
        ])

        # Final prediction layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize
        self.initialize_weights()


        self.projection_ratios = [
            (2, 2), # first 3 pairs, full
            (4, 16), # next 5 pairs, 1/2
        ] # total of 14 layer pairs

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in all blocks
        for block in list(self.in_blocks) + [self.mid_block] + list(self.out_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, grid_info):
        c = self.out_channels
        T, H, W = grid_info
        assert T * H * W == x.shape[1]
        p = 1

        x = x.view(x.shape[0], T*H*W, c)
        return x

    def forward(self, x, t, act, grid_info, mask=None, kv_collection=None):
        x = self.x_embedder(x) #+ self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        action = self.y_embedder(act)#, self.training)    # (N, D)
        action = self.action_norm(action)
        t = self.time_norm(t)

        c = self.cond_combiner(torch.cat([t, action], dim=-1))   # (B, T, D)

        T, H, W = grid_info
        # ============ Encoder (in_blocks) ============
        skips = []

        if kv_collection is None:
            kv_collection = {}
        for i, blk in enumerate(self.in_blocks):
            # ratio, should_resample = get_block_info(i, self.projection_ratios, 'down')
            x, kv_out = blk(x, c, grid_info=grid_info, mask=mask, kv_cache=kv_collection[f"in/{i}"] if f"in/{i}" in kv_collection else None)  # no skip yet
            kv_collection[f"in/{i}"] = kv_out
            skips.append(x)

            # if should_resample:
            #     x = resample_tokens(x, grid_info, 2.0)
            #     grid_info = (T, H//ratio, W//ratio)
            #     mask = create_causal_temporal_mask(T, int((H//ratio)*(W//ratio)), 1)[0].to(x.device)
            
            

        # ============ Mid-block ============
        x, kv_out = self.mid_block(x, c, grid_info=grid_info, mask=mask, kv_cache=kv_collection[f"mid"] if f"mid" in kv_collection else None)
        kv_collection[f"mid"] = kv_out

        for i, blk in enumerate(self.out_blocks):
            skip_x = skips.pop()
            # ratio = int((skip_x.shape[1] / x.shape[1]) ** 0.5 ) # This gives us the needed upsample ratio
            # if ratio != 1:
            #     # print(f"Resampling up with ratio =", ratio, "grid_info =", grid_info)
            #     x = resample_tokens(x, grid_info, 1/ratio)
            #     currentH, currentW = grid_info[1:]
            #     grid_info = (T, int(currentH*ratio), int(currentW*ratio))
            #     mask = create_causal_temporal_mask(T, int((currentH*ratio)*(currentW*ratio)), 1)[0].to(x.device)
            x, kv_out = blk(x, c, grid_info, skip=skip_x, mask=mask, kv_cache=kv_collection[f"out/{i}"] if f"out/{i}" in kv_collection else None)
            kv_collection[f"out/{i}"] = kv_out

        # ============ Final Prediction + Unpatchify ============
        x = self.final_layer(x, c, grid_info=grid_info)
        x = self.unpatchify(x, grid_info)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x, kv_collection

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Classifier-free guidance pass. Similar to your original logic.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def lucid_model():
    return SiT(depth=28, hidden_size=2048, patch_size=1, num_heads=16)


def create_causal_temporal_mask(
    context_length: int,
    num_patches: int,
    num_heads: int,
) :
    T = context_length
    N = num_patches
    total_tokens = T * N
    frame_indices = torch.repeat_interleave(torch.arange(T), N)  # Shape: (T*N,)
    mask_matrix = frame_indices[None, :] <= frame_indices[:, None]  # Shape: (T*N, T*N)
    float_mask = torch.where(mask_matrix, True, False)#.astype(jnp.float32)  # Shape: (T*N, T*N)
    float_mask = float_mask[None, None, :, :]
    float_mask = torch.broadcast_to(float_mask, (1, num_heads, T*N, T*N))
    return float_mask



# def kv_sample(model, output)
@torch.no_grad()
def sample_ar(model, frames, act, grid_info, n_steps):
    """Autoregressive frame sampling with KV caching"""
    ds = 1.0 / n_steps
    kv_collection = None
    device = frames.device
    output = frames

    for step in range(n_steps, 0, -1):
        # t_step = torch.tensor([step / n_steps], device=device)
        
        # t_step[:, :-1] += 0.3

        if kv_collection is None:
            t_step = torch.zeros((B, grid_info[0]), device=device, dtype=frames.dtype)
            t_step[:, :-1] = 0.2
            t_step[:, -1] = (step/n_steps)
            pred, kv_collection = model(output, t_step, act, grid_info=grid_info)
            pred = rearrange(pred, "b (t h w) c -> b t h w c", h=grid_info[1], w=grid_info[2])
        else:
            t_step = torch.zeros((B, 1), device=device, dtype=frames.dtype) + (step/n_steps)
            pred, _ = model(output[:, -1:], t_step, act[:, -1:], grid_info=(1, grid_info[1], grid_info[2]), kv_collection=kv_collection)
            pred = rearrange(pred, "b (t h w) c -> b t h w c", h=grid_info[1], w=grid_info[2])
        _output = rearrange(output, "b (t h w) c -> b t h w c", h=grid_info[1], w=grid_info[2])
        _output[:, -1] = _output[:, -1] - (pred[:, -1] * ds)
        output = rearrange(_output, "b t h w c -> b (t h w) c")

    return output
# from torch.profiler import profile, record_function, ProfilerActivity
if __name__ == "__main__":
    model = lucid_model()
    # number of params
    number_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {number_params}")
    model = model.to(torch.bfloat16).cuda()
    model.eval()
    B = 2
    T, H, W = 32, 8, 8
    frames = torch.rand(B, T* H* W, 128).cuda().bfloat16()
    y_cond = torch.rand(B, T, 12).cuda().bfloat16()
    number_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {number_params}")

    mask = create_causal_temporal_mask(T, H*W, 1)[0].cuda().bfloat16()

    # sample_ar = torch.compile(sample_ar)c
    # sample_ar = torch.compile(sample_ar)

    print("looping model purely")
    for i in tqdm.trange(10):
        with torch.no_grad():
            for i in range(10):
                out = model(frames, torch.zeros((B, T), device=frames.device, dtype=frames.dtype), y_cond, (T, H, W), mask=mask)[0]

    model = torch.compile(model)

    print("looping model purely w/ compile")
    for i in tqdm.trange(10):
        with torch.no_grad():
            for i in range(10):
                out = model(frames, torch.zeros((B, T), device=frames.device, dtype=frames.dtype), y_cond, (T, H, W), mask=mask)[0]
    print("sampling w/compile")
    for i in range(5):
        with torch.no_grad():
            # out = model(frames, torch.zeros((B, T), device=frames.device, dtype=frames.dtype), y_cond, (T, H, W), mask=mask)[0]
            out = sample_ar(model, frames, y_cond, (T, H, W), 20) 

    for i in tqdm.trange(10000):
        with torch.no_grad():
            # out = model(frames, torch.zeros((B, T), device=frames.device, dtype=frames.dtype), y_cond, (T, H, W), mask=mask)[0]
            out = sample_ar(model, frames, y_cond, (T, H, W), 20)

    print(out.shape, out.min(), out.max())
