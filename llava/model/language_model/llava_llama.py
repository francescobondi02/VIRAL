#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# IMAGENET_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
# IMAGENET_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]

# CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
# CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def visualize_feature_pca(features, save_path='feature.png'):
    """
    This method performs PCA on the features, reduces them to 3 dimensions.
    """
    # CPU로 이동하고 numpy로 변환
    features_np = features.float().squeeze(0).detach().cpu().numpy()
    
    # PCA로 768차원을 3차원으로 축소
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(features_np)
    
    # 정규화: 중앙값과 IQR 기반
    median = np.median(feat_pca, axis=0)
    q1 = np.percentile(feat_pca, 25, axis=0)
    q3 = np.percentile(feat_pca, 75, axis=0)
    iqr = q3 - q1
    scaled = (feat_pca - median) / (iqr + 1e-6)
    feat_pca_norm = 0.5 * (np.tanh(scaled) + 1)
    
    # 24x24x3 이미지로 재구성
    size = int(np.sqrt(features_np.shape[0]))  # target_size//16
    rgb_image = feat_pca_norm.reshape(size, size, 3)
    
    # 시각화 및 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_image)
    plt.title(f'Feature Map Visualization ({size}x{size})')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rgb_image


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def is_main_process():
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        return is_main_process
    return True

class AlignmentProjector(nn.Module):
    """
    Class which handles the projection, to map hidden states of the LLM (hidden_size) into feature space of the VFM (z_dim)
    """
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        """self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)"""
        
        self.projector = nn.Linear(hidden_size, z_dim)
        nn.init.xavier_uniform_(self.projector.weight, gain=3.0)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        return self.projector(x)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache

class ResidualLlamaModel(LlamaModel):
    """
    Llama Model with residual connection from input embeddings to specified layers.
    Used to add residual connection at layer 16 for VIRAL.
    """
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,                         # * Inputs, shape = (B, L)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,          # * Alternative inputs, shape = (B, L)
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,                   # * Should return the attention maps or not?
        output_hidden_states: Optional[bool] = None,                # * Should output the hidden states or not?
        return_dict: Optional[bool] = None,
        residual: Optional[bool] = False,                           # * If TRUE, applies the VIRAL method of residual connection
        target_layers: Optional[List[int]] = None,                  # * Specifies the target layers to which apply the residual connection
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if residual:
            assert target_layers is not None, "target_layers must be specified if residual is True"

        # * If the user didn't specify these values, use the default from the config.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # * Retrieve input_ids or inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # * Handles cache
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        # * If user didn't pass position_ids, compute them (using the cached values)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # * If the user didn't pass embeddings but only ids, then we convert them into embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # * Prepare the attention mask, depending on the implementation used
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # * Embed positions (shape = (B, L, hidden_size)), now we have the hidden state before
        # * adding any residual connection
        hidden_states = inputs_embeds

        # * Save the hidden states / attentions if requested by the user
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # * Loop over the layers of the decoder
        for idx, decoder_layer in enumerate(self.layers):
            # * Save the hidden state in the data structure
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # * Get the output of the layer (either with checkpoint or by direct call)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # ^ REMINDER layer_outputs = [hidden_states, attn_weights, past_key_value (optional)]

            # * Update hidden states with the output of the current layer and by adding the residual connection (if needed)
            hidden_states = layer_outputs[0]
            if residual and (idx+1 in target_layers):
                hidden_states = hidden_states + inputs_embeds

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            # * Save the attention score if the user requested it
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # * Normalize the hidden states
        hidden_states = self.norm(hidden_states)

        # * Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        # * Format and returns the output (depends on return_dict)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaLlamaModel(LlavaMetaModel, ResidualLlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        # * Call the LLM Backbone
        self.model = LlavaLlamaModel(config)

        # * Base parameters for model
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # * Specific features for VIRAL and residuals (cannot have both, they have the same scope)
        self.vra_loss = config.vra_loss if hasattr(config, 'vra_loss') else False
        self.residual = config.residual if hasattr(config, 'residual') else False
        self.diffusion_loss = config.diffusion_loss if hasattr(config, 'diffusion_loss') else False
        if self.diffusion_loss:
            self.diffusion_weight = config.diffusion_weight if hasattr(config, 'diffusion_weight') else 1.0
        assert not(self.vra_loss & self.residual), "vra loss and residual cannot be true at same time"
        self.residual_target_layers = config.residual_target_layers if hasattr(config, 'residual_target_layers') else None

        # * Include Contrastive Loss settings
        self.contrastive_loss = config.contrastive_loss if hasattr(config, 'contrastive_loss') else False
        self.contrastive_weight = config.contrastive_weight if hasattr(config, 'contrastive_weight') else 0.2
        self.contrastive_temperature = config.contrastive_temperature if hasattr(config, 'contrastive_temperature') else 0.8
        self.contrastive_n_samples = config.contrastive_n_samples if hasattr(config, 'contrastive_n_samples') else 150
        self.contrastive_sum_in_log = config.contrastive_sum_in_log if hasattr(config, 'contrastive_sum_in_log') else False

        # * Logic for VIRAL loss
        if self.vra_loss:
            self.only_coco = config.only_coco if hasattr(config, 'only_coco') else False
            self.target_layers = config.target_layers if hasattr(config, 'target_layers') else [15,16]

            # * Which VFM to use for alignment
            self.vra_target = config.vra_target if hasattr(config, 'vra_target') else "dinov2-vit-b" # sam_vit_b_01ec64, dinov2-vit-b, clip
            self.vra_weight = config.vra_weight if hasattr(config, 'vra_weight') else 0.5

            # * Size of the projector and z_dim
            self.projector_dim = config.projector_dim if hasattr(config, 'projector_dim') else 2048 # VRA Default
            self.z_dim = config.z_dim if hasattr(config, 'z_dim') else 768 # DINO Default 768, 256 if SAM, 1024 if CLIP-L

            # * Types of loss
            self.alignment_loss = config.alignment_loss if hasattr(config, 'alignment_loss') else "direct" #"direct" # direct, similarity
            self.use_projector = config.use_projector if hasattr(config, 'use_projector') else False # If False, use the mid hidden states directly, only for similarity loss.
            self.use_multiple_projectors = config.use_multiple_projectors if hasattr(config, 'use_multiple_projectors') else False # If True, use multiple projectors for each layer, only for similarity loss.
            if self.target_layers is not None:
                # * Create AlignmentProject(s) to go from hidden_size (LLM) to z_dim (VFM)
                if self.use_multiple_projectors:
                    self.alignment_projector = nn.ModuleList([
                        AlignmentProjector(config.hidden_size, self.projector_dim, self.z_dim) for _ in range(len(self.target_layers))
                    ])
                else:
                    self.alignment_projector = AlignmentProjector(config.hidden_size, self.projector_dim, self.z_dim)

                # * Handles the loading of the VFM (handles the different cases)
                if 'dinov2' in self.vra_target:
                    import timm
                    model, _, model_config = self.vra_target.split('-')
                    if 'reg' in self.vra_target:
                        self.alignment_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
                    else:
                        self.alignment_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
                    del self.alignment_encoder.head
                    patch_resolution = 24
                    self.alignment_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                        self.alignment_encoder.pos_embed.data, [patch_resolution, patch_resolution],
                    )
                    self.alignment_encoder.head = nn.Identity()
                elif 'sam' in self.vra_target:
                    from segment_anything import sam_model_registry, SamPredictor
                    if os.path.exists(self.vra_target):
                        sam_checkpoint = self.vra_target
                    else:
                        sam_checkpoint = f"./playground/vfm_weights/{self.vra_target}.pth"
                    model_type = "vit_b" if "vit_b" in sam_checkpoint else "vit_l" if "vit_l" in sam_checkpoint else "vit_h"
                    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    predictor = SamPredictor(sam)
                    self.alignment_encoder = predictor.model.image_encoder
                    del sam, predictor
                    torch.cuda.empty_cache()
                    patch_resolution = 24
                elif 'clip' in self.vra_target:
                    self.alignment_encoder = None
                    # we will use CLIPVisionTower later on
                elif 'radio' in self.vra_target.lower():
                    self.alignment_encoder = torch.hub.load('NVlabs/RADIO', 'radio_model', version=self.vra_target, progress=True, skip_validation=True)
                elif 'depth_anything' in self.vra_target.lower():
                    from ..depth_anything_v2.dpt import DepthAnythingV2
                    
                    model_configs = {
                        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                    }
                    
                    if "vitb" in self.vra_target.lower():
                        encoder = "vitb"
                    else:
                        raise NotImplementedError(f"Unknown encoder type for Depth-Anything: {self.vra_target}")
                    
                    dv2 = DepthAnythingV2(**model_configs[encoder])
                    dv2.load_state_dict(torch.load(f'./playground/vfm_weights/{self.vra_target}.pth', map_location='cpu'))
                    self.alignment_encoder = dv2.pretrained
                    
                    del dv2
                    torch.cuda.empty_cache()

        print("=" * 80)
        print("[LLAVA INIT] Model Configuration Summary")
        print("=" * 80)
        print(f"[VRA LOSS]")
        print(f"  Enabled: {self.vra_loss}")
        if self.vra_loss:
            print(f"  Target Layers: {self.target_layers if hasattr(self, 'target_layers') else 'NOT SET YET'}")
            print(f"  VRA Target: {self.vra_target if hasattr(self, 'vra_target') else 'NOT SET YET'}")
            print(f"  VRA Weight: {self.vra_weight if hasattr(self, 'vra_weight') else 'NOT SET YET'}")
            print(f"  Z Dim: {self.z_dim if hasattr(self, 'z_dim') else 'NOT SET YET'}")
            print(f"  Projector Dim: {self.projector_dim if hasattr(self, 'projector_dim') else 'NOT SET YET'}")
            print(f"  Alignment Loss Type: {self.alignment_loss if hasattr(self, 'alignment_loss') else 'NOT SET YET'}")
        print(f"\n[CONTRASTIVE LOSS]")
        print(f"  Enabled: {self.contrastive_loss}")
        if self.contrastive_loss:
            print(f"  Weight: {self.contrastive_weight}")
            print(f"  Temperature: {self.contrastive_temperature}")
            print(f"  N Samples: {self.contrastive_n_samples}")
            print(f"  Sum in Log: {self.contrastive_sum_in_log}")
        print(f"\n[RESIDUAL]")
        print(f"  Enabled: {self.residual}")
        if self.residual:
            print(f"  Target Layers: {self.residual_target_layers}")
        print(f"\n[DIFFUSION]")
        print(f"  Enabled: {self.diffusion_loss}")
        if self.diffusion_loss:
            print(f"  Weight: {self.diffusion_weight}")
        print("=" * 80)
                    
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward_dep(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None, # Need input ids for vra loss
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        is_coco = None,
        patch_labels: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # * Prepare the input, insert in the sequence also token-image and generate inputs_embeds = [img_patch_1, …, img_patch_576, text...]
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        # * Create a mask to identify image tokens (assumed to have id -200)
        img_token_where = (input_ids == -200)
        if self.training and inputs_embeds is not None:
            input_ids = None
        
        # * Get the configs
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.vra_loss:
            output_hidden_states = True
        print(f"[INFO][contrastive_loss={self.contrastive_loss}, training={self.training}]")
        if self.contrastive_loss and self.training:
            output_attentions = True
            
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # * Call the LLM Backbone (ResidualLlamaModel), extracts hidden states per layer and returns them (together with attentions, if requested)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            residual=self.residual,
            target_layers=self.residual_target_layers
        )

        hidden_states = outputs[0]

        # * Start working for the VRA_LOSS
        if self.vra_loss:
            if self.target_layers is None:
                # * Take all as targets
                self.target_layers = list(range(0, len(self.model.layers)+1))

            # * Obtain hidden states for target layers (and put them in a list)
            mid_hidden_states = [outputs.hidden_states[i] for i in self.target_layers]
            
            del outputs.hidden_states
            
            # * Extract visual features from the VFM (depending on which one is used)
            if 'dinov2' in self.vra_target:
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder.forward_features(images_resized)['x_norm_patchtokens']
                del images_resized
            elif 'clip' in self.vra_target:
                self.alignment_encoder = self.model.vision_tower # Assume grad in disabled
                assert self.alignment_encoder is not None, "CLIP vision tower is not loaded."
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder(images_resized)
                del images_resized
            elif 'sam' in self.vra_target:
                images_resized = F.interpolate(images, size=(384, 384), mode="bilinear")
                padded_size = 1024
                feature_size = 24
                normalized_mean = torch.tensor([0, 0, 0], dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                padded_images = torch.ones((images_resized.shape[0], 3, padded_size, padded_size), dtype=images_resized.dtype, device=images_resized.device) * normalized_mean
                start_h = (padded_size - 384) // 2
                start_w = (padded_size - 384) // 2
                padded_images[:, :, start_h:start_h+384, start_w:start_w+384] = images_resized
            
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder(padded_images)
                del images_resized, padded_images
                B, C, H, W = alignment_feature.shape
                start_idx = (H - feature_size) // 2
                end_idx = start_idx + feature_size
                alignment_feature = alignment_feature[:, :, start_idx:end_idx, start_idx:end_idx] # [B, C, 24, 24]
                alignment_feature = alignment_feature.permute(0, 2, 3, 1).reshape(B, -1, C) # [B, 576, C]
            elif 'radio' in self.vra_target.lower():
                images_resized = F.interpolate(images, size=(384, 384), mode="bilinear")
                
                mean = torch.tensor(self.model.vision_tower.image_mean, dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                std = torch.tensor(self.model.vision_tower.image_std, dtype=images_resized.dtype, device=images_resized.device).view(1, 3, 1, 1)
                images_resized = torch.clamp(images_resized * std + mean, 0, 1)
                
                self.alignment_encoder.eval()
                with torch.no_grad():
                    if images_resized.dtype == torch.bfloat16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            summary, alignment_feature = self.alignment_encoder(images_resized)
                    else:
                        summary, alignment_feature = self.alignment_encoder(images_resized)
                del images_resized, summary
            elif 'depth_anything' in self.vra_target:
                images_resized = F.interpolate(images, size=(336, 336), mode="bilinear") # Maybe it is better to use 224 and upsample feature or downsample llm feature
                if 'vitb' in self.vra_target.lower():
                    target_layer = 11
                else:
                    raise NotImplementedError(f"Unknown Depth-Anything model: {self.vra_target}")
                
                self.alignment_encoder.eval()
                with torch.no_grad():
                    alignment_feature = self.alignment_encoder.get_intermediate_layers(images_resized, [target_layer], return_class_token=False)[0]
                del images_resized
            
            # ^ alignment_feature[b] = (576, z_dim)
            # * Converts the min_hidden_states into shape (B, num_layers, seq_len, hidden_size)
            mid_hidden_states = torch.stack(mid_hidden_states, dim=1)
            bsz, num_layers, seq_len, hidden_size = mid_hidden_states.shape # seq_len should be change to image patches

            print(f"\n[DEBUG BEFORE PROJECTOR]")
            print(f"  mid_hidden_states[0, 0, :10]: {mid_hidden_states[0, 0, :10]}")  # Prime 10 dim
            print(f"  mean={mid_hidden_states.mean():.6f}, std={mid_hidden_states.std():.6f}")
            print(f"  min={mid_hidden_states.min():.6f}, max={mid_hidden_states.max():.6f}")
            print(f"  L2 norm per token: {torch.norm(mid_hidden_states[0, 0], p=2, dim=-1).mean():.6f}\n")

            # * Projection of the mid hidden states into the space of VFM (hidden_size -> z_dim)
            if self.alignment_loss == "direct" or (self.alignment_loss == "similarity" and self.use_projector):
                if self.use_multiple_projectors:
                    projected_feature = []
                    for idx in range(len(self.target_layers)):
                        proj_b = self.alignment_projector[idx](mid_hidden_states[:, idx, :])
                        projected_feature.append(proj_b)
                    projected_feature = torch.stack(projected_feature, dim=1)
                else:
                    mid_hidden_states = mid_hidden_states.view(-1, seq_len, hidden_size) # flatten the layers
                    projected_feature = self.alignment_projector(mid_hidden_states)
                    projected_feature = projected_feature.view(bsz, num_layers, seq_len, -1) # reshape to bsz, num_layers, seq_len, z_dim
                    # * projected_feature[b, layer_idx, img_tokens, :] -> describes the image tokens projected into the space VFM (I THINK THIS COULD BE VERY USEFUL)
            elif self.alignment_loss == "similarity":
                projected_feature = mid_hidden_states
            
            del mid_hidden_states
            # torch.cuda.empty_cache() chatgpt told me so
            
            vra_loss = 0.
            valid_batch_count = 0
            contrastive_loss = 0.0
            # * Start computing the VRA Loss (PROBABLY NEED TO ADD HERE THE CONTRASTIVE PART)
            for b in range(bsz):
                img_tokens_b = img_token_where[b] # * Get the image tokens for the b-th sample
                if img_tokens_b.any() and img_tokens_b.sum() == 576:
                    if self.only_coco and not is_coco[b]:
                        continue
                    valid_batch_count += 1
                    # * For each target layer
                    for idx in range(len(self.target_layers)):
                        if self.alignment_loss == "direct":
                            proj_b = projected_feature[b, idx, img_tokens_b, :] # * Projected feature for the image tokens (in the VFM space)

                            print(f"[DEBUG PRE-NORM b={b}] shape={proj_b.shape}")
                            print(f"[DEBUG PRE-NORM b={b}] mean={proj_b.mean():.6f}, std={proj_b.std():.6f}")
                            print(f"[DEBUG PRE-NORM b={b}] min={proj_b.min():.6f}, max={proj_b.max():.6f}")
                            print(f"[DEBUG PRE-NORM b={b}] L2 norms: mean={torch.norm(proj_b, p=2, dim=-1).mean():.6f}")

                            proj_b_vra = F.normalize(proj_b, p=2, dim=-1)
                            alig_b = F.normalize(alignment_feature[b], dim=-1) # * normalize the alignment feature (this comes from DINO)
                            print(f"[DEBUG POST-NORM b={b}, now it's proj_b_vra] std={proj_b_vra.std():.6f}")  # ← Questo è il tuo 0.029!
                            print(f"[DEBUG POST-NORM b={b}, now it's proj_b_vra] L2 norms: {torch.norm(proj_b_vra, p=2, dim=-1).mean():.6f} (should be 1.0)")

                            # * Add the VRA loss
                            vra_loss += (-(proj_b_vra * alig_b).sum(dim=-1)).mean()

                            # TODO PROBABLY NEED TO ADD HERE THE COMPUTATION FOR THE CONTRASTIVE LOSS
                            if self.contrastive_loss and patch_labels is not None:
                                # print(f"[DEBUG CALL] b={b}, bsz={bsz}")
                                # print(f"[DEBUG CALL] patch_labels.shape = {patch_labels.shape}")
                                # print(f"[DEBUG CALL] patch_labels[{b}].shape = {patch_labels[b].shape}")
                                # print(f"[DEBUG CALL] proj_b.shape = {proj_b.shape}")

                                # Verifica dimensioni
                                assert patch_labels[b].dim() == 1, \
                                    f"Expected 1D single-label tensor, got shape {patch_labels[b].shape}"
                                
                                contrastive_b = self.compute_contrastive_loss(
                                    patch_features=proj_b,         # (576, D)
                                    patch_labels=patch_labels[b],  # (576,) ← single label per patch
                                    temperature=self.contrastive_temperature,
                                    n_samples=self.contrastive_n_samples,
                                    verbose=True
                                )
                                contrastive_loss += contrastive_b
                        elif self.alignment_loss == "similarity":
                            proj_b = projected_feature[b, idx, img_tokens_b, :]
                            alig_b = alignment_feature[b]
                            proj_b = F.normalize(proj_b, dim=-1)
                            alig_b = F.normalize(alig_b, dim=-1)
                            
                            proj_b = torch.matmul(proj_b, proj_b.transpose(-2, -1))
                            alig_b = torch.matmul(alig_b, alig_b.transpose(-2, -1))
                            sim_loss = F.mse_loss(proj_b, alig_b)
                            vra_loss += sim_loss
                        else:
                            raise ValueError(f"Unknown alignment loss: {self.alignment_loss}")
            if valid_batch_count > 0:
                vra_loss /= (valid_batch_count * len(self.target_layers))
                if self.contrastive_loss:
                    contrastive_loss /= (valid_batch_count * len(self.target_layers))
                    if not isinstance(contrastive_loss, torch.Tensor):
                        contrastive_loss = torch.tensor(contrastive_loss, device=hidden_states.device)
                    # * Plot contrastive loss
                    if is_main_process() and self.training:
                        wandb.log({"contrastive loss": contrastive_loss.item()})
            
        # * Compute logits
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # * Compute the usual language modeling loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        # * Fusion of the losses
        # ^ loss qui è solo la loss base di language modeling (Next Token Prediction)
        # ! QUI SAREBBE DA FARE L'AGGIUNTA DELLA CONTRASTIVE
        if self.vra_loss:
            print(f"VRA loss: {vra_loss} || NTP loss: {loss}")
            if is_main_process() and self.training:
                wandb.log({"ntp loss": loss.item()})
                if vra_loss == 0.0:
                    wandb.log({"vra loss": 0.0})
                else:
                    wandb.log({"vra loss": vra_loss.item()})
            loss = loss + self.vra_weight * vra_loss
        elif self.residual:
            if self.training:
                print(f"VRA loss: {0.0} || NTP loss: {loss}")
                if is_main_process():
                    wandb.log({"ntp loss": loss.item()})
        
        # ^ Contrastive Loss (added)
        if self.contrastive_loss:
            loss = loss + self.contrastive_weight * contrastive_loss
        
        # * Diffusion loss (not relevant for VIRAL paper)
        diffusion_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.diffusion_loss:
            bsz = hidden_states.shape[0]
            if img_token_where.sum() == bsz * 576:
                diffusion_loss = self.compute_vm_loss(images, hidden_states, img_token_where)
                
            print(f"Diffusion loss: {diffusion_loss}")
            if is_main_process():
                wandb.log({"diffusion loss": diffusion_loss.item()})
                # wandb.log({"ntp loss": loss.item()})
            # * Add the diffusion loss (now loss = NTP + VRA + Diffusion)
            loss = loss + self.diffusion_weight * diffusion_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Inference time, generates the output given text + images
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    # ^ ADDED FOR CONTRASTIVE LOSS IMPLEMENTATION
    def compute_contrastive_loss(
        self,
        patch_features: torch.Tensor,
        patch_labels: torch.Tensor,
        temperature: float = 0.07,
        n_samples: int = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss con formula corretta.
        """
        device = patch_features.device
        N, D = patch_features.shape
        
        # ═══════════════════════════════════════════════════════════
        # VALIDAZIONE
        # ═══════════════════════════════════════════════════════════
        assert patch_features.dim() == 2
        assert patch_labels.dim() == 1
        assert patch_labels.shape[0] == N
        assert temperature > 0
        
        # ═══════════════════════════════════════════════════════════
        # 1. Filtra patch valide (label > 0, escludi background)
        # ═══════════════════════════════════════════════════════════
        valid_mask = patch_labels > 0
        num_valid = valid_mask.sum().item()
        
        if num_valid < 2:
            if verbose:
                print(f"[CONTRASTIVE] ⚠️ Less than 2 valid patches → returning 0.0")
            return torch.tensor(0.0, device=device, dtype=patch_features.dtype)
        
        # Check: almeno 2 oggetti diversi (altrimenti no negatives)
        unique_labels = torch.unique(patch_labels[valid_mask])
        if unique_labels.numel() == 1:
            if verbose:
                print(f"[CONTRASTIVE] ⚠️ Only 1 unique object → no negatives → returning 0.0")
            return torch.tensor(0.0, device=device, dtype=patch_features.dtype)
        
        if verbose:
            print(f"\n[CONTRASTIVE] ═══ Input Info ═══")
            print(f"[CONTRASTIVE] Valid patches: {num_valid}/{N} ({100*num_valid/N:.1f}%)")
            print(f"[CONTRASTIVE] Unique objects: {unique_labels.tolist()}")
        
        # ═══════════════════════════════════════════════════════════
        # 2. Normalizza features e calcola similarity
        # ═══════════════════════════════════════════════════════════
        print(f"[CONTRASTIVE] Feature std before normalization : {patch_features.std().item():.4f}")
        patch_features_norm = F.normalize(patch_features, p=2, dim=-1)
        print(f"[CONTRASTIVE] Feature std after normalization : {patch_features_norm.std().item():.4f}")
        sim_matrix = torch.matmul(patch_features_norm, patch_features_norm.T) / temperature
        # Shape: (N, N), già diviso per temperature!
        
        # ═══════════════════════════════════════════════════════════
        # 3. Costruisci Positive Mask
        # ═══════════════════════════════════════════════════════════
        labels_eq = patch_labels.unsqueeze(1) == patch_labels.unsqueeze(0)  # (N, N)
        valid_pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(0)
        pos_mask = labels_eq & valid_pair_mask  # (N, N) bool
        
        # Escludi diagonale (self-similarity)
        eye_mask = torch.eye(N, device=device, dtype=torch.bool)
        pos_mask = pos_mask & ~eye_mask
        
        # ═══════════════════════════════════════════════════════════
        # 4. Seleziona anchors con almeno 1 positive
        # ═══════════════════════════════════════════════════════════
        has_positives = pos_mask.sum(dim=1) > 0
        anchor_indices = has_positives.nonzero(as_tuple=False).squeeze(-1)
        
        if anchor_indices.numel() == 0:
            if verbose:
                print(f"[CONTRASTIVE] ⚠️ No valid anchors → returning 0.0")
            return torch.tensor(0.0, device=device, dtype=patch_features.dtype)
        
        # Subsampling
        if n_samples is not None and anchor_indices.numel() > n_samples:
            perm = torch.randperm(anchor_indices.numel(), device=device)
            anchor_indices = anchor_indices[perm[:n_samples]]
        
        A = anchor_indices.numel()
        
        if verbose:
            print(f"[CONTRASTIVE] Using {A} anchors")
        
        # ═══════════════════════════════════════════════════════════
        # 5. Calcola InfoNCE Loss (FORMULA CORRETTA)
        # ═══════════════════════════════════════════════════════════
        sim_anchor = sim_matrix[anchor_indices]  # (A, N)
        pos_anchor = pos_mask[anchor_indices]    # (A, N) bool
        
        # Maschera per denominatore: tutti tranne anchor stesso
        denom_mask = ~eye_mask[anchor_indices]  # (A, N) bool
        
        # Numeratore: Σ_{j∈P(i)} exp(sim_ij / τ)
        # Nota: sim_matrix è già diviso per temperature!
        exp_sim = torch.exp(sim_anchor)  # (A, N)
        numerator = (exp_sim * pos_anchor.float()).sum(dim=1)  # (A,)
        
        # Denominatore: Σ_{k≠i} exp(sim_ik / τ)
        denominator = (exp_sim * denom_mask.float()).sum(dim=1)  # (A,)
        
        # Loss: -log(numerator / denominator)
        # Equivalente: log(denominator) - log(numerator)
        loss_per_anchor = torch.log(denominator) - torch.log(numerator)  # (A,)
        
        # Safety checks
        if torch.any(numerator <= 0) or torch.any(denominator <= 0):
            if verbose:
                print(f"[CONTRASTIVE] ❌ ERROR: Zero in log! num={numerator.min():.2e}, denom={denominator.min():.2e}")
            return torch.tensor(0.0, device=device, dtype=patch_features.dtype)
        
        loss = loss_per_anchor.mean()
        
        # ═══════════════════════════════════════════════════════════
        # 6. Debug info
        # ═══════════════════════════════════════════════════════════
        if verbose:
            pos_sim = sim_anchor[pos_anchor] * temperature  # Riporta a scala originale
            neg_mask = (~pos_anchor) & denom_mask
            neg_sim = sim_anchor[neg_mask] * temperature
            
            pos_neg_gap = pos_sim.mean() - neg_sim.mean()
            
            print(f"[CONTRASTIVE] Similarity stats:")
            print(f"[CONTRASTIVE]   Positive: mean={pos_sim.mean().item():.3f}, "
                f"range=[{pos_sim.min().item():.3f}, {pos_sim.max().item():.3f}]")
            print(f"[CONTRASTIVE]   Negative: mean={neg_sim.mean().item():.3f}, "
                f"range=[{neg_sim.min().item():.3f}, {neg_sim.max().item():.3f}]")
            print(f"[CONTRASTIVE]   Gap: {pos_neg_gap.item():.3f} (target >0.15)")
            
            # Feature variance check
            feature_std = patch_features.std(dim=0).mean()
            print(f"[CONTRASTIVE] Feature std: {feature_std.item():.4f} (target 0.3-0.5)")
            
            print(f"[CONTRASTIVE] ─── Loss ───")
            print(f"[CONTRASTIVE] Value: {loss.item():.4f}")
            print(f"[CONTRASTIVE] Per anchor: mean={loss_per_anchor.mean().item():.3f}, "
                f"range=[{loss_per_anchor.min().item():.3f}, {loss_per_anchor.max().item():.3f}]")
            
            if pos_neg_gap < 0.1:
                print(f"[CONTRASTIVE] ⚠️ WARNING: Feature collapse risk!")
            if feature_std < 0.1:
                print(f"[CONTRASTIVE] ⚠️ WARNING: Feature variance too low!")
            
            print(f"[CONTRASTIVE] ═══════════════════════════\n")
        
        # Validazione finale
        assert not torch.isnan(loss), "Loss is NaN!"
        assert not torch.isinf(loss), "Loss is Inf!"
        assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
        
        return loss

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
