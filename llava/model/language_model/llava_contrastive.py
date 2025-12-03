"""
Contrastive loss manager per LLaVA basato su EgoLifter
Obiettivo: Migliorare la coerenza delle vision features usando semantic segmentation
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image


class LLaVAContrastiveManager:
    """
    Gestisce la contrastive loss per allineamento vision-semantic in LLaVA
    
    Basato su EgoLifter ma adattato per:
    - Features CLIP invece di NeRF radiance fields
    - Maschere di segmentazione COCO
    - Training/Inference mode
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        n_samples: int = 256,
        sum_in_log: bool = True,
        sim_exp: float = 1.0,
        min_pixels_per_object: int = 5,
        use_multilabel: bool = False,
    ):
        """
        Args:
            temperature: Temperatura per softmax (0.07-0.2 tipici)
            n_samples: Quanti pixel campionare per batch (per efficienza)
            sum_in_log: True = sum-in-log (più robusta), False = sum-out-log (più precisa)
            sim_exp: Esponente per similarity (per multi-label IoU)
            min_pixels_per_object: Minimo pixel per oggetto valido
            use_multilabel: Supporta oggetti sovrapposti
        """
        self.temperature = temperature
        self.n_samples = n_samples
        self.sum_in_log = sum_in_log
        self.sim_exp = sim_exp
        self.min_pixels_per_object = min_pixels_per_object
        self.use_multilabel = use_multilabel
    
    def prepare_features_and_labels(
        self,
        features: torch.Tensor,
        masks: List[np.ndarray],  # ← Accetta numpy arrays
        feature_hw: Tuple[int, int],
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, W = feature_hw
        
        if features.dim() == 3:
            features = features.view(-1, features.size(-1))
        
        # CRITICO: Usa il device delle features
        device = features.device
        label_map = torch.ones(H, W, dtype=torch.long, device=device) * -1
        
        for obj_idx, mask in enumerate(masks):
            down_mask = self._downsample_mask(mask, (H, W))
            
            # Sposta mask su CUDA
            down_mask = down_mask.to(device)  # Sposta su CUDA
            
            label_map[down_mask] = obj_idx
        
        labels_flat = label_map.view(-1)
        attention_flat = None
        if attention_weights is not None:
            # Verifica che abbiano la shape corretta
            if attention_weights.shape != (H, W):
                raise ValueError(
                    f"attention_weights shape {attention_weights.shape} "
                    f"doesn't match feature_hw {(H, W)}"
                )
            # Sposta su stesso device delle features
            attention_weights = attention_weights.to(device)
            attention_flat = attention_weights.view(-1)  # (H*W,)

        # AGGIUNGI QUESTO DEBUG:
        num_foreground = (label_map >= 0).sum().item()
        num_background = (label_map == -1).sum().item()
        print(f"  Label map: {num_foreground} foreground, {num_background} background")
        return features, labels_flat, attention_flat
    
    def _downsample_mask(self, mask: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Downsample mask usando nearest neighbor"""
        h, w = target_hw
        # Usa torch direttamente (più veloce)
        mask_torch = torch.from_numpy(mask.astype(np.float32))
        mask_resized = F.interpolate(
            mask_torch.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
            size=(h, w),
            mode='nearest'
        ).squeeze(0).squeeze(0)  # (H, W)
        return mask_resized > 0.5
    
    def compute_loss(
        self,
        features: torch.Tensor,
        instance_labels: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola contrastive loss (implementazione EgoLifter-style)
        
        Args:
            features: (N, D) features dei pixel campionati
            instance_labels: (N,) label degli oggetti
            sample_weights: (N,) optional peso per pixel
            
        Returns:
            loss: scalar tensor
        """
        N = features.size(0)
        
        # Early exit se troppo pochi sample
        if N < 8:
            print(f"  ⚠️ Too few samples ({N} < 8), returning 0.0")
            return torch.tensor(0.0, device=features.device)
        
        # 1. Costruisci similarity mask (quali pixel appartengono allo stesso oggetto)
        sim_masks = self._build_similarity_mask(instance_labels)  # (N, N)
        
        # 2. Calcola distance matrix
        distance_sq = self._compute_distance_matrix(features)  # (N, N)
        
        # 3. Temperature differenziata (alta per positive, bassa per negative)
        temperature_matrix = self._build_temperature_matrix(sim_masks)  # (N, N)
        
        # 4. Similarity kernel (Gaussian RBF)
        similarity_kernel = torch.exp(-distance_sq / temperature_matrix)  # (N, N)
        
        # 5. Apply attention-based weights
        if attention_weights is not None:
            # Verifica shape corretta
            if attention_weights.size(0) != N:
                raise ValueError(
                    f"attention_weights size {attention_weights.size(0)} "
                    f"doesn't match features size {N}"
                )
            
            # Normalizza attention weights (opzionale ma consigliato)
            # Questo assicura che non dominino completamente la loss
            attn_normalized = attention_weights / (attention_weights.mean() + 1e-8)
            
            # Costruisci weight matrix: w[i,j] = attention[i] * attention[j]
            # Pixel con alta attention → coppia ha peso alto
            # Pixel con bassa attention → coppia ha peso basso
            weight_matrix = attn_normalized.unsqueeze(1) * attn_normalized.unsqueeze(0)  # (N, N)
            
            # Applica i pesi al similarity kernel
            # Effetto: coppie di pixel entrambi importanti contribuiscono di più
            similarity_kernel = similarity_kernel * weight_matrix
            
            print(f"[DEBUG] Applied attention weighting: "
                f"attn range [{attention_weights.min():.6f}, {attention_weights.max():.6f}], "
                f"normalized mean={attn_normalized.mean():.4f}")
        
        # 6. Compute probability and loss
        prob_before_norm = torch.exp(similarity_kernel)  # Double exp (come EgoLifter)
        
        if self.sum_in_log:
            loss = self._compute_loss_sum_in_log(prob_before_norm, sim_masks, N)
        else:
            loss = self._compute_loss_sum_out_log(prob_before_norm, sim_masks, N)
        
        return loss
    
    def _build_similarity_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Costruisci maschera di similarità (1 se stesso oggetto, 0 altrimenti)
        
        Args:
            labels: (N,) integer labels
            
        Returns:
            sim_mask: (N, N) binary mask
        """
        N = labels.size(0)
        sim_mask = labels.view(-1, 1).repeat(1, N).eq(labels.clone())  # (N, N)
        sim_mask = sim_mask.fill_diagonal_(0, wrap=False)  # Ignora auto-similarità
        return sim_mask.float()
    
    def _compute_distance_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calcola matrice di distanze Euclidee al quadrato
        
        Args:
            features: (N, D)
            
        Returns:
            distance_sq: (N, N)
        """
        # Broadcasting trick: (N, 1, D) - (1, N, D) = (N, N, D)
        distance_sq = torch.pow(
            features.unsqueeze(1) - features.unsqueeze(0), 2
        ).sum(dim=-1)  # (N, N)
        return distance_sq
    
    def _build_temperature_matrix(self, sim_masks: torch.Tensor) -> torch.Tensor:
        """
        Temperatura alta (self.temperature) per positive pairs, 1 per negative
        
        Args:
            sim_masks: (N, N) binary mask
            
        Returns:
            temperature: (N, N)
        """
        temperature = torch.ones_like(sim_masks) * self.temperature
        temperature = torch.where(
            sim_masks == 1, 
            temperature, 
            torch.ones_like(temperature)
        )
        return temperature
    
    def _compute_loss_sum_in_log(
        self, 
        prob_before_norm: torch.Tensor, 
        sim_masks: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Sum-in-log strategy (più robusta, default)
        
        Formula: -1/N * Σ_i log(Σ_j(positive_j) / Σ_k(all_k))
        """
        Z = prob_before_norm.sum(dim=-1)  # (N,) denominatore
        p = torch.mul(prob_before_norm, sim_masks).sum(dim=-1)  # (N,) numeratore (solo positive)
        
        prob = torch.div(p, Z.clamp(min=1e-6))  # clamp invece di +epsilon
        prob_masked = torch.masked_select(prob, prob.ne(0))  # Rimuovi zeri
        
        if prob_masked.numel() == 0:
            return torch.tensor(0.0, device=prob_before_norm.device)
        
        loss = -prob_masked.log().sum() / batch_size
        return loss
    
    def _compute_loss_sum_out_log(
        self, 
        prob_before_norm: torch.Tensor, 
        sim_masks: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:
        """
        Sum-out-log strategy (più precisa, per fine-tuning)
        
        Formula: -1/N * Σ_i Σ_j(positive_j) log(prob_j / |positive|)
        """
        Z = prob_before_norm.sum(dim=-1, keepdim=True)  # (N, 1)
        prob = torch.div(prob_before_norm, Z + 1e-8)  # (N, N)
        log_prob = torch.log(prob + 1e-8)  # (N, N)
        
        # Pesato per numero di positive pairs
        weighted_log_prob = torch.mul(log_prob, sim_masks)  # (N, N)
        num_positive = sim_masks.ne(0).sum(-1, keepdim=True).float() + 1e-6
        weighted_log_prob = weighted_log_prob / num_positive  # (N, N)
        
        log_prob_masked = torch.masked_select(weighted_log_prob, weighted_log_prob.ne(0))
        
        if log_prob_masked.numel() == 0:
            return torch.tensor(0.0, device=prob_before_norm.device)
        
        loss = -log_prob_masked.sum() / batch_size
        return loss
    
    def sample_and_filter(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        ✅ VERSIONE GRADIENT-SAFE: usa weights invece di hard indexing
        """
        # Crea mask di validità come float weights (0 o 1)
        valid_mask = (labels >= 0).float()  # (H*W,)
        
        print(f"[MANAGER] sample_and_filter:")
        print(f"  Total pixels: {labels.numel()}")
        print(f"  Valid (foreground) pixels: {valid_mask.sum().item()}")
        
        if valid_mask.sum() == 0:
            return features[:0], labels[:0], None
        
        # ✅ Invece di indexing, usa gumbel-softmax o top-k differenziabile
        # Per semplicità, usa tutti i pixel validi pesati
        n_valid = int(valid_mask.sum().item())
        
        if n_valid > self.n_samples:
            # Campionamento differenziabile con gumbel trick
            logits = torch.rand_like(valid_mask).log()  # Gumbel noise
            logits = logits * valid_mask + (-1e9) * (1 - valid_mask)  # Mask invalidi
            
            # Top-k sampling (mantiene gradienti attraverso straight-through)
            _, top_indices = torch.topk(logits, self.n_samples)
            
            # Crea one-hot mask per sampling
            sample_mask = torch.zeros_like(valid_mask)
            sample_mask[top_indices] = 1.0
        else:
            sample_mask = valid_mask
        
        # ✅ Usa il mask come weight, non come index
        # Questo mantiene TUTTI i pixel nel computation graph
        sample_mask = sample_mask.unsqueeze(1)  # (H*W, 1)
        
        # Weighted features (mantiene computation graph)
        weighted_features = features * sample_mask
        
        # Per la loss, estrai solo i campionati
        # NOTA: Questo è ancora un indexing, ma ora i gradienti dovrebbero fluire
        # perché valid_mask è usato come weight prima
        sampled_indices = torch.where(sample_mask.squeeze() > 0)[0]
        
        sampled_features = weighted_features[sampled_indices]
        sampled_labels = labels[sampled_indices]
        sampled_attention = attention_weights[sampled_indices] if attention_weights is not None else None
        
        return sampled_features, sampled_labels, sampled_attention
    
    def forward(
        self,
        features: torch.Tensor,
        masks: List[np.ndarray],
        feature_hw: Tuple[int, int],
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Entry point principale per calcolare la loss
        
        Args:
            features: (H, W, D) o (H*W, D) vision features
            masks: Lista di maschere di segmentazione
            feature_hw: (H, W) dimensioni griglia
            
        Returns:
            loss: scalar
        """
        # 1. Prepara features, labels e attention
        features_flat, labels_flat, attention_flat = self.prepare_features_and_labels(
            features, masks, feature_hw, attention_weights  # ← Passa attention
        )
        
        # 2. Campiona e filtra
        sampled_features, sampled_labels, sampled_attention = self.sample_and_filter(
            features_flat, labels_flat, attention_flat  # ← Passa attention
        )
        
        # 3. Verifica che ci siano abbastanza sample validi
        if sampled_features.size(0) < 8:
            return torch.tensor(0.0, device=features.device)
        
        # 4. Compute contrastive loss con attention weighting
        loss = self.compute_loss(
            sampled_features, 
            sampled_labels,
            attention_weights=sampled_attention  # ← Passa attention
        )
        
        return loss


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def strip_cls_and_grid(features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Rimuovi CLS token e converti in griglia 2D
    
    Args:
        features: (seq_len, D) dove seq_len = H*W o H*W+1
        
    Returns:
        features_2d: (H, W, D)
        grid_size: (H, W)
    """
    seq_len = features.size(0)
    
    # Prova senza CLS token
    g = int(round(math.sqrt(seq_len)))
    if g * g == seq_len:
        return features.view(g, g, -1), (g, g)
    
    # Prova con CLS token (rimuovi il primo)
    seq_len_wo_cls = seq_len - 1
    g2 = int(round(math.sqrt(seq_len_wo_cls)))
    if g2 * g2 == seq_len_wo_cls:
        return features[1:].view(g2, g2, -1), (g2, g2)
    
    raise ValueError(f"Unexpected vision sequence length: {seq_len}")