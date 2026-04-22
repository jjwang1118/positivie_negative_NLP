"""
model.py — Attention Pooling Classifier for Exp01 (Input Reduction).

Architecture
------------
  Text
    → SBERT encoder (frozen)  — sentence-transformers/all-MiniLM-L6-v2
    → token embeddings  [B, T, 384]
    → Attention Layer   Linear(384 → 1) + softmax over T
    → Weighted sum      [B, 384]
    → MLP head          Linear(384→64) → ReLU → Dropout → Linear(64→2)
    → logits            [B, 2]

Only the Attention Layer and MLP head are trained; the SBERT encoder is kept
frozen so that token embeddings remain stable for the key-string extraction step.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Attention Pooling Classifier
# ---------------------------------------------------------------------------

class AttentionPoolingClassifier(nn.Module):
    """
    Parameters
    ----------
    pretrained_name : str
        HuggingFace model identifier for the SBERT backbone.
    hidden_dim : int
        Output dimension of the SBERT encoder (384 for all-MiniLM-L6-v2).
    mlp_hidden : int
        Intermediate dimension of the MLP classification head.
    dropout : float
        Dropout probability applied before the final linear layer.
    num_labels : int
        Number of output classes (2 for binary sentiment).
    freeze_encoder : bool
        If True (default), freeze all SBERT parameters.
    """

    def __init__(
        self,
        pretrained_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 384,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        num_labels: int = 2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # --- Backbone ---
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # --- Attention Layer ---
        # Produces a scalar importance score per token, then softmax over T
        self.attention = nn.Linear(hidden_dim, 1)

        # --- MLP Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids       : (B, T)
        attention_mask  : (B, T)  — 1 for real tokens, 0 for padding
        token_type_ids  : (B, T)  — optional, ignored by most SBERT models
        return_attention_weights : bool
            If True, also return the softmax attention weight tensor (B, T).

        Returns
        -------
        logits              : (B, num_labels)
        attention_weights   : (B, T)   — only when return_attention_weights=True
        """
        # 1. Encode — token-level embeddings: (B, T, H)
        encoder_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        token_embeddings = outputs.last_hidden_state  # (B, T, H)

        # 2. Attention scores — (B, T, 1) → (B, T)
        attn_scores = self.attention(token_embeddings).squeeze(-1)  # (B, T)

        # Mask padding positions before softmax (set to -inf)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, T)

        # 3. Weighted sum → sentence vector (B, H)
        sentence_vec = torch.bmm(
            attn_weights.unsqueeze(1),   # (B, 1, T)
            token_embeddings,            # (B, T, H)
        ).squeeze(1)                     # (B, H)

        # 4. Classify
        logits = self.classifier(sentence_vec)  # (B, num_labels)

        if return_attention_weights:
            return logits, attn_weights
        return logits


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_tokenizer(pretrained_name: str):
    """Return the AutoTokenizer for the given HuggingFace model identifier."""
    return AutoTokenizer.from_pretrained(pretrained_name)


# ---------------------------------------------------------------------------
# Sequence Classifier (for full fine-tune — Exp03)
# ---------------------------------------------------------------------------

class SequenceClassifier(nn.Module):
    """
    Full fine-tune classifier using [CLS] token pooling.
    Compatible with any BERT/RoBERTa-family model.
    All encoder parameters are trainable by default.
    """

    def __init__(
        self,
        pretrained_name: str = "roberta-base",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        num_labels: int = 2,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        cls_vec = outputs.last_hidden_state[:, 0, :]   # [CLS] token: (B, H)
        return self.classifier(self.dropout(cls_vec))   # (B, num_labels)


def build_model_from_config(hparams: dict, freeze_encoder: bool = True) -> AttentionPoolingClassifier:
    """
    Instantiate AttentionPoolingClassifier from an EXP01_HPARAMS-style dict.

    Parameters
    ----------
    hparams       : dict — expects keys: model_name (resolved via MODEL_REGISTRY),
                    hidden_dim, mlp_hidden, dropout, num_labels.
    freeze_encoder : bool — passed through to the constructor.

    Returns
    -------
    AttentionPoolingClassifier
    """
    from src.config import MODEL_REGISTRY  # local import to avoid circular deps

    pretrained_name = MODEL_REGISTRY[hparams["model_name"]]
    return AttentionPoolingClassifier(
        pretrained_name=pretrained_name,
        hidden_dim=hparams["hidden_dim"],
        mlp_hidden=hparams["mlp_hidden"],
        dropout=hparams["dropout"],
        num_labels=hparams["num_labels"],
        freeze_encoder=freeze_encoder,
    )
