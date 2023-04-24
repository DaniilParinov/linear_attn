import torch
from torch import nn
from torch.nn.functional import elu
from transformers import BartConfig, BartForSequenceClassification
from typing import Optional, Tuple

MODEL_NAME = "facebook/bart-base"


class LinearAttention(nn.Module):
    """Multi-headed linear attention from Transformers are RNNs: Fast Autoregressive Transformers with Linear
    Attention. Unmasked version so will be used only in decoder part, unfortunately since it unmasked function will
    attend to pad tokens"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != self.embed_dim:
            error_msg = f"embed_dim must be divisible by num_heads " \
                        f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            raise ValueError(error_msg)

        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # mapping function for softmax linearization
        self.map_fn = lambda x: elu(x) + 1

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Internal method which is used for projection of 3D tensor to 4D with head_qty dimension"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
                hidden_states: torch.Tensor,
                key_value_states: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                layer_head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: batch x seq_len x emb_dim"""

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj and map them to another space
        query_states = self.map_fn(self.q_proj(hidden_states))  # shape: [batch_size, tgt_len, embed_dim]

        # as result will get k, v tensors with shape: [batch_size, num_heads, head_dim, src_len]
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.map_fn(self.k_proj(hidden_states)), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.map_fn(self.k_proj(hidden_states)), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        src_len = key_states.size(2)

        """by changing this part you should understand that this code does not support masking
        so should be used with care in decoder"""
        assert not self.is_decoder
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, -1, bsz)  # [bsz, num_heads, tgt_len, head_dim]
        KV = torch.einsum("bhsd,bhsm->bhmd", key_states, value_states)  # [bsz, num_heads, head_dim, head_dim]
        Z = 1 / (torch.einsum("bhsd,bhd->bhs", query_states, key_states.sum(dim=2)) + 1e-6)  # [bsz, num_heads, tgt_len]
        attn_output = torch.einsum("bhsd,bhmd,bhs->bshm", query_states, KV, Z)  # [bsz, tgt_len, num_heads, head_dim]

        if output_attentions:
            raise ValueError('Linear attention does not support attn weights by it is nature')
        else:
            attn_weights_reshaped = None

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim).contiguous()

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


def get_model(
    custom_attention: bool = False,
    glue_task: str = "mnli",
    path: Optional[str] = None
) -> BartForSequenceClassification:
    """Function for downloading pretrained model(classification head newly initialized)
     for MNLI and QNLI task and changing attention to custom one if required.
     If path is specified downloads model weights from path"""
    
    config = BartConfig.from_pretrained(MODEL_NAME)
    
    if path:
        model = BartForSequenceClassification.from_pretrained(path)
    else:
        possible_tasks = ["qnli", "mnli"]
        if glue_task == "qnli":
            config.num_labels = 2
            config.id2label = {0: "ENTAILMENT", 1: "NOT_ENTAILMENT"}
            config.label2id = {"ENTAILMENT": 0, "NOT_ENTAILMENT": 1}
        elif glue_task == "mnli":
            config.num_labels = 3
            config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
            config.label2id = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
        else:
            raise NotImplementedError(f"No such task configuration, possible tasks are {possible_tasks}")

        model = BartForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    if custom_attention:
        attention_weights = []
        for i in range(config.encoder_layers):
            attention_weights.append(model.model.encoder.layers[i].self_attn.state_dict())
        for i in range(config.encoder_layers):
            model.model.encoder.layers[i].self_attn = LinearAttention(
                config.d_model, config.encoder_attention_heads, config.dropout
            )
            model.model.encoder.layers[i].self_attn.load_state_dict(attention_weights[i])
    return model
