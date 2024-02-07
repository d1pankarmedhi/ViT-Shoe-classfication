from torch import nn 

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(
            self, 
            embedding_dim: int = 768, # Table 1 from paper for ViT Base
            num_heads: int = 12, # From Table 1 from paper
            attention_dropout: float = 0,  # No dropout mentioned in paper
            ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(
            normalized_shape=embedding_dim
            )
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True, 
        ) 
        
    def forward(self, x):
        x = self.layer_norm(x)
        attention_output, _ = self.multihead_attention(
            query=x,
            key=x,
            value=x, 
            need_weights=False,
        )
        return attention_output