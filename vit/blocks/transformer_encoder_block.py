from torch import nn 
from vit.blocks.multiheadselfattn_block import MultiheadSelfAttentionBlock
from vit.blocks.multilayerperceptron_block import MultiLayerPerceptronBlock

class TransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768, # From Table 1 of paper
            num_heads:int = 12, # From Table 1 for ViT-base
            mlp_size: int = 3072, # From Table 1 for ViT base
            mlp_dropout:float = 0.1, # From Table 3 for ViT base
            attention_droput: float = 0, # For attention layer
        ) -> None:
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dimension=embedding_dim,
            num_heads=num_heads,
            attention_dropout=attention_droput
        )

        self.mlp_block = MultiLayerPerceptronBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout=mlp_dropout
        )


    def forward(self, x):
        x = self.msa_block(x) + x # residual connections
        x = self.mlp_block(x) + x 
        return x 