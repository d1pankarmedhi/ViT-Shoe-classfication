from torch import nn 

class MultiLayerPerceptronBlock(nn.Module):

    def __init__(
            self,
            embedding_dim:int=768, # Table 1 from paper for ViT-Base
            mlp_size:int=3072, # From Table 1 for ViT-Base
            dropout:float=0.1 # From Table 3 for ViT-Base
        ): 
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # as per paper, multilayer perceptron block contains 2 Linear layers and a GELU non-linear layer
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), 
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, 
                      out_features=embedding_dim),
            nn.Dropout(p=dropout) 
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x