from torch import nn 



class PatchEmbedding(nn.Module):
    
    def __init__(
            self,
            in_channels: int = 3, # image channels
            patch_size:int = 16,  # Table 5 from Paper
            embedding_dimension: int = 768, # Table 1 of paper 
        ) -> None:
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dimension = embedding_dimension

        self.patcher = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embedding_dimension,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )
        self.flatten = nn.Flatten(
            start_dim=2, 
            end_dim=3
        )
    
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)