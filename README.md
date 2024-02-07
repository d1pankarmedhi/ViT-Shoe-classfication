# ViT-vision-transformer Shoe classification

ViT paper implementation in PyTorch for classification of shoe brands.

## Classification of Shoe brand using ViT 🪙

There are total 3 brand shoe images in the dataset and the object is to classify them as `Adidas` , `Converse` or `Nike`.

## Final result looks like this 🧑‍🍳

<div style="display: flex">
<div>
    <div>
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/e8de7c64-3547-4698-a3fa-1391991a5f8b" width=270 alt="Image 1" >
    </div>
    <div>
    <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/4a95bb01-5207-43a8-ac82-8fe55ad15f25" width=265 alt="Image 2">
    </div>
</div>
<div style="margin-left: 10px;">
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/bff5e439-9655-4831-b3e5-19d6c0dcb52a" alt="Image 3">
</div>
</div>

## Model Summary

A complete model summary of the Vision Transformer with around `86M` parameters. This implementation is based on the **ViT-Base** model.

<div>
<img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/21dde5ed-e9db-46de-9ec5-101854ed24d9">
</div>

For blocks, embeddings scripts, check out the `vit` directory under the `blocks` directory. This has all the architecture code, including the patch embedding, multi head attention, multi layer perceptron and the transformer encoder block code.
