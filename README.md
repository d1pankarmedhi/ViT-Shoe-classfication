# ViT-vision-transformer Shoe classification

ViT paper implementation in PyTorch for classification of shoe brands.

## Classification of Shoe brand using ViT 🪙

There are total 3 brand shoe images in the dataset and the object is to classify them as `Adidas` , `Converse` or `Nike`.

## Final result looks like this 🧑‍🍳

<div>
    <div>
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/4fc00ff4-958d-47c0-bd07-443ddf024fb4" alt="loss and accuracy">
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/bff5e439-9655-4831-b3e5-19d6c0dcb52a" alt="Image 3">
    </div>
</div>

## Model Summary

A complete model summary of the Vision Transformer with around `86M` parameters. This implementation is based on the **ViT-Base** model.

<div>
<img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/21dde5ed-e9db-46de-9ec5-101854ed24d9">
</div>

For blocks, embeddings scripts, check out the `vit` directory under the `blocks` directory. This has all the architecture code, including the patch embedding, multi head attention, multi layer perceptron and the transformer encoder block code.
