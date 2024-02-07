<div align="center">
<h1>ViT - Vision Transformer Shoe classification</h1>

ViT paper implementation in PyTorch for classification of shoe brands.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

</div>

## Classification of Shoe brand using ViT 🪙

There are total 3 brand shoe images in the dataset and the object is to classify them as `Adidas` , `Converse` or `Nike` based on the input image.

## Final result looks like this 🧑‍🍳

<div>
    <div>
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/4fc00ff4-958d-47c0-bd07-443ddf024fb4" alt="loss and accuracy">
        <img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/bff5e439-9655-4831-b3e5-19d6c0dcb52a" alt="Image 3">
    </div>
</div>

## Model Summary 🏋️

A complete model summary of the Vision Transformer with around `86M` parameters. This implementation is based on the **ViT-Base** model.

Due to data and hardware limitations, the original implementation model did not perform well. To solve this issue, the VIT model weights from pytorch was used.

[VIT_B_16](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#vit-b-16) weights are used for transfer learning.

<div>
<img src="https://github.com/d1pankarmedhi/ViT-vision-transformer/assets/136924835/21dde5ed-e9db-46de-9ec5-101854ed24d9">
</div>

For blocks, embeddings scripts, check out the `vit` directory under the `blocks` directory. This has all the architecture code, including the patch embedding, multi head attention, multi layer perceptron and the transformer encoder block code.
