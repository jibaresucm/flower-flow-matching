# Latent Flow Matching for Flower Generation

## Model

A lightweight **Latent Flow Matching** model with approximately **4 million parameters**, capable of generating **256×256 images** conditioned on class labels **(class-to-image)**.

The model is trained on the **Oxford 102 Flowers dataset** to perform class-to-image generation.

The model follows a classic conditioned DiT architecture, convolutions to generate the token embeddings and multi head attention blocks to learn the direction the latent needs to move in.

Learnable embeddings are used to implement the class condition into the latent flow matching.

It uses a pretrained VAE Encoder to compute the latents for the training of the model and the matching Decoder to decode the finished latent into the final image.
## Files

Everything needed to build and run the model is included in `flowerdiff.ipynb`. The only additional requirement is to download the dataset.

## How to Run

At the end of `flowerdiff.ipynb`, the code required to run the trained model and generate images is provided.

## Results

These are some of the results
<img width="3499" height="4769" alt="gen4" src="https://github.com/user-attachments/assets/5ef86ba6-506e-42fd-a649-05983b7471ee" />
<img width="3499" height="4769" alt="gen62" src="https://github.com/user-attachments/assets/aa2c9eea-f364-4de1-8025-79241ea813ad" />
<img width="3499" height="4769" alt="gen57" src="https://github.com/user-attachments/assets/f79bde40-fd4b-4511-88ee-a6b1a07d6066" />
<img width="3499" height="4769" alt="gen16" src="https://github.com/user-attachments/assets/29f74a08-fade-4c17-88ae-b32b8a2ce014" />

The model learned colors, backgrounds and shapes of the different types of flowers.

However as we can observe the model fails to successfully generate a realistic image. The generated images lack depth and contrast, some flower like textures and structures can be seen but the image is not realistic at all.

The size of the model **(4M parameters)** is one of the main reasons for its unrealistic images. This model was built as a learning experience and the size of the model was selected for fast trraining and inference due to limited computational resources.

A bigger model should provide better results with the same architecture and training methodologies.
## Conclusions
This project was started as a learning experience to understand image generation and attention mechanisms better. Overall the development process was very enlightening and the results satisfactory.


