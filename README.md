# Latent Flow Matching for Flower Generation

## Model

A lightweight **Latent Flow Matching** model with approximately **10 million parameters**, capable of generating **256×256 images** conditioned on class labels **(class-to-image)**.

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

<img width="3499" height="4769" alt="gen4" src="https://github.com/user-attachments/assets/fa518b51-0801-4d6f-a4bb-747829b94e46" />
<img width="3499" height="4769" alt="gen16" src="https://github.com/user-attachments/assets/ecbba41d-20aa-4740-8929-16263c80b3b4" />
<img width="3499" height="4769" alt="gen33" src="https://github.com/user-attachments/assets/65f8a7a7-45ee-4030-9b1f-defafa2f0404" />
<img width="3499" height="4769" alt="gen62" src="https://github.com/user-attachments/assets/1a46bd06-45b2-4799-a57d-79d1e3990add" />
<img width="3499" height="4769" alt="gen57" src="https://github.com/user-attachments/assets/0b632d0b-87a2-412b-9f3a-c3a1a386dc6a" />

The model learned colors, backgrounds and shapes of the different types of flowers.

However as we can observe the model fails to successfully generate a realistic image. The generated images lack depth and contrast, some flower like textures and structures can be seen but the image is not realistic at all.

The size of the model **(10M parameters)** is one of the main reasons for its unrealistic images. This model was built as a learning experience and the size of the model was selected for fast trraining and inference due to limited computational resources.

A bigger model should provide better results with the same architecture and training methodologies.
## Conclusions
This project was started as a learning experience to understand image generation and attention mechanisms better. Overall the development process was very enlightening and the results satisfactory.


