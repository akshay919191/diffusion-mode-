🖼️ Text-to-Image Diffusion Model (DDPM)

This project implements a Denoising Diffusion Probabilistic Model (DDPM) with text conditioning, built entirely using TensorFlow and Keras. This architecture is inspired by modern generative models and is capable of generating high-quality images conditioned on semantic input (labels or short phrases).

🚀 Key Features

Custom DDPM Implementation: Full implementation of the forward (noise scheduler) and reverse (sampling) process.

Conditioned U-Net: The noise prediction network (U-Net) is conditioned on both the time step and a text embedding, enabling text-to-image generation.

Transformer-based Text Encoder: Includes custom layers for processing textual input and generating semantic embeddings.

Modular Design: Code is separated into logical components (ddpm_unet, reverse, text_encoder, etc.) for clarity and maintenance.

📦 Project Structure

The repository is organized into the core model components and the main training script:

.
├── ddpm_project.py      # Main training and generation pipeline (Keras Model class)
├── ddpm_unet.py         # The U-Net architecture (noise predictor)
├── ddpm_forward.py      # Defines the noise scheduler (alphas, betas, variances)
├── reverse.py           # Implements the reverse (sampling/denoising) step
├── text_encoder.py      # Custom transformer-based text encoder
├── cross.py             # Cross-Attention layer for integrating text embeddings
├── self.py              # High-level Self-Attention components
├── decoder.py           # Decoder and Residual blocks
├── encoder.py           # Encoder blocks (using padding for stride 2 convolutions)
└── README.md


🛠️ Requirements and Setup

This project requires TensorFlow and several associated libraries.

Prerequisites

TensorFlow: This entire project is built using TensorFlow 2.x and Keras.

Dependencies:

pip install tensorflow numpy matplotlib


Running the Model

The main script handles data loading (CIFAR-10) and the training loop.

Start Training: Execute the main project file. The script will automatically download the CIFAR-10 dataset (if not already present).

python ddpm_project.py


Training Parameters:

Time Steps: 1000

Image Size: 32x32 (for CIFAR-10)

Optimizer: Adam with a learning rate of 1e-4.

🧠 Architectural Details

1. The Core Model (ddpm_project.py)

The Diffusion_model class inherits from keras.Model and manages the entire process:

Noise Scheduling: Initializes the $\beta_t$ and $\alpha_t$ parameters using noise_scheduler from ddpm_forward.py.

Loss Calculation: The model is trained to minimize the mean squared error (MSE) between the actual noise added to the image ($\epsilon$) and the noise predicted by the U-Net ($\epsilon_\theta$).

Training Loop (train_step):

Sample a random time step $t \sim [1, T]$.

Sample random noise $\epsilon \sim \mathcal{N}(0, I)$.

Generate the noisy image $x_t$ using the closed-form forward process equation.

Predict the noise $\epsilon_\theta(x_t, t, \text{text\_embed})$.

Compute loss: $\text{Loss} = ||\epsilon - \epsilon_\theta||^2$.

2. Conditioned U-Net (ddpm_unet.py, cross.py)

The U-Net is the backbone of the noise prediction.

It follows a standard contraction (downsampling) and expansion (upsampling) path with skip connections.

It incorporates Time Embeddings (positional encodings for the time step $t$).

It integrates Text Embeddings using Cross-Attention layers (cross.py) to guide the noise prediction based on the input prompt.

3. Text Encoder (text_encoder.py, clip.py)

This module transforms simple text labels (e.g., "cat", "dog") into a dense, semantic vector used for conditioning.

It uses a custom Transformer block structure (Cliplayer) to process the token embeddings and generate a rich, contextualized representation of the input text.

✨ Image Generation (Inference)

Image generation is handled by the generate_images function in reverse.py. This is the iterative denoising process:

Start with pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$.

Iterate backward from $t=T$ to $t=1$.

At each step $t$, the U-Net predicts the noise $\epsilon_\theta(x_t, t, \text{text\_embed})$.

The prediction is used in the reverse diffusion formula to calculate $x_{t-1}$, gradually reducing the noise and revealing the final image $x_0$.
