# Robustness-to-Prompt-Variations-in-VLMs-Evaluating-Noisy-Prompts-with-Ensembling-Strategies

## Project Metadata
### Authors
- **Team:** Aishah Altamimi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** SABIC, ARAMCO, KFUPM, IAU

## Introduction
<p style="line-height:1.15;" align="justify">
 Vision Language Models are models that integrate visual and textual modalities to perform tasks and enable sophisticated applications like visual question answering, image captioning, and visual reasoning [1]. 

Recent advances in VLM include Contrastive Language-Image Pretraining (CLIP) and Context Optimization (CoOp) models. CLIP is a zero-shot vision–language model that learns to match images with text rather than relying only on class names. It turns an image into a vector of numbers and a text prompt into another vector, then checks how similar those vectors are. This makes the model rely on prompt templates like “a photo of a {}.” instead of simply using a class name in image classification tasks. This makes the model very sensitive to the structure of prompt templates [2]. To mitigate this limitation, a CoOp has proposed a few-shot prompt learning method that replaces CLIP’s hand-crafted prompts with learnable tokens optimized for specific tasks [3].CoOp  shows high performance with correctly crafted prompts compared to CLIP.
</p>
 

## Problem Statement
<p align="justify">
 Studies show that VLMs such as CLIP and CoOp achieve high accuracy with carefully engineered prompts; however, they are highly sensitive to the way people craft the prompt (wording + structure). Which means accuracy can change a lot based on how the text input is phrased [3]. For example, “a photo of a cat” and “an image of the cat” both have the same meaning however, these variations can result in significant performance differences. Also, adding articles, using synonyms, adding context like “in the wild”, “at night”, or changing word order. All of these can be treated differently by these models.

This limits their usability in noisy, multilingual, or user-generated contexts, which are common real-world situations where user prompts often include noise, typos, or informal expressions.
</p>



## Application Area and Project Domain
<p align="justify">
 This project is an intersection between natural language processing, computer vision, and deep learning. The application area includes systems where users interact with models through text prompts, such as searching for an image, educational tools like a visual homework helper, a recommendation system, and E-commerce visual search, where users frequently introduce informal expressions, non-standard spellings, or typographic noise, making robustness to prompt variation a key requirement. By building a reproducible noise benchmark and evaluating sensitivity, the project directly contributes to the broader domain of robust and trustworthy AI.
</p>


## What is the paper trying to do, and what are you planning to do?
<p align="justify">
 The study in the “Learning to Prompt for Vision-Language Models” paper [3] illustrates the limitation of the CLIP model, in which its accuracy strongly depends on prompt templates, and even small wording changes can noticeably impact predictions, as shown in Figure 1.

<p align="center">
 <img  align="center" alt="image" src="https://github.com/user-attachments/assets/39018743-bb75-4840-bf7b-e49786e9159b" />
  <br>
  <em>Figure 1: Impact of different prompt templates on zero-shot CLIP [2].</em>
</p>

To mitigate this, the paper proposes a new method called Context Optimization (CoOp). This is a prompt learning method that replaces hand-crafted prompts, as in the  CLIP mode, with learnable tokens optimized for classification tasks. Its primary goal is to automate and improve prompt learning.  It shows high performance compared to CLIP, as illustrated in Figure 2.

<p align="center">
<img alt="image" src="https://github.com/user-attachments/assets/0f9756d0-e713-49fc-b0be-822a55c34d0f" />
  <br>
  <em>Figure 2:Prompt engineering vs Context Optimization (CoOp)[3]. </em>
</p>

The paper illustrates that, while CoOp improves performance, the prompts are not human-readable, making them harder to debug, more prone to overfitting, and strongly tied to the trained model, which makes CoOp sensitive to noisy labels. 

This leads to these research Questions:

**RQ1: How sensitive is CLIP  & CoOp to noisy prompts?**

**RQ2: Does prompt ensembling improve robustness consistently?**


Thus, the main objective of this project is, first, to **reproduce baseline performance** by establishing clean accuracy for CLIP and CoOp models using the OxfordPets dataset, saving trained checkpoints, and reporting Top-1 accuracy (what is already done in the paper). Next, will **develop a standardized noise benchmark** by implementing noise functions such as typo, random_case, extra_space, and emoji_tail, while defining severity levels (s=0 clean; s=1 low; s=2 medium; s=3 high). Building on this, the project will **evaluate the sensitivity of CLIP and CoOp to prompt noise** by measuring accuracy when prompts contain different levels of noise—this directly answers the first research question and shows how much accuracy is affected when the prompt is noisy. Then, the project will **propose a robustness mitigation using ensembling**, during test-time (without retraining) as a low-cost robustness strategy. Specifically, accuracy will be evaluated at noise levels s=1/2/3 using different ensemble sizes (K=3, K=5). This will be done by constructing K prompt variants for each class (e.g., one clean + K−1 noisy/paraphrased). For each image, the model will be run across all prompts, logits will be converted to probabilities using the softmax function, the probabilities will be averaged across prompts, and the final prediction will be selected with argmax. This will answer the second research question, showing whether prompt ensembling improves the robustness of VLMs consistently.
</p>

## References
[1]	Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, and G. Shi, “A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges,” no. 1, 2025, [Online]. Available: http://arxiv.org/abs/2501.02189

[2]	A. Li, Z. Liu, X. Li, J. Zhang, P. Wang, and H. Wang, “Modeling Variants of Prompts for Vision-Language Models,” 2025, [Online]. Available: http://arxiv.org/abs/2503.08229

[3]	K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to Prompt for Vision-Language Models,” Int. J. Comput. Vis., vol. 130, no. 9, pp. 2337–2348, 2022, doi: 10.1007/s11263-022-01653-1.



# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
