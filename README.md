# Robustness-to-Prompt-Variations-in-VLMs-Evaluating-Noisy-Prompts-with-Ensembling-Strategies

## Project Metadata
### Authors
- **Team:** Aishah Altamimi
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM, IAU

## Introduction
<p align="justify">
Recent advances in vision–language models (VLMs) include Contrastive Language–Image Pretraining (CLIP), SigLIP (Sigmoid-based Language–Image Pre-Training), and Context Optimization (CoOp). CLIP and SigLIP are zero-shot VLMs that learn to match images with text prompts rather than class names. They encode an image into a vector and a text prompt into another vector, then measure similarity between the two vectors. As this approach depends on text prompt templates such as “a photo of a {class}” instead of the class name. This makes the model’s behavior affected by the way prompts are structured [2]. To mitigate this sensitivity, CoOp has been proposed, which replaces CLIP’s handcrafted prompts with learnable continuous tokens optimized for specific tasks [3]. CoOp shows high performance with clean prompts. These models facilitate many tasks such as zero-shot image classification, caption generation, and image–text retrieval [3]. Additionally, VLMs are increasingly used in real-world applications where users interact with systems through natural language prompts—such as image search engines, educational tools (e.g. visual homework helpers), recommendation systems, and e-commerce visual search. In these applications, user queries often contain mistakes, informal phrasing,non-standard spellings, or typographic noise. We observed
that these models give high accuracy with clean prompts as illustrated in Figure1 (a); however, they show different behavior under noisy prompts Figure1 (b), and we noticed that Ensembling during test time recovers the accuracy Figure1 (c).
</p>


<p align="center">
 <img  align="center" alt="image"  width="300" height="400" src="https://github.com/BRAIN-Lab-AI/Robustness-to-Prompt-Variations-in-VLMs-Evaluating-Noisy-Prompts-with-Ensembling-Strategies/blob/main/Fig1.png?raw=true" />
  <br>
  <em>Fig. 1. Illustration of how noisy prompts affect VLMs on the Birman class (Oxford Pets). Clean prompt = correct (a), noisy prompt = incorrect (b), ensembling recovers accuracy (c).</em>
</p>
 

## Problem Statement
<p align="justify">
 Studies show that VLMs such as CLIP achieve high accuracy with carefully engineered prompts; however, they are highly sensitive to the way people craft the prompt (wording + structure). Which means accuracy can change a lot based on how the text input is phrased [3]. For example, “a photo of a cat” and “an image of the cat” both have the same meaning however, these variations can result in significant performance differences. Also, adding articles, using synonyms, adding context like “in the wild”, “at night”, or changing word order. All of these can be treated differently by these models.

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
 <img  align="center" alt="image" src="https://github.com/user-attachments/assets/39018743-bb75-4840-bf7b-e49786e9159b"  width="400" height="400"/>
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

**RQ1: How sensitive is CLIP, SigLIP & CoOp to noisy prompts?**

**RQ2: Does prompt ensembling improve robustness consistently?**


Thus, the main objective of this project is, first, to **reproduce baseline performance** by establishing clean accuracy for CLIP,SigLIP and CoOp models using the OxfordPets dataset, saving trained checkpoints, and reporting Top-1 accuracy (what is already done in the paper). Next, will **develop a standardized noise benchmark** by implementing noise functions such as typo, random_case, extra_space, and emoji_tail, while defining severity levels (s=0 clean; s=1 low; s=2 medium; s=3 high). Building on this, the project will **evaluate the sensitivity of CLIP,SigLIP and CoOp to prompt noise** by measuring accuracy when prompts contain different levels of noise—this directly answers the first research question and shows how much accuracy is affected when the prompt is noisy. Then, the project will **propose a robustness mitigation using ensembling**, during test-time (without retraining) as a low-cost robustness strategy. Specifically, accuracy will be evaluated at noise levels s=1/2/3 using different ensemble sizes (K=1, K=5). This will be done by constructing K prompt variants for each class (e.g., one clean + K−1 noisy/paraphrased). For each image, the model will be run and evalauted across all prompts usind 5 benchmark datasets. This will answer the second research question, showing whether prompt ensembling improves the robustness of VLMs consistently.
</p>

## References
[1] Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, and G. Shi, “A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges,” 2025. [Online]. Available: http://arxiv.org/abs/2501.02189
[2] A. Li, Z. Liu, X. Li, J. Zhang, P. Wang, and H. Wang, “Modeling Variants of Prompts for Vision-Language Models,” 2025. [Online]. Available: http://arxiv.org/abs/2503.08229
[3] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to Prompt for Vision- Language Models,” Int. J. Comput. Vis., vol. 130, no. 9, pp. 2337–2348,2022.
[4] X. Zhai et al., “Sigmoid Loss for Language Image Pre-Training,” in Proc. IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 11975–11986.
[5] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-Language Models for Vision Tasks: A Survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, pp. 1–24.
[6] Q. Ye, M. Axmed, R. Pryzant, and F. Khani, “Prompt Engineering a Prompt Engineer,” 2024, pp. 355–385.
[7] Z. Li, B. Peng, P. He, and X. Yan, “Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection,” 2024, pp.557–568.
[8] Q. Xie, Z. Dai, E. Hovy, M. Luong, and Q. V Le, “Unsupervised Data Augmentation for Consistency Training,” no. NeurIPS, pp. 1–13, 2020.
[9] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar, “Cats and dogs,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012. Available: https://www.robots.ox.ac.uk/∼vgg/data/pets/
[10] L. Fei-Fei, R. Fergus, and P. Perona, “Learning generative visual models from few training examples: An incremental Bayesian approach tested on 101 object categories,” in 2004 Conference on Computer Vision and Pattern Recognition Workshop, 2004, pp. 178–178. Available: https://data.caltech.edu/records/mzrjq-6wc02
[11] L. Bossard, M. Guillaumin, and L. Van Gool, “Food-101 – mining discriminative components with random forests,” in European Conference on Computer Vision (ECCV), 2014, pp. 446–461. Available: https://www.kaggle.com/datasets/dansbecker/food-101
[12] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, “Describing textures in the wild,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 3606–3613. Available:https://www.robots.ox.ac.uk/∼vgg/data/dtd/
[13] P. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217–2226, 2019. Available:https://github.com/phelber/eurosat


# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [Learning to Prompt for Vision-Language Models](https://dl.acm.org/doi/10.1007/s11263-022-01653-1)

### Reference Dataset
-**Oxford Pets:**
https://www.robots.ox.ac.uk/~vgg/data/pets/

-**Food101:**
https://www.vision.ee.ethz.ch/datasets_extra/food-101/

-**DTD:**
https://www.robots.ox.ac.uk/~vgg/data/dtd/

-**EuroSAT:**
https://github.com/phelber/EuroSAT

-**Caltech-101:**
https://data.caltech.edu/records/20086


## Project Technicalities

### Terminologies
- **Vision–Language Model (VLM):** A model that connects images and text so it can understand and relate both at the same time.
- **CLIP:** A vision–language model that learns by matching images with their text descriptions and separating them from mismatched pairs.
- **CoOp (Context Optimization):** A method that learns the best text prompts automatically instead of using fixed, hand-written prompts.
- **Prompt:** The text input given to the model (like “a photo of a cat”) that tells it what to look for in an image.
- **Noisy Prompt:** A prompt that contains small mistakes such as typos, random casing, extra spaces, or emojis.
- **Prompt Ensembling:** A technique where we use several different versions of a prompt and combine the model’s outputs to get a more stable and accurate prediction.
- **Robustness:** The ability of a model to keep working well even when the input is imperfect, noisy, or slightly changed.
- **Contrastive Learning:** A training method where the model pulls matching image–text pairs closer together and pushes non-matching pairs further apart.
- **Text Encoder:** The part of the model that converts a text prompt into a vector (numeric representation) in embedding space.
- **Image Encoder:** The part of the model that converts an image into a vector so it can be compared with text vectors.
- **Zero-Shot Classification:** When the model can recognize new classes it was not explicitly trained on, just by using text prompts.
-**Consistency Regularization:** A training idea that encourages the model to give similar predictions for clean and noisy versions of the same input.
- **Noise-Aware Fine-Tuning:** A training phase where the model (or adapter) is updated using noisy prompts so it learns to handle them better.
- **Adapter:** A small extra module added to a pre-trained model to adjust its behavior without retraining the whole model.
- **Ensemble Size (K):** The number of different prompts combined together when using prompt ensembling.


### Problem Statements
- **Problem 1:** Vision–language models like CLIP and SigLIP show strong performance only when the prompt is perfectly phrased, making them highly sensitive to small wording changes.
- **Problem 2:** Semantically similar prompts (e.g., “a photo of a cat” vs. “an image of the cat”) can produce very different predictions, leading to unstable and unpredictable model behavior.
- **Problem 3:** Minor text variations such as typos, casing differences, extra spaces, or added context (e.g., “in the wild,” “at night”) can cause significant drops in accuracy.
- **Problem 4:** Current manually designed prompts (used in CLIP and SigLIP) are not robust to natural user mistakes, which limits real-world usability.
- **Problem 5:** It is unclear how different training objectives (softmax vs. sigmoid vs. learned prompts) affect robustness, and there is no unified evaluation across multiple models.
- **Problem 6:** There is a need to evaluate and compare the robustness of VLMs—CLIP, SigLIP, and CoOp—under noisy prompts and investigate whether ensembling can reduce performance degradation.


### Loopholes or Research Areas

- **Lack of Robustness:** Existing VLMs (e.g., CLIP, SigLIP) cannot handle small prompt mistakes, yet real users make such errors often.
- **Limited Understanding:** There is no clear explanation of why different models react differently to noisy prompts.
- **Missing Benchmarks:** Few works evaluate multiple VLMs across several datasets under controlled noise conditions.
- **Prompt Dependency:** Models still rely heavily on hand-written prompts, which are not scalable.
- **Training Objective Uncertainty:** The impact of softmax vs. sigmoid vs. learned prompts on robustness is not well explored.
- **Data Limitations:** Research rarely examines robustness on diverse datasets like EuroSAT, DTD, or Food101.
-**Inference Overhead:** Existing solutions for robustness often increase computational cost.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
- **Prompt Ensembling Strategy:** Combine multiple versions of a prompt (clean + noisy variations) to stabilize predictions and reduce sensitivity to typos, casing, and added context.
- **Noise-Aware Training / Fine-Tuning:** Introduce controlled prompt noise during training to help the model learn stable representations and become more robust to real-world user errors.
- **Cross-Model Robustness Evaluation Framework:** Build a unified evaluation pipeline to compare CLIP, SigLIP, and CoOp under different noise types and severities to identify which training objective handles noise best.

### Proposed Solution: Code-Based Implementation
This project provides a complete implementation for evaluating and improving the robustness of vision–language models (CLIP, SigLIP, and CoOp) under noisy prompts. The solution includes:

-**Noise Bank Generation:** A dedicated module that automatically creates multiple noisy versions of any prompt, including typos, random casing, extra spaces, and emojis.

-**Robustness Evaluation with Ensembling:** All three models (CLIP, SigLIP, CoOp) are evaluated using clean and noisy prompts. An ensembling strategy is applied to combine predictions from different prompt variants, significantly improving stability under noise.

-**Noise-Aware Adapter for CLIP:** A lightweight adapter is implemented and trained with a consistency regularization loss. This enables CLIP to learn from noisy prompts directly, reducing sensitivity and improving robustness without retraining the full model.

### Key Components

- **prompt_noise_eval_v4.py:** Main CLIP evaluation script supporting five datasets, four noise types, and three ensembling strategies (K=1, K=5, K=5+Clean).
- **train_prompt_noise_adapter.py:** Noise-aware adapter training script using Cross-Entropy and KL divergence regularization to fine-tune CLIP under noisy prompts.
- **prompt_noise_eval_siglip.py:** SigLIP evaluation script built with HuggingFace transformers, supporting prompt noise, severity levels, and prompt ensembling.
- **coop_noise_eval_min_v5.py:** CoOp robustness evaluation script that tests learned prompts under noisy class-name corruption.
- **Noise Operators (functions):** Four prompt-corruption functions (typo, case, space, emoji) defined inside all evaluation and training scripts to generate controlled noise at severities 0–3.
- **TextAdapter (class):** Identity-initialized linear adapter used in training and evaluation scripts to learn noise-robust text embeddings.
- **SigLIP_Robustness.ipynb:** Jupyter notebook used for running experiments, plotting results, and validating all robustness evaluations.

## Model Workflow
The workflow of the **Robust Vision-Language Evaluation Framework** is designed to measure how CLIP, SigLIP, and CoOp behave under clean and noisy prompts, and how ensembling or noise-aware fine-tuning improves their robustness.

---

## 1. Input

### Clean and Noisy Prompts

Class names (e.g., "Siamese cat") are inserted into a prompt template:
- `"a photo of a {class}"`

Four noise transformations create corrupted versions:
- **typo** - Character substitutions, deletions, or swaps
- **case-change** - Random capitalization alterations
- **extra space** - Insertion of whitespace characters
- **emoji tail** - Addition of emoji characters

Each applied with **severity levels 0–3** (0 = clean, 3 = maximum corruption).

### Prompt Banks (for Ensembling)

For each class:
- **K = 1** → single prompt (clean or noisy)
- **K = 5** → five noisy variants
- **K = 5 + Clean** → one clean + four noisy variants

These are used to test noisy-prompt robustness and ensemble stability.

---

## 2. Evaluation Process

### Text Encoding

Each prompt (clean or noisy) is processed by the model's text encoder:
- **CLIP** → CLIP text encoder
- **SigLIP** → SigLIP text encoder (HuggingFace)
- **CoOp** → Learned context encoder

### Image Encoding

Each input image is passed through the corresponding image encoder to produce image embeddings.

### Similarity & Prediction Logic

#### CLIP & CoOp
- Use **cosine similarity × 100**
- Apply **softmax** over classes to get class probabilities

#### SigLIP
- Does **NOT** use softmax
- Instead uses **sigmoid-based pairwise scoring** between image and text embeddings
- Scores are normalized or directly averaged (matching the official SigLIP inference procedure)

This ensures the workflow matches the true SigLIP inference mechanics.

### Prompt Ensembling (K > 1)

For models supporting ensembling:
- **CLIP / CoOp** → average of softmax probabilities
- **SigLIP** → average of sigmoid pairwise matching scores
- The class with the **highest averaged score** becomes the final prediction

This step greatly increases stability under noise.

### Noise-Aware Adapter (Optional for CLIP)

When using the trained adapter:
1. Text embeddings are passed through the `TextAdapter`
2. The adapter modifies them to be more stable against prompt noise
3. Prediction proceeds as usual for each model type

---

## 3. Output

### Accuracy Results

For each noise severity (0–3), the framework reports:
- Clean accuracy
- Noisy accuracy  
- Ensemble vs. non-ensemble performance
- **Model comparisons:**
  - CLIP vs. SigLIP vs. CoOp
  - CLIP vs. CLIP + Adapter

### Robustness Curves

JSON results are visualized as:
- Accuracy vs. severity
- Ensemble improvement curves
- Adapter vs. baseline curves

### Model Comparison

The final step compares robustness performance across:
- **CLIP** (baseline)
- **SigLIP** (improved stability)
- **CoOp** (perfect robustness)
- **CLIP + Adapter** (noise-aware fine-tuning)

---



## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/AishahALtamimi/Robustness-to-Prompt-Variations-in-VLMs-Evaluating-Noisy-Prompts-with-Ensembling-Strategies.git
     cd Robustness-to-Prompt-Variations-in-VLMs-Evaluating-Noisy-Prompts-with-Ensembling-Strategies

    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
    ```

3. **Datasets**
   This project evaluates the robustness of CLIP, SigLIP, and CoOp on five publicly available vision datasets.  
   Below is a brief description of each dataset:

   - **Oxford-IIIT Pets:** Cat and dog images covering 37 pet breeds.  
   - **Food-101:** Food images across 101 categories.  
   - **DTD (Describable Textures Dataset):** Texture patterns such as “bumpy” or “striped.”  
   - **EuroSAT:** Satellite images representing different land-use types.  
   - **Caltech-101:** General object images across 101 categories.

   Please download them manually using the links below:

   ```bash
     Oxford Pets:
     https://www.robots.ox.ac.uk/~vgg/data/pets/
     
     Food101:
     https://www.vision.ee.ethz.ch/datasets_extra/food-101/
     
     DTD:
     https://www.robots.ox.ac.uk/~vgg/data/dtd/
     
     EuroSAT:
     https://github.com/phelber/EuroSAT
     
     Caltech-101:
     https://data.caltech.edu/records/20086
    ```
   
3. **Evaluate CLIP (Clean or Noisy Prompts):**
 ```bash
python prompt_noise_eval_v4.py \
  --dataset_name oxford_pets \
  --dataset_dir path/to/data \
  --severity_list 0,1,2,3 \
  --prompt_noises typo,case,space,emoji \
  --ensemble_k 5 \
  --include_clean True
```


4. **Evaluate SigLIP:**
```bash
python prompt_noise_eval_siglip.py \
  --dataset_name oxford_pets \
  --dataset_dir path/to/data \
  --severity_list 0,1,2,3 \
  --prompt_noises typo,case,space,emoji \
  --ensemble_k 5
```
6. **Evaluate CoOp Under Noisy Prompts:**
 ```bash
python coop_noise_eval_min_v5.py \
  --dataset_name oxford_pets \
  --dataset_dir path/to/data \
  --severity_list 0,1,2,3 \
  --prompt_noises typo,case,space,emoji \
  --ensemble_k 1
```
7. **Train the Noise-Aware Adapter for CLIP:**
 ```bash
   python train_prompt_noise_adapter.py \
  --dataset_name oxford_pets \
  --dataset_dir path/to/data \
  --epochs 10 \
  --severity_list 1,2,3 \
  --prompt_noises typo,case,space,emoji
```
8. **Evaluate CLIP + Adapter (After Training):**
 ```bash
   python noise_eval.py \
  --dataset_name oxford_pets \
  --dataset_dir path/to/data \
  --adapter_path output/text_adapter/adapter_last.pth \
  --severity_list 0,1,2,3 \
  --prompt_noises typo,case,space,emoji
```

This show how to run each model with oxford_pets dataset. For other datasets only change the dataset name
  

## Acknowledgments
- **Open-Source Communities:** We acknowledge the contributions of the PyTorch, Hugging Face, and OpenCLIP communities whose tools enabled this work.
- **CoOp Implementation:** This project makes use of the official CoOp (Context Optimization) implementation released by the original authors.
- **Individuals:** We extend our gratitude to all individuals who provided guidance, support, and constructive feedback throughout the project.
- **Resource Providers:** This research was conducted using Google Colab and GPU resources, which provided the necessary computing environment for training and evaluation.


