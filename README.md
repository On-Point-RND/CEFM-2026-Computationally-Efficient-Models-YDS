# Computationally-Efficient-Models-YDS-2026 Course Description

This course focuses on computational efficiency in modern machine learning, particularly in the context of large language models (LLMs). It covers optimization techniques that take hardware characteristics into account, including GPU kernel programming, quantization, pruning, model compilation, as well as automated architecture search and hyperparameter tuning. 
**Course Timeline**

- **MIDTERM**: The midterm portion of the project and Homework 1 must be submitted with at least 5 points earned.

**Projects and Assignments**

1. **Team Project (10 points)** – Teams of up to 3 students  
   - Example projects from  2025 : https://github.com/On-Point-RND/Efficient-Models-course-ITMO-2025?tab=readme-ov-file#projects-from-the-2025-itmo-course-by-category  

   - **Midterm Deliverable (due by April 1):**  
     (1) Team formed  
     (2) Code for at least one experiment implemented and submitted  
     (3) A PDF report containing:  
         - Plan for all experiments  
         - Weekly team work schedule  
         - Description and results of the first experiment  

   - **Final Deliverable (due by April 25) + In-person defense on April 27:**  
     - Full implementation of assigned tasks  
     - GitHub repository containing code, instructions, results description, and a final presentation (PDF)

2. **Homework Assignments**  
   - **HW1**: Triton + Kernel puzzles (March 9 – April 1)  
   - **HW2**: Triton + Quantization (March 9 – April 20)  

   Both homework assignments contribute to the final grade.

**Grading System**

- Maximum score: **30 points**  
  - **18–20 points** → Pass  
  - **21–25 points** → Good  
  - **26–30 points** → Excellent  

---


# SHAD 2026 Project Categories:

> (1) Model Quantization: Triton and CUDA Kernels  
(2) U-Net Inference Acceleration  
(2) Speculative Decoding with a Quantized Draft Model  
(3) Whisper Acceleration on CPU  
(4) LLM / VLM Profiling and Energy Consumption  
(5) Quantization-Aware Training (QAT) and CPU Evaluation  
(6) Running Models on Mobile Devices  
(7) Binary Neural Networks  
(8) Efficient Training: Sparse Gradients  
(9) Efficient Training: Data Selection  
(10) Neural Architecture Search (NAS) and Knowledge Distillation

## Model Quantization: Triton and CUDA Kernels

- **Triton kernels for LLM weight quantization and quantized model inference**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1mfkmonvGEsfjXr_rkqNkm9gTMF7HI90A9P5LkmmfVnA/edit?usp=sharing) | [GitHub](https://github.com/innaanikina/efficientml)

(Team 2)  [Slides](https://github.com/Alex-Andrv/project/blob/main/SLIDES.pdf) | [GitHub](https://github.com/Alex-Andrv/project)

(Team 3)  [Slides](https://docs.google.com/presentation/d/1AXwUdJ3VzjGWMC2ITtyaG5tKCLEod4XqI-Xup0gHVso/edit?slide=id.p#slide=id.p) | [GitHub](https://github.com/kirillTerra/efdl_project)

- **CUDA kernels for LLM weight quantization and quantized model inference**

[Slides](https://drive.google.com/file/d/1TClPWkx-jNh0wnXHkAR1_Z8awxccAOtK/view?usp=sharing) | [GitHub](https://github.com/meawing/eml_project/tree/main)

## U-Net Inference Acceleration

- **Accelerating UNet model inference**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1yZugXE3OPq5qCX6boNjz4uBIv-Jp4hBALTQdABy2UCU/edit?usp=sharing) | [GitHub](https://github.com/WoodieDudy/unet-infer-optimization)

(Team 2)  [GitHub](https://github.com/Sheshkin/efml-project/tree/first_step)

(Team 3)  [Slides](https://github.com/Tyumentseva/Unet-Inference-Acceleration-Project/blob/main/unet_inference.pdf) | [GitHub](https://github.com/Tyumentseva/Unet-Inference-Acceleration-Project)

(Team 4)  [Slides](https://github.com/RomanLomovsky/unet-acceleration/blob/main/EM_project_presentation.pdf) | [GitHub](https://github.com/RomanLomovsky/unet-acceleration)

(Team 5)  [GitHub](https://github.com/aapetukhov/efml-unet)

(Team 6)  [Slides](https://docs.google.com/presentation/d/1fGGqB5Tlv-S8g-rp0C9NRROQxqN-Cq_OXmu17WEQbKg/edit?usp=sharing) | [GitHub](https://github.com/legaliza-bit/efficient-unet-inference/tree/main)

(Team 7)  [Slides](https://docs.google.com/presentation/d/1Y9mMaGFmB8NIl2ze7fqp870O9UyAZXYw5UxNUhFN25o/edit?usp=sharing) | [GitHub](https://github.com/algorithm-pirogok/effml_unet_check)

## Speculative Decoding

- **Speculative decoding with a quantized draft model**

(Team 1)  [Slides](https://drive.google.com/file/d/1n80gSGUJo0IWNOyV82XbQscBYYVVKNnn/view?usp=drivesdk) | [GitHub](https://github.com/Aidaricus/quant_spec/tree/main)

(Team 2)  [Slides](https://docs.google.com/presentation/d/19xBl1eQOfPMJvOmnK7wt3ZynREdvCiI0MTUHZLXkZlA/edit?usp=sharing) | [GitHub](https://github.com/olevanss/eff_ml_project)

## Whisper Acceleration on CPU

- **Whisper CPU — real-time inference**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1DotOLgRRlS1MpjIzYnlIg6ZG46rW53b9/edit?usp=sharing&ouid=115238071965921588071&rtpof=true&sd=true) | [GitHub](https://github.com/Anuiel/asr-cpu)

(Team 2)  [Slides](https://docs.google.com/presentation/d/1vZhzR3M3B3xBN9qK3ZIaAebtOtZVpyrZKABKWc6dkv8/edit?usp=sharing) | [GitHub](https://github.com/MichaelNotDeveloper/WhisperRT-CPU/tree/main)

## LLM / VLM Profiling and Energy Consumption

- **Profiling LLM and VLLM (with energy consumption)**

(Team 1)  [GitHub](https://github.com/smirnovlad/vlm-profiler)

(Team 2)  [Slides](https://docs.google.com/presentation/d/1-ZLqcAVyG2ZDxtzJehix48sNRSV5GzCd5dN31-2cr9g/edit?slide=id.p1#slide=id.p1) | [GitHub](https://github.com/ars200200/ml_efficiency_course_work)

## Quantization-Aware Training (QAT) and CPU Evaluation

- **QAT with Int8 conversion and quality evaluation on CPU**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1m50eLYrmPDU2MokgCwb9EC_c5Tly_Vj90w5Q_HQgE5k/edit?usp=sharing) | [GitHub](https://github.com/Snikem/SASRec.pytorch)

(Team 2)  [Slides](https://docs.google.com/presentation/d/13CvfhOKczxETNucPRGnJokrquSH9eoxiV5OCc4FGxAc/edit?slide=id.p#slide=id.p) | [GitHub](https://github.com/Matewl/EFF_DL_QAT)

## Running Models on Mobile Devices

- **Running a quantized Int8 TTS model on a phone**

[GitHub](https://github.com/mievst/Mobile-TTS-YDS)

## Binary Neural Networks

- **Training a model with binary weights for keyword spotting (voice command classification)**

(Team 1)  [Slides](https://github.com/maya-belozer/YSDA_EMML_Bin/blob/main/Binary_Speech_Commands.pdf) | [GitHub](https://github.com/maya-belozer/YSDA_EMML_Bin)

(Team 2)  [GitHub](https://github.com/seniorfroggy/1Bit-ResNet18-KWS)

## Efficient Training: Sparse Gradients

- **Training transformers with sparse gradients (nanoGPT + long context)**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1KVn6UtWpgZ49kny-S4lrSfx_QLI-JYuid-AozL4yaJ0/edit?usp=sharing) | [GitHub](https://github.com/BurmistrovaEO/sparse-activations-MOC-llm)

(Team 2)  [Slides](https://www.overleaf.com/read/pfkhjzswyhgx#81a1fa) | [GitHub](https://github.com/letit6E/sparse-gpt-research)

(Team 3)  [Slides](https://docs.google.com/presentation/d/1j-NOFZOYtzWDeAGK-UWUWFdKENZycNXeY2QN1ZQ4kG0/edit?usp=sharing) | [GitHub](https://github.com/kostyayatsok/ysda-efficient-models-project-mixture-of-channels)

## Efficient Training: Data Selection

- **Data selection methods based on geometric properties**

(Team 1)  [Slides](https://docs.google.com/presentation/d/1mp0BITkbBOY_YmWiQ69Yt_gDdtEPzlT3/edit?usp=sharing&ouid=107187012210002792933&rtpof=true&sd=true) | [GitHub](https://github.com/grvlko/data-selection-with-geometric-methods)

(Team 2)  [Slides](https://github.com/khilling2/Eff_ML_project/blob/main/%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82_%D1%8D%D1%84%D1%84__ML.pdf) | [GitHub](https://github.com/khilling2/Eff_ML_project)

## Neural Architecture Search (NAS) and Knowledge Distillation

- **NAS and knowledge distillation**

[Slides](https://github.com/binary-wolfishness/nas-kd/blob/nas-evolution-kd-hinton/docs/NAS_KD.pdf) | [GitHub](https://github.com/binary-wolfishness/nas-kd/tree/nas-evolution-kd-hinton)
