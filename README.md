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

**Brief List of Course Topics**

- Computational efficiency and scaling laws  
- Model profiling using PyTorch Profiler  
- Automated Machine Learning (AutoML)  
- Neural Architecture Search (NAS), including differentiable search  
- Overview of high-level (vLLM, SGLang, Ollama) and low-level (CUTLASS, CuTile) libraries  
- CPU/GPU architecture, memory hierarchy, GPU arithmetic  
- Introduction to Triton and custom kernel development  
- PyTorch 2.0: JIT tracing, torch.compile, ONNX conversion  
- Model pruning: structured/unstructured, iterative, magnitude-based  
- Fundamentals of quantization: methods (e.g., LSQ), low-precision data types  
- LLM-specific compression techniques  
- LLM inference optimization: KV-Cache, PagedAttention, Gradient Checkpointing  
- Methods for searching and optimizing GPU kernels
