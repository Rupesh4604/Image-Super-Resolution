# **Project Title:** Deep Learning-Based 4× Image Super-Resolution for Gaming Imagery

### **Abstract**

This project investigates the application of deep learning techniques for single-image super-resolution (SISR) in the context of gaming imagery. The primary objective is to reconstruct high-resolution (HR) images from corresponding low-resolution (LR) inputs using a fixed upscaling factor of **4×**. Emphasis is placed on restoring fine-grained details, enhancing perceptual quality, and preserving semantic consistency in domain-specific gaming scenes. The model was developed and evaluated as part of a competitive machine learning challenge hosted on Kaggle.

Github link: https://github.com/Rupesh4604/Image-Super-Resolution/
---

### **1. Dataset Description**

The dataset was obtained from the [GNR 638 Super-Resolution Challenge on Kaggle](https://www.kaggle.com/competitions/gnr638/data), comprising:

* **Training Set:**

  * 4,500 LR images of resolution 480×270 pixels
  * 4,500 corresponding HR images of resolution 1920×1080 pixels
* **Test Set:**

  * 500 LR images (480×270 pixels) without ground truth labels

The total dataset size is approximately **16 GB**, and all images are derived from gaming environments, offering a domain-specific benchmark for evaluating perceptual image enhancement algorithms.

---

### **2. Methodology**

A **Lightweight Convolutional Neural Network (CNN)** architecture was designed and implemented in PyTorch to perform 4× super-resolution. The architecture consists of the following key components:

* **Initial Feature Extraction:**
  A 3×3 convolutional layer followed by ReLU activation to extract shallow features from the input LR image.

* **Residual Learning Module:**
  A sequence of **eight residual blocks**, each with internal skip connections, enabling the model to learn high-frequency textures while mitigating the vanishing gradient problem.

* **Global Residual Connection:**
  A convolutional layer applied post-residual block stack is added back to the initial feature representation, improving gradient flow and model stability.

* **Upsampling Stage:**
  Two **PixelShuffle-based** sub-pixel convolution layers are employed to achieve efficient and accurate 4× spatial resolution enhancement.

* **Reconstruction Layer:**
  A final 3×3 convolution layer reconstructs the HR RGB image from the upsampled feature maps.

A custom PyTorch `Dataset` class was implemented for efficient data loading, patch-based training, and augmentation (including random cropping, flipping, and dynamic resizing for robustness).

---

### **3. Training Configuration**

* **Loss Function:** Mean Absolute Error (L1 Loss), chosen for its ability to preserve sharp image structures.
* **Optimizer:** Adam
* **Learning Rate Scheduler:** Cosine Annealing Learning Rate (LR) scheduler
* **Precision Mode:** Mixed-precision training using PyTorch AMP (Automatic Mixed Precision)
* **Best Model Selection:** Based on validation PSNR

Training was conducted using paired LR-HR image patches with batch-based optimization and real-time loss monitoring via the `tqdm` interface.

---

### **4. Evaluation Metrics**

The trained model was evaluated using both traditional and perceptual metrics:

* **PSNR (Peak Signal-to-Noise Ratio):** Measures reconstruction fidelity relative to the ground truth.
* **SSIM (Structural Similarity Index):** Quantifies perceptual similarity in terms of luminance, contrast, and structural patterns.
* **Composite Metric:**
  A joint score defined by the competition as:

  $$
  \text{Score} = 40 \times \text{SSIM} + \text{PSNR}
  $$

  This metric emphasizes perceptual quality while maintaining pixel-level accuracy.

---

### **5. Results**

The proposed CNN-based architecture demonstrated effective recovery of structural details and fine textures from low-resolution gaming imagery. Qualitative visualizations and quantitative results confirmed the model's ability to outperform baseline interpolation methods (e.g., bicubic) in both fidelity and perceptual quality.

---

### **6. Tools and Frameworks**

* **Programming Language:** Python
* **Frameworks and Libraries:** PyTorch, OpenCV, NumPy, scikit-image, tqdm
* **Development Environment:** Jupyter Notebook with CUDA-accelerated GPU support

---

### **7. Future Work**

* **Transformer-based Super-Resolution:**
  Investigate the integration of Vision Transformers (e.g., ViT, SwinIR) to model global dependencies in gaming images.

* **Adversarial and Perceptual Learning:**
  Incorporate perceptual loss functions (e.g., VGG-based) and adversarial training (e.g., SRGAN) to further enhance visual realism.

* **Real-Time Deployment:**
  Optimize model inference speed and memory footprint for deployment in real-time applications such as gaming engines or interactive streaming platforms.
