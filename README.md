# Patch-based-Diffusion-Model-for-multiplicative-noise-removal

# Abstract


Image restoration, particularly deblurring and denoising, remains a critical challenge in com-
puter vision. This paper presents the Patch Diffusion Inverse Solver for Multiplicative Noise
(PaDIS-MN), a novel approach leveraging score-based generative models to address inverse prob-
lems involving multiplicative noise. Unlike traditional methods that often struggle with complex
noise distributions, our method operates in the logarithmic domain, effectively transforming mul-
tiplicative noise into additive noise, which is amenable to diffusion models. PaDIS-MN employs a
U-Net based Score Network trained to predict noise in image patches. During inference, an itera-
tive Ordinary Differential Equation (ODE) solver combines the denoising capabilities of the score
model with a data consistency term, ensuring fidelity to the observed corrupted measurement. We
demonstrate the efficacy of PaDIS-MN on a standard image dataset, showcasing significant improve-
ments in image quality (PSNR, SSIM, LPIPS) compared to the corrupted inputs. Our patch-based
approach enables efficient processing of high-resolution images, making PaDIS-MN a robust solution
for real-world image degradation.


# Citation

1) Hu, J. (2024). *PaDIS: Patch-based Diffusion Models* [Source code]. GitHub. Retrieved from https://github.com/jasonhu4/PaDIS

2) Vuong, A. (2025). *sde_multiplicative_noise_removal* [Source code]. GitHub. Retrieved from https://github.com/anvuongb/sde_multiplicative_noise_removal

# Acknowledgement

We'd like to acknowledge the following open-source projects which have been instrumental in this work:

PaDIS: Patch-based Diffusion Models
We thank Jason Hu for developing and maintaining the PaDIS repository. This project's work on learning image priors through patch-based diffusion models was highly insightful.
Hu, J. (2025). PaDIS: Patch-based Diffusion Models [Source code]. GitHub. Retrieved from https://github.com/jasonhu4/PaDIS

SDE Multiplicative Noise Removal
We also extend our gratitude to An Vuong for the sde_multiplicative_noise_removal repository, which provided valuable insights into stochastic differential equations for noise removal.
Vuong, A. (2025). sde_multiplicative_noise_removal [Source code]. GitHub. Retrieved from https://github.com/anvuongb/sde_multiplicative_noise_removal
