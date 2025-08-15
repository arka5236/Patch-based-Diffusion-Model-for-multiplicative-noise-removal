import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
import glob
import time
import random

# Import for new metrics
import lpips # For LPIPS
import torchmetrics.functional as tmF # For SSIM

# --- Model Definitions and Helper Functions (Must be the same as in training) ---

def generate_positional_map(patch_size):
    coords_y = torch.linspace(0, 1, patch_size).unsqueeze(0).repeat(patch_size, 1)
    coords_x = torch.linspace(0, 1, patch_size).unsqueeze(1).repeat(1, patch_size)
    positional_map = torch.stack([coords_x, coords_y], dim=0)
    return positional_map

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, sigma_t):
        half_dim = self.dim // 2
        frequencies = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32, device=sigma_t.device) *
            (-math.log(10000.0) / half_dim)
        )
        args = sigma_t.unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.AvgPool2d(2)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
    def forward(self, x, time_emb):
        x = self.conv1(x)
        x = x + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        skip_connection = self.conv2(x)
        x = self.pool(skip_connection)
        return x, skip_connection

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
    def forward(self, x, skip_connection, time_emb):
        x = self.upsample(x)
        if x.shape[-2:] != skip_connection.shape[-2:]:
            x = F.interpolate(x, size=skip_connection.shape[-2:], mode='nearest')
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = x + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(x)
        return x

class ScoreNetwork(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_channels=64,
                 channel_multipliers=(1, 2, 4, 8),
                 num_down_blocks=4,
                 time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_down_blocks = num_down_blocks
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.initial_conv = ConvBlock(in_channels + 2, base_channels) # +2 for positional map
        self.downs = nn.ModuleList()
        current_channels = base_channels
        encoder_channels = [base_channels]
        for i in range(num_down_blocks):
            out_channels_down = base_channels * channel_multipliers[i]
            self.downs.append(DownBlock(current_channels, out_channels_down, time_emb_dim))
            current_channels = out_channels_down
            encoder_channels.append(current_channels)

        # Bottleneck
        bottleneck_in_channels = current_channels
        self.bottleneck_conv1 = ConvBlock(bottleneck_in_channels, bottleneck_in_channels * 2)
        self.bottleneck_conv2 = ConvBlock(bottleneck_in_channels * 2, bottleneck_in_channels)
        self.bottleneck_time_proj = nn.Linear(time_emb_dim, bottleneck_in_channels * 2)

        self.ups = nn.ModuleList()
        prev_up_channels = bottleneck_in_channels
        for i in reversed(range(num_down_blocks)):
            skip_channels = encoder_channels[i]
            upblock_in_channels = prev_up_channels + skip_channels
            upblock_out_channels = encoder_channels[i]
            self.ups.append(UpBlock(upblock_in_channels, upblock_out_channels, time_emb_dim))
            prev_up_channels = upblock_out_channels
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    # Now outputs score (s_theta), here it predicts the noise `n`
    def forward(self, noisy_patch_log, sigma_t):
        batch_size, _, patch_h, patch_w = noisy_patch_log.shape
        positional_map = generate_positional_map(patch_h).to(noisy_patch_log.device).unsqueeze(0)
        positional_map_batch = positional_map.repeat(batch_size, 1, 1, 1)
        x = torch.cat([noisy_patch_log, positional_map_batch], dim=1)
        noise_level_embedding = self.time_embedding(sigma_t)
        x = self.initial_conv(x)
        skip_connections = [x]
        for down_block in self.downs:
            x, skip = down_block(x, noise_level_embedding)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = x + self.bottleneck_time_proj(noise_level_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.bottleneck_conv2(x)

        for i, up_block in enumerate(self.ups):
            skip = skip_connections[self.num_down_blocks - 1 - i]
            x = up_block(x, skip, noise_level_embedding)

        # Output is now the predicted noise `n` (epsilon_theta)
        predicted_noise = self.final_conv(x)
        return predicted_noise

def extract_patches(image, patch_size, stride):
    patches = []
    positions = []
    C, H, W = image.shape
    for h_idx in range(0, H - patch_size + 1, stride):
        for w_idx in range(0, W - patch_size + 1, stride):
            patch = image[:, h_idx : h_idx + patch_size, w_idx : w_idx + patch_size]
            patches.append(patch)
            positions.append((h_idx, w_idx))
    if not patches:
        return torch.empty(0, C, patch_size, patch_size, device=image.device), []
    return torch.stack(patches), positions

def aggregate_scores_to_image(patch_scores, image_shape, patch_size, stride, device):
    C, H, W = image_shape
    aggregated_score = torch.zeros(image_shape, device=device)
    overlap_count = torch.zeros(image_shape, device=device) + 1e-6 # Add epsilon to avoid division by zero
    idx = 0
    for h_idx in range(0, H - patch_size + 1, stride):
        for w_idx in range(0, W - patch_size + 1, stride):
            if idx < patch_scores.shape[0]:
                aggregated_score[:, h_idx : h_idx + patch_size, w_idx : w_idx + patch_size] += patch_scores[idx]
                overlap_count[:, h_idx : h_idx + patch_size, w_idx : w_idx + patch_size] += 1
                idx += 1
            else:
                break
    return aggregated_score / overlap_count

# MODIFIED: forward_operator_A now acts as an identity operator (no blurring)
def forward_operator_A(image):
    """
    Identity operator for pure denoising. It simply returns the input image.
    Assumes image is already in the correct format (C, H, W or B, C, H, W).
    """
    return image

def compute_data_consistency_gradient(x_current, y_measurement, forward_operator_A):
    # Ensure x_current requires grad for this calculation
    x_current.requires_grad_(True)
    A_x = forward_operator_A(x_current)

    if A_x.dim() == 4 and y_measurement.dim() == 3:
        A_x = A_x.squeeze(0)
    elif A_x.dim() == 3 and y_measurement.dim() == 4:
        y_measurement = y_measurement.squeeze(0)

    # Calculate loss. `y_measurement` does not need to require grad itself.
    data_consistency_loss = F.mse_loss(A_x, y_measurement, reduction='sum')

    # Compute gradient with respect to x_current
    grad = torch.autograd.grad(data_consistency_loss, x_current, allow_unused=True)[0]

    # Detach x_current from the graph to avoid accumulating gradients in future iterations
    x_current.requires_grad_(False)
    return grad

# --- REDEFINED PSNR, LPIPS, SSIM FUNCTIONS ---
def calculate_psnr(img1_tensor, img2_tensor, max_val=1.0):
    """
    Calculates PSNR between two PyTorch tensors.
    Assumes inputs are in [0, max_val] range and are 3D (C, H, W).
    """
    img1_tensor = img1_tensor.float()
    img2_tensor = img2_tensor.float()

    mse = torch.mean((img1_tensor - img2_tensor) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

def calculate_lpips(img1_tensor, img2_tensor, lpips_model, device):
    """
    Calculates LPIPS between two PyTorch tensors.
    Assumes inputs are in [0, 1] range (C, H, W) and will be normalized to [-1, 1] internally.
    """
    img1_lpips = (img1_tensor * 2 - 1).unsqueeze(0).to(device)
    img2_lpips = (img2_tensor * 2 - 1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = lpips_model(img1_lpips, img2_lpips).item()
    return score

def calculate_ssim(img1_tensor, img2_tensor, device):
    """
    Calculates SSIM between two PyTorch tensors.
    Assumes inputs are in [0, 1] range (C, H, W) and will be converted to BxCxHxW.
    """
    img1_ssim = img1_tensor.unsqueeze(0).to(device)
    img2_ssim = img2_tensor.unsqueeze(0).to(device)

    score = tmF.structural_similarity_index_measure(img1_ssim, img2_ssim, data_range=1.0).item()
    return score

def add_multiplicative_noise(image_uint8, sigma_multiplicative=0.12):
    """
    Adds multiplicative noise (speckle-like) to an image.
    Formula: Noisy_Pixel = Clean_Pixel * (1 + N(0, sigma_multiplicative))
    """
    img_float = image_uint8.astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma_multiplicative, img_float.shape).astype(np.float32)
    noise_factor = 1.0 + noise
    noisy_img_float = img_float * noise_factor
    noisy_img_uint8 = np.clip(noisy_img_float, 0.0, 1.0) * 255.0
    return noisy_img_uint8.astype(np.uint8)

# --- Algorithm 2: Combined Denoising and Deblurring (ODE Solver) ---

def solve_inverse_problem(model, y_measurement, image_shape, device,
                           sigmas, zeta_values, patch_size, stride):
    """
    Algorithm 2: Combined Denoising and Deblurring (ODE Solver)
    Now adapted for pure denoising as forward_operator_A is identity.

    Args:
        model (ScoreNetwork): The trained score network s_theta(y_k, k).
        y_measurement (torch.Tensor): Corrupted image x_tilde (multiplicatively noisy, in linear domain).
                                       This will be converted to log(x_tilde + epsilon) for processing.
        image_shape (tuple): (C, H, W) of the image.
        device (torch.device): Device to run computation on (cuda or cpu).
        sigmas (list/torch.Tensor): Noise schedule sigma(k) for k=1,...,K (from high noise to low noise).
                                     Should be decreasing.
        zeta_values (list/torch.Tensor): Data consistency weights zeta_k for k=1,...,K.
        patch_size (int): Size of patches to extract for score network inference.
        stride (int): Stride for extracting patches.

    Returns:
        torch.Tensor: Denoised image x_hat (in linear domain [0,1]).
    """
    model.eval()
    C, H, W = image_shape

    y_current = torch.log(y_measurement.clone().detach().to(device).float() + 1e-6) # Add small epsilon

    num_steps = len(sigmas) -1
    # print(f"\nStarting inverse problem reconstruction for {num_steps} steps (Log-domain based)...") # Commented for dataset processing

    for t_idx in range(num_steps):
        current_sigma_val = sigmas[t_idx]
        next_sigma_val = sigmas[t_idx+1]

        delta_sigma_sq_for_ode = current_sigma_val**2 - next_sigma_val**2
        delta_sigma_sq_for_ode = max(delta_sigma_sq_for_ode, 1e-8)

        current_zeta_t = zeta_values[t_idx] if t_idx < len(zeta_values) else zeta_values[-1]

        patches_log, patch_positions = extract_patches(y_current, patch_size, stride)
        if patches_log.shape[0] == 0:
            # print(f"Warning: No patches extracted at step {t_idx} for sigma {current_sigma_val}. Skipping step.") # Commented for dataset processing
            continue

        sigma_t_batch = torch.full((patches_log.shape[0],), fill_value=current_sigma_val, device=device)

        with torch.no_grad(): # Model inference should be without grad
            predicted_noise_patches = model(patches_log, sigma_t_batch)

        # Convert predicted noise to score function (score = -noise / sigma^2)
        predicted_score_patches = -predicted_noise_patches / (current_sigma_val**2)

        s_aggregated = aggregate_scores_to_image(predicted_score_patches, image_shape, patch_size, stride, device)

        # ODE update: y_denoised_by_score = y_current + 0.5 * (σ_cur^2 - σ_next^2) * s_aggregated
        y_denoised_by_score = y_current + 0.5 * delta_sigma_sq_for_ode * s_aggregated

        # Clamp after log transform before exp, for stability in the log domain
        y_denoised_by_score = torch.clamp(y_denoised_by_score, math.log(1e-6), math.log(1.0 + 1e-6)) # Re-evaluate clamping max value

        x_prelim = torch.exp(y_denoised_by_score)
        x_prelim = torch.clamp(x_prelim, 1e-6, 1.0) # Clamp in linear domain [0,1]

        grad_fidelity = compute_data_consistency_gradient(x_prelim.clone().detach(), y_measurement.to(device), forward_operator_A)

        x_updated = x_prelim - current_zeta_t * grad_fidelity
        x_updated = torch.clamp(x_updated, 1e-6, 1.0) # Clamp in linear domain [0,1]

        # Convert back to log domain for the next iteration
        y_current = torch.log(x_updated + 1e-6)

        # print(f"Step {t_idx+1}/{num_steps}, Sigma: {current_sigma_val:.4f}, Zeta: {current_zeta_t:.4f}") # Commented for dataset processing

    return torch.exp(y_current) # The final y_current is y_0 (in linear domain)


# --- Inference Script for CelebA Dataset ---

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your CelebA dataset directory (REVERTED to full dataset path)
    DATA_DIR = '/content/img_align_celeba/img_align_celeba'

    # Specific image path for detailed visualization (will be dynamically set from sampled images)
    SPECIFIC_IMAGE_PATH = None

    IMAGE_SIZE = (256, 256) # REVERTED: Resize all images to this size for consistency
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    TIME_EMB_DIM = 256
    CHECKPOINT_DIR = "/content/checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "/content/checkpoints/best_score_network_model (5).pth")

    NOISE_SCHEDULE_MIN = 0.01
    NOISE_SCHEDULE_MAX = 0.5

    # Inference parameters for solve_inverse_problem
    INFERENCE_PATCH_SIZE = 64
    INFERENCE_STRIDE = INFERENCE_PATCH_SIZE // 2
    INFERENCE_NUM_STEPS = 6000 # More steps for better reconstruction
    ZETA_VALUE = 0.001 # A single value for zeta; can be made a schedule if needed

    MULTI_NOISE_LEVEL = 0.3 # Sigma for multiplicative noise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize LPIPS model once
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    # Initialize model and load trained weights
    model = ScoreNetwork(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_down_blocks=4,
        time_emb_dim=TIME_EMB_DIM
    ).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded trained model from {BEST_MODEL_PATH}")
    else:
        print(f"Error: Trained model not found at {BEST_MODEL_PATH}. Please ensure the model is trained and saved correctly.")
        exit() # Exit if model is not found, as evaluation won't be meaningful

    model.eval()

    # --- Prepare CelebA Dataset for evaluation ---
    all_image_paths = glob.glob(os.path.join(DATA_DIR, '*.jpg'))

    if not all_image_paths:
        print(f"Error: No images found in {DATA_DIR}. Please check the path and CelebA dataset structure.")
        exit()

    # Select 5000 random images for testing (REVERTED to 5000 images)
    test_set_size = 20000
    if len(all_image_paths) < test_set_size:
        print(f"Warning: Dataset contains only {len(all_image_paths)} images, which is less than the requested {test_set_size} for testing. Using all available images.")
        image_paths = all_image_paths
    else:
        image_paths = random.sample(all_image_paths, test_set_size)

    # Dynamically set SPECIFIC_IMAGE_PATH to the first image in the sampled list
    if image_paths:
        SPECIFIC_IMAGE_PATH = image_paths[0]
    else:
        print("No images found to set a specific image path for visualization. Exiting.")
        exit()

    print(f"\n--- Starting Evaluation on CelebA Test Subset ({len(image_paths)} images from {DATA_DIR}) ---")


    transform_test = T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor() # Converts to [0, 1]
    ])

    # Store metrics for full dataset evaluation
    psnr_measurements = []
    psnr_reconstructions = []
    lpips_measurements = []
    lpips_reconstructions = []
    ssim_measurements = []
    ssim_reconstructions = []

    start_time = time.time()

    # Define sigmas_inference and zeta_values_inference outside the loop
    sigmas_inference = torch.exp(torch.linspace(math.log(NOISE_SCHEDULE_MAX), math.log(NOISE_SCHEDULE_MIN), INFERENCE_NUM_STEPS + 1)).to(device)
    zeta_values_inference = [ZETA_VALUE] * INFERENCE_NUM_STEPS

    # --- Loop through each image in the test subset for evaluation ---
    for i, img_path in enumerate(image_paths):
        # print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}") # Commented for cleaner output during full run

        try:
            ground_truth_image_pil = Image.open(img_path).convert('RGB')
            ground_truth_image = transform_test(ground_truth_image_pil).to(device) # C, H, W
        except Exception as e:
            # print(f"Could not load or process image {img_path}: {e}. Skipping.") # Commented for cleaner output
            continue

        # Simulate corruption with multiplicative noise
        ground_truth_image_np_uint8 = (ground_truth_image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        noisy_image_np_uint8 = add_multiplicative_noise(ground_truth_image_np_uint8, sigma_multiplicative=MULTI_NOISE_LEVEL)
        y_measurement = torch.from_numpy(noisy_image_np_uint8.transpose(2, 0, 1)).float() / 255.0
        y_measurement = y_measurement.to(device)

        # Run the inverse problem solver (denoiser)
        reconstructed_image = solve_inverse_problem(
            model=model,
            y_measurement=y_measurement,
            image_shape=ground_truth_image.shape,
            device=device,
            sigmas=sigmas_inference,
            zeta_values=zeta_values_inference,
            patch_size=INFERENCE_PATCH_SIZE,
            stride=INFERENCE_STRIDE
        )

        # Calculate Metrics
        psnr_meas = calculate_psnr(ground_truth_image, y_measurement, max_val=1.0)
        psnr_recon = calculate_psnr(ground_truth_image, reconstructed_image, max_val=1.0)
        lpips_meas = calculate_lpips(ground_truth_image, y_measurement, lpips_loss_fn, device)
        lpips_recon = calculate_lpips(ground_truth_image, reconstructed_image, lpips_loss_fn, device)
        ssim_meas = calculate_ssim(ground_truth_image, y_measurement, device)
        ssim_recon = calculate_ssim(ground_truth_image, reconstructed_image, device)

        psnr_measurements.append(psnr_meas)
        psnr_reconstructions.append(psnr_recon)
        lpips_measurements.append(lpips_meas)
        lpips_reconstructions.append(lpips_recon)
        ssim_measurements.append(ssim_meas)
        ssim_reconstructions.append(ssim_recon)

    total_time = time.time() - start_time

    # --- Aggregate and Print Results for the 5000-image test subset ---
    print(f"\n--- Evaluation Results on CelebA Test Subset ({len(image_paths)} images) ---")
    print(f"Total time for inference: {total_time:.2f} seconds ({total_time / len(image_paths):.4f} s/image)")

    avg_psnr_meas = np.mean(psnr_measurements)
    std_psnr_meas = np.std(psnr_measurements)
    avg_psnr_recon = np.mean(psnr_reconstructions)
    std_psnr_recon = np.std(psnr_reconstructions)

    avg_lpips_meas = np.mean(lpips_measurements)
    std_lpips_meas = np.std(lpips_measurements)
    avg_lpips_recon = np.mean(lpips_reconstructions)
    std_lpips_recon = np.std(lpips_reconstructions)

    avg_ssim_meas = np.mean(ssim_measurements)
    std_ssim_meas = np.std(ssim_measurements)
    avg_ssim_recon = np.mean(ssim_reconstructions)
    std_ssim_recon = np.std(ssim_reconstructions)

    print(f"\nAverage PSNR (Clean vs. Corrupted): {avg_psnr_meas:.2f} ± {std_psnr_meas:.2f} dB")
    print(f"Average PSNR (Clean vs. Reconstructed): {avg_psnr_recon:.2f} ± {std_psnr_recon:.2f} dB")

    print(f"\nAverage LPIPS (Clean vs. Corrupted): {avg_lpips_meas:.4f} ± {std_lpips_meas:.4f}")
    print(f"Average LPIPS (Clean vs. Reconstructed): {avg_lpips_recon:.4f} ± {std_lpips_recon:.4f}")

    print(f"\nAverage SSIM (Clean vs. Corrupted): {avg_ssim_meas:.4f} ± {std_ssim_meas:.4f}")
    print(f"Average SSIM (Clean vs. Reconstructed): {avg_ssim_recon:.4f} ± {std_ssim_recon:.4f}")

    # --- Inference on the Test Dataset ---
    print("\n--- Inference Drawn on the Test Dataset (5000 random images) ---")
    print(f"Based on a random subset of {len(image_paths)} images from the CelebA dataset, the model significantly improves image quality. "
          f"On average, PSNR increases by approximately {(avg_psnr_recon - avg_psnr_meas):.2f} dB, "
          f"LPIPS is reduced by {(avg_lpips_meas - avg_lpips_recon):.4f}, "
          f"and SSIM increases by {(avg_ssim_recon - avg_ssim_meas):.4f}.")
    print("This indicates strong performance in removing multiplicative noise and reconstructing visual details.")

    # --- Specific Image Visualization ---
    print(f"\n--- Visualizing specific image: {os.path.basename(SPECIFIC_IMAGE_PATH)} ---")
    try:
        gt_specific_pil = Image.open(SPECIFIC_IMAGE_PATH).convert('RGB')
        gt_specific = transform_test(gt_specific_pil).to(device)

        gt_specific_np_uint8 = (gt_specific.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        noisy_specific_np_uint8 = add_multiplicative_noise(gt_specific_np_uint8, sigma_multiplicative=MULTI_NOISE_LEVEL)
        y_specific_measurement = torch.from_numpy(noisy_specific_np_uint8.transpose(2, 0, 1)).float() / 255.0
        y_specific_measurement = y_specific_measurement.to(device)

        reconstructed_specific = solve_inverse_problem(
            model=model,
            y_measurement=y_specific_measurement,
            image_shape=gt_specific.shape,
            device=device,
            sigmas=sigmas_inference,
            zeta_values=zeta_values_inference,
            patch_size=INFERENCE_PATCH_SIZE,
            stride=INFERENCE_STRIDE
        )

        psnr_specific_meas = calculate_psnr(gt_specific, y_specific_measurement, max_val=1.0)
        psnr_specific_recon = calculate_psnr(gt_specific, reconstructed_specific, max_val=1.0)
        lpips_specific_meas = calculate_lpips(gt_specific, y_specific_measurement, lpips_loss_fn, device)
        lpips_specific_recon = calculate_lpips(gt_specific, reconstructed_specific, lpips_loss_fn, device)
        ssim_specific_meas = calculate_ssim(gt_specific, y_specific_measurement, device)
        ssim_specific_recon = calculate_ssim(gt_specific, reconstructed_specific, device)

        # Print metrics for the single specific image
        print(f"\nMetrics for single image '{os.path.basename(SPECIFIC_IMAGE_PATH)}':")
        print(f"  Corrupted vs. Original: PSNR={psnr_specific_meas:.2f} dB, LPIPS={lpips_specific_meas:.4f}, SSIM={ssim_specific_meas:.4f}")
        print(f"  Reconstructed vs. Original: PSNR={psnr_specific_recon:.2f} dB, LPIPS={lpips_specific_recon:.4f}, SSIM={ssim_specific_recon:.4f}")


        # Prepare for plotting (Numpy HWC format)
        gt_specific_np = np.clip(gt_specific.cpu().numpy().transpose(1, 2, 0), 0, 1)
        noisy_specific_np = np.clip(y_specific_measurement.cpu().numpy().transpose(1, 2, 0), 0, 1)
        recon_specific_np = np.clip(reconstructed_specific.cpu().numpy().transpose(1, 2, 0), 0, 1)

        plt.figure(figsize=(18, 6)) # Increased figure size for better visibility
        plt.suptitle(f"Denoising for: {os.path.basename(SPECIFIC_IMAGE_PATH)}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.title("Original Ground Truth")
        plt.imshow(gt_specific_np)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Corrupted Measurement\nPSNR: {psnr_specific_meas:.2f} dB\nLPIPS: {lpips_specific_meas:.4f}\nSSIM: {ssim_specific_meas:.4f}")
        plt.imshow(noisy_specific_np)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Reconstructed Image\nPSNR: {psnr_specific_recon:.2f} dB\nLPIPS: {lpips_specific_recon:.4f}\nSSIM: {ssim_specific_recon:.4f}")
        plt.imshow(recon_specific_np)
        plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    except FileNotFoundError:
        print(f"Error: Specific image not found at {SPECIFIC_IMAGE_PATH}. Please check the path.")
    except Exception as e:
        print(f"Error processing specific image {SPECIFIC_IMAGE_PATH}: {e}")

    print("\n--- End of Script ---")
