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

def calculate_psnr(img1, img2, max_val=1.0):
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach()

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

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

    y_current = torch.log(y_measurement.clone().detach().to(device) + 1e-6) # Add small epsilon

    num_steps = len(sigmas) -1
    print(f"\nStarting inverse problem reconstruction for {num_steps} steps (Log-domain based)...")

    for t_idx in range(num_steps):
        current_sigma_val = sigmas[t_idx]
        next_sigma_val = sigmas[t_idx+1]

        delta_sigma_sq_for_ode = current_sigma_val**2 - next_sigma_val**2
        delta_sigma_sq_for_ode = max(delta_sigma_sq_for_ode, 1e-8)

        current_zeta_t = zeta_values[t_idx] if t_idx < len(zeta_values) else zeta_values[-1]

        patches_log, patch_positions = extract_patches(y_current, patch_size, stride)
        if patches_log.shape[0] == 0:
            print(f"Warning: No patches extracted at step {t_idx} for sigma {current_sigma_val}. Skipping step.")
            continue

        sigma_t_batch = torch.full((patches_log.shape[0],), fill_value=current_sigma_val, device=device)

        with torch.no_grad(): # Model inference should be without grad
            predicted_noise_patches = model(patches_log, sigma_t_batch)

        # Convert predicted noise to score function (score = -noise / sigma^2)
        predicted_score_patches = -predicted_noise_patches / (current_sigma_val**2)


        s_aggregated = aggregate_scores_to_image(predicted_score_patches, image_shape, patch_size, stride, device)

        # ODE update: y_denoised_by_score = y_current + 0.5 * (σ_cur^2 - σ_next^2) * s_aggregated
        y_denoised_by_score = y_current + 0.5 * delta_sigma_sq_for_ode * s_aggregated

        y_denoised_by_score = torch.clamp(y_denoised_by_score, math.log(1e-6), math.log(1.0))

        x_prelim = torch.exp(y_denoised_by_score)
        x_prelim = torch.clamp(x_prelim, 1e-6, 1.0)

        # compute_data_consistency_gradient should correctly track gradients for x_prelim
        # Since forward_operator_A is now identity, this is equivalent to
        # grad_fidelity = torch.autograd.grad(F.mse_loss(x_prelim, y_measurement, reduction='sum'), x_prelim)[0]
        grad_fidelity = compute_data_consistency_gradient(x_prelim.clone().detach(), y_measurement.to(device), forward_operator_A)

        x_updated = x_prelim - current_zeta_t * grad_fidelity
        x_updated = torch.clamp(x_updated, 1e-6, 1.0)

        y_current = torch.log(x_updated + 1e-6)

        print(f"Step {t_idx+1}/{num_steps}, Sigma: {current_sigma_val:.4f}, Zeta: {current_zeta_t:.4f}")

    return torch.exp(y_current) # The final y_current is y_0


# --- Inference Script ---

if __name__ == "__main__":
    DATA_DIR = '/content/img_align_celeba/img_align_celeba' # CHANGE THIS TO YOUR IMAGE DIRECTORY
    IMAGE_SIZE = (256, 256)
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    TIME_EMB_DIM = 256
    CHECKPOINT_DIR = "/content/checkpoints"
    # Ensure BEST_MODEL_PATH is correct relative to the current working directory or an absolute path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "/content/checkpoints/best_score_network_model (1).pth")

    NOISE_SCHEDULE_MIN = 0.01
    NOISE_SCHEDULE_MAX = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        print(f"Error: Trained model not found at {BEST_MODEL_PATH}. Please run the training script first.")
        # Create a dummy model for demonstration if not found
        print("Creating a dummy model for demonstration (will not perform well without training)...")
        # In a real scenario, you'd exit or ensure model is trained.
        # For demonstration, we'll proceed with an untrained model.
        # This block is for robustness in a demo environment, not for production.

    model.eval()

    # Find a test image
    all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True) + \
                     glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True) + \
                     glob.glob(os.path.join(DATA_DIR, '**', '*.jpeg'), recursive=True)

    if not all_image_paths:
        print(f"\nFATAL ERROR: No images found in {DATA_DIR} or its subdirectories. Cannot run inference.")
        print("Creating dummy images for demonstration...")
        dummy_dir = os.path.join(DATA_DIR, "dummy_inference_images")
        os.makedirs(dummy_dir, exist_ok=True)
        for i in range(1): # Create at least one dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8))
            dummy_image.save(os.path.join(dummy_dir, f"dummy_inference_image_{i}.png"))
        all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True) # Re-glob after creating dummy
        if not all_image_paths:
            print("Failed to create and find dummy images. Exiting.")
            exit()

    test_image_path = all_image_paths[0] # Take the first image found for testing

    ground_truth_image_pil = Image.open(test_image_path).convert('RGB')
    transform_test = T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor() # Converts to [0, 1]
    ])
    ground_truth_image = transform_test(ground_truth_image_pil).to(device) # C, H, W

    # Simulate corruption: ONLY multiplicative noise, NO blurring
    multiplicative_noise_level_test = 0.12 # Adjust for desired noise level
    log_noise_test = torch.randn_like(ground_truth_image) * multiplicative_noise_level_test
    multiplicative_noise_factor_test = torch.exp(log_noise_test)
    noisy_image_linear_test = ground_truth_image * multiplicative_noise_factor_test
    y_measurement = torch.clamp(noisy_image_linear_test, 0, 1) # y_measurement is now just the noisy image

    # Inference parameters for solve_inverse_problem
    inference_patch_size = 64 # Use a patch size that covers the image or larger patches
    inference_stride = inference_patch_size // 2 # Overlapping patches

    inference_num_steps = 6000 # More steps for better reconstruction
    sigmas_inference = torch.exp(torch.linspace(math.log(NOISE_SCHEDULE_MAX), math.log(NOISE_SCHEDULE_MIN), inference_num_steps + 1)).to(device)
    zeta_values_inference = [0.001] * inference_num_steps # Adjust as needed, can be a schedule

    # Run the inverse problem solver (now acting as a denoiser)
    reconstructed_image = solve_inverse_problem(
        model=model,
        y_measurement=y_measurement, # This is the purely noisy image
        image_shape=ground_truth_image.shape,
        device=device,
        sigmas=sigmas_inference,
        zeta_values=zeta_values_inference,
        patch_size=inference_patch_size,
        stride=inference_stride
    )

    psnr_measurement = calculate_psnr(ground_truth_image, y_measurement)
    psnr_reconstruction = calculate_psnr(ground_truth_image, reconstructed_image)

    print(f"\nPSNR of Simulated Measurement (multiplicative noise only): {psnr_measurement:.2f} dB")
    print(f"PSNR of Reconstructed Image (Denoised): {psnr_reconstruction:.2f} dB")

    print("\n--- Final Reconstruction Results Visualization ---")
    reconstructed_image_np = reconstructed_image.cpu().numpy().transpose(1, 2, 0)
    reconstructed_image_np = np.clip(reconstructed_image_np, 0, 1)

    ground_truth_np = ground_truth_image.cpu().numpy().transpose(1, 2, 0)
    ground_truth_np = np.clip(ground_truth_np, 0, 1)

    y_measurement_np = y_measurement.cpu().numpy().transpose(1, 2, 0)
    y_measurement_np = np.clip(y_measurement_np, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Ground Truth")
    plt.imshow(ground_truth_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Corrupted Measurement (Multiplicative Noise)\nPSNR: {psnr_measurement:.2f} dB")
    plt.imshow(y_measurement_np)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Reconstructed Image (Denoised)\nPSNR: {psnr_reconstruction:.2f} dB")
    plt.imshow(reconstructed_image_np)
    plt.axis('off')

    plt.tight_layout()
    plt.show()