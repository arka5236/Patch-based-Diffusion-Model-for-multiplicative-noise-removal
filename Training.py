import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import glob
import time
import random

# --- Helper Functions and Model Definitions (as provided) ---

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

class CustomImageDataset(Dataset):
    def __init__(self, image_paths_list, image_size, patch_sizes, num_patches_per_image, transform=None, specific_patch_size=None):
        self.image_paths = image_paths_list
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.num_patches_per_image = num_patches_per_image
        self.specific_patch_size = specific_patch_size
        if not self.image_paths:
            print("Warning: No image paths provided to CustomImageDataset.")
        self.base_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.ToTensor() # Converts to [0, 1]
        ])
        self.additional_transform = transform
        if self.specific_patch_size is None:
            self.current_patch_size = max(patch_sizes)
        else:
            self.current_patch_size = self.specific_patch_size

    def __len__(self):
        return len(self.image_paths) * self.num_patches_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.num_patches_per_image
        img_path = self.image_paths[image_idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.base_transform(image)
        if self.additional_transform:
            image_tensor = self.additional_transform(image_tensor)

        h, w = image_tensor.shape[1], image_tensor.shape[2]
        if h < self.current_patch_size or w < self.current_patch_size:
            image_tensor = T.Resize((self.current_patch_size, self.current_patch_size),
                                    interpolation=T.InterpolationMode.BILINEAR)(image_tensor)

        random_crop_transform = T.RandomCrop(self.current_patch_size)
        clean_patch = random_crop_transform(image_tensor)

        # Apply log transformation to the clean patch before returning
        clean_patch_log = torch.log(clean_patch + 1e-6) # Add a small epsilon to prevent log(0)
        return clean_patch_log, clean_patch # Return both log and original for validation/display

def calculate_psnr(img1, img2, max_val=1.0):
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach()

    # Ensure tensors have batch dimension if PSNR is calculated on patches/single images
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10((max_val ** 2) / mse)
    return psnr.item()

# --- Training Script ---

if __name__ == "__main__":
    DATA_DIR = '/content/img_align_celeba/img_align_celeba' # CHANGE THIS TO YOUR IMAGE DIRECTORY
    IMAGE_SIZE = (64, 64)
    PATCH_SIZES = [32, 48, 64]

    # Noise schedule for training will be on the *log* domain's added noise.
    NOISE_SCHEDULE_MIN = 0.01
    NOISE_SCHEDULE_MAX = 0.5

    NUM_PATCHES_PER_IMAGE = 4
    BATCH_SIZE = 16
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    TIME_EMB_DIM = 256
    NUM_EPOCHS = 1
    LEARNING_RATE = 1e-4
    SAVE_EVERY_EPOCHS = 1
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_score_network_model.pth")

    # Early Stopping Parameters
    PATIENCE = 1
    best_val_psnr = -1.0
    epochs_no_improve = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True) # Ensure your DATA_DIR exists and contains images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True) + \
                      glob.glob(os.path.join(DATA_DIR, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(DATA_DIR, '**', '*.jpeg'), recursive=True)

    if not all_image_paths:
        print(f"\nFATAL ERROR: No images found in {DATA_DIR} or its subdirectories. Please populate your DATA_DIR.")
        print("Creating dummy images for demonstration...")
        dummy_dir = os.path.join(DATA_DIR, "dummy")
        os.makedirs(dummy_dir, exist_ok=True)
        for i in range(5):
            dummy_image = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            dummy_image.save(os.path.join(dummy_dir, f"dummy_image_{i}.png"))
        all_image_paths = glob.glob(os.path.join(DATA_DIR, '**', '*.png'), recursive=True)

    random.shuffle(all_image_paths)
    train_split_ratio = 0.9
    split_idx = int(len(all_image_paths) * train_split_ratio)
    train_image_paths = all_image_paths[:split_idx]
    val_image_paths = all_image_paths[split_idx:]

    print(f"Total images found: {len(all_image_paths)}")
    print(f"Training images: {len(train_image_paths)}")
    print(f"Validation images: {len(val_image_paths)}")

    model = ScoreNetwork(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_down_blocks=4,
        time_emb_dim=TIME_EMB_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Starting training for {NUM_EPOCHS} epochs with early stopping (patience={PATIENCE})...")

    for epoch in range(NUM_EPOCHS):
        current_train_patch_size = np.random.choice(PATCH_SIZES)

        train_dataset = CustomImageDataset(
            image_paths_list=train_image_paths,
            image_size=IMAGE_SIZE,
            patch_sizes=PATCH_SIZES,
            num_patches_per_image=NUM_PATCHES_PER_IMAGE,
            transform=None,
            specific_patch_size=current_train_patch_size
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (clean_patch_log, clean_patch_original) in enumerate(train_loader):
            clean_patch_log = clean_patch_log.to(device)

            sigma_t = (NOISE_SCHEDULE_MIN + (NOISE_SCHEDULE_MAX - NOISE_SCHEDULE_MIN) * torch.rand(clean_patch_log.shape[0], device=device)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            noise = torch.randn_like(clean_patch_log)
            noisy_patch_log = clean_patch_log + sigma_t * noise

            optimizer.zero_grad()
            predicted_noise = model(noisy_patch_log, sigma_t.squeeze())
            loss = criterion(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader)
        end_time = time.time()
        print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.6f} - Time: {end_time - start_time:.2f}s")

        # --- Validation Loop (Training part's validation) ---
        model.eval()
        val_running_psnr = 0.0
        val_samples = 0
        current_val_patch_size = max(PATCH_SIZES)
        val_dataset = CustomImageDataset(
            image_paths_list=val_image_paths,
            image_size=IMAGE_SIZE,
            patch_sizes=PATCH_SIZES,
            num_patches_per_image=NUM_PATCHES_PER_IMAGE,
            transform=None,
            specific_patch_size=current_val_patch_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        print(f"Starting validation with patch size {current_val_patch_size}x{current_val_patch_size}...")

        with torch.no_grad():
            for batch_idx, (clean_patch_log_val, clean_patch_original_val) in enumerate(val_loader):
                clean_patch_log_val = clean_patch_log_val.to(device)

                sigma_t_val = (NOISE_SCHEDULE_MIN + (NOISE_SCHEDULE_MAX - NOISE_SCHEDULE_MIN) * torch.rand(clean_patch_log_val.shape[0], device=device)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                noise_val = torch.randn_like(clean_patch_log_val)
                noisy_patch_log_val = clean_patch_log_val + sigma_t_val * noise_val
                predicted_noise_val = model(noisy_patch_log_val, sigma_t_val.squeeze())

                # PSNR calculated on the denoising capability within the log-domain
                denoised_log_patch = noisy_patch_log_val - predicted_noise_val * sigma_t_val # y_t - epsilon_theta * sigma_t
                val_running_psnr += calculate_psnr(clean_patch_log_val, denoised_log_patch) * clean_patch_log_val.size(0)
                val_samples += clean_patch_log_val.size(0)

        if val_samples > 0:
            avg_val_psnr = val_running_psnr / val_samples
            print(f"Epoch {epoch+1} Validation PSNR (denoising only, in log domain): {avg_val_psnr:.2f} dB")
        else:
            avg_val_psnr = -1.0 # No validation samples to calculate PSNR
            print("No validation samples processed.")


        # Early Stopping check
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model with PSNR: {best_val_psnr:.2f} dB")
        else:
            epochs_no_improve += 1
            print(f"Validation PSNR did not improve. Epochs without improvement: {epochs_no_improve}")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining finished.")