import argparse
import os
from math import log10

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ============================================================
# Mask and metric utilities
# ============================================================
def generate_circle_triangle_masks(height, width, device):
    """Split an image into the paper's Circle and Triangle checkerboard sets."""
    rows = torch.arange(height, device=device).view(height, 1)
    cols = torch.arange(width, device=device).view(1, width)
    circle = ((rows + cols) % 2 == 0).float().view(1, 1, height, width)
    triangle = 1.0 - circle
    return circle, triangle


def masked_mse_loss(pred, target, mask):
    """MSE over only the valid predicted set, not over zero-filled blank pixels."""
    diff = (pred - target) * mask
    denom = mask.sum() * target.shape[0]
    return diff.pow(2).sum() / denom.clamp_min(1.0)


def masked_psnr(pred, target, mask):
    mse = masked_mse_loss(pred, target, mask).item()
    if mse == 0:
        return 100.0
    return 10 * log10(1.0 / mse)


def masked_ssim(img1, img2, mask):
    """A compact masked SSIM proxy for validation logging."""
    mask = mask.expand_as(img1).bool()
    x = img1[mask]
    y = img2[mask]

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std(unbiased=False)
    sigma_y = y.std(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
    )
    return ssim.item()


# ============================================================
# ACNNP: stage 1, Circle -> Triangle
# ============================================================
class AsymmetricConvBlock(nn.Module):
    """The paper's asymmetric block: 1xK, KxK, Kx1 branches + two 3x3 convs."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2

        self.conv_1xk = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            padding_mode="reflect",
        )
        self.conv_kxk = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(pad, pad),
            padding_mode="reflect",
        )
        self.conv_kx1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            padding_mode="reflect",
        )

        self.tail = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        x = self.conv_1xk(x) + self.conv_kxk(x) + self.conv_kx1(x)
        return self.tail(x)


class ACNNP(nn.Module):
    """Asymmetric CNN-based predictor described in Section 2.1.2."""

    def __init__(self, channels=32):
        super().__init__()
        self.branch3 = AsymmetricConvBlock(1, channels, kernel_size=3)
        self.branch5 = AsymmetricConvBlock(1, channels, kernel_size=5)
        self.branch7 = AsymmetricConvBlock(1, channels, kernel_size=7)

        # Four 3x3 conv layers. The feature-extraction output is fused after layer 2.
        self.pred_conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.pred_conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.pred_conv3 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")
        self.pred_conv4 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode="reflect")

        self.reconstruction = nn.Conv2d(
            channels, 1, 3, padding=1, padding_mode="reflect"
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, circle_image):
        features = (
            self.branch3(circle_image)
            + self.branch5(circle_image)
            + self.branch7(circle_image)
        )

        x = self.activation(self.pred_conv1(features))
        x = self.activation(self.pred_conv2(x))
        x = x + features
        x = self.activation(self.pred_conv3(x))
        x = self.activation(self.pred_conv4(x))

        return self.reconstruction(x)


# ============================================================
# AMP: stage 2, Circle -> Circle self-prediction
# ============================================================
class AdaptiveMeanPredictor(nn.Module):
    """Paper baseline AMP.

    AMP is not trained. It predicts Circle pixels by averaging two diagonal
    Circle pixels in one of four scan directions, then chooses the direction
    with the lowest Circle-set MSE for each image.
    """

    directions = ("top", "left", "right", "bottom")

    def forward(self, image, circle_mask):
        candidates = self.predict_all_directions(image)
        losses = [
            masked_mse_loss(candidates[name], image, circle_mask)
            for name in self.directions
        ]
        best_index = int(torch.argmin(torch.stack(losses)).item())
        best_direction = self.directions[best_index]
        return candidates[best_direction] * circle_mask, best_direction

    def predict_all_directions(self, image):
        # The formulas below are the 0-indexed tensor version of Eqs. (2)-(5).
        return {
            "top": self._predict_top(image),
            "left": self._predict_left(image),
            "right": self._predict_right(image),
            "bottom": self._predict_bottom(image),
        }

    @staticmethod
    def _predict_top(image):
        pred = torch.zeros_like(image)
        pred[:, :, 0, :] = image[:, :, 0, :]
        pred[:, :, :, 0] = image[:, :, :, 0]
        pred[:, :, :, -1] = image[:, :, :, -1]
        pred[:, :, 1:, 1:-1] = torch.floor(
            (image[:, :, :-1, :-2] + image[:, :, :-1, 2:]) / 2.0
        )
        return pred

    @staticmethod
    def _predict_left(image):
        pred = torch.zeros_like(image)
        pred[:, :, :, 0] = image[:, :, :, 0]
        pred[:, :, 0, :] = image[:, :, 0, :]
        pred[:, :, -1, :] = image[:, :, -1, :]
        pred[:, :, 1:-1, 1:] = torch.floor(
            (image[:, :, :-2, :-1] + image[:, :, 2:, :-1]) / 2.0
        )
        return pred

    @staticmethod
    def _predict_right(image):
        pred = torch.zeros_like(image)
        pred[:, :, 0, :] = image[:, :, 0, :]
        pred[:, :, :, -1] = image[:, :, :, -1]
        pred[:, :, -1, :] = image[:, :, -1, :]
        pred[:, :, 1:-1, :-1] = torch.floor(
            (image[:, :, :-2, 1:] + image[:, :, 2:, 1:]) / 2.0
        )
        return pred

    @staticmethod
    def _predict_bottom(image):
        pred = torch.zeros_like(image)
        pred[:, :, :, 0] = image[:, :, :, 0]
        pred[:, :, :, -1] = image[:, :, :, -1]
        pred[:, :, -1, :] = image[:, :, -1, :]
        pred[:, :, :-1, 1:-1] = torch.floor(
            (image[:, :, 1:, :-2] + image[:, :, 1:, 2:]) / 2.0
        )
        return pred


class PaperBaselinePredictor(nn.Module):
    """ACNNP + AMP baseline, with a clean seam for replacing AMP by PixelCNN."""

    def __init__(self, acnnp=None, circle_predictor=None):
        super().__init__()
        self.acnnp = acnnp if acnnp is not None else ACNNP()
        self.circle_predictor = (
            circle_predictor if circle_predictor is not None else AdaptiveMeanPredictor()
        )

    def predict_triangle(self, image, circle_mask, triangle_mask):
        circle_image = image * circle_mask
        pred_triangle = self.acnnp(circle_image) * triangle_mask
        return pred_triangle

    def predict_circle(self, image, circle_mask):
        circle_image_255 = torch.round(image * 255.0) * circle_mask
        pred_circle_255, direction = self.circle_predictor(circle_image_255, circle_mask)
        return (pred_circle_255 / 255.0).clamp(0.0, 1.0), direction

    def forward(self, image):
        _, _, height, width = image.shape
        circle_mask, triangle_mask = generate_circle_triangle_masks(
            height, width, image.device
        )
        pred_triangle = self.predict_triangle(image, circle_mask, triangle_mask)
        pred_circle, amp_direction = self.predict_circle(image, circle_mask)
        predicted_full = pred_circle + pred_triangle
        return {
            "circle_mask": circle_mask,
            "triangle_mask": triangle_mask,
            "pred_triangle": pred_triangle,
            "pred_circle": pred_circle,
            "predicted_full": predicted_full,
            "amp_direction": amp_direction,
        }


# ============================================================
# Training
# ============================================================
def train_model_imagenette(
    root_dir="imagenette2-320",
    device="cuda",
    epochs=20,
    batch_size=8,
    lr=1e-3,
    min_lr=1e-5,
    weight_decay=1e-3,
    output_path="baseline_acnnp_amp.pth",
):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_dir, "train"),
        transform=transform,
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(root_dir, "val"),
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = PaperBaselinePredictor().to(device)
    optimizer = torch.optim.Adam(
        model.acnnp.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr,
    )

    best_val_psnr = float("-inf")
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            _, _, height, width = imgs.shape
            circle_mask, triangle_mask = generate_circle_triangle_masks(
                height, width, device
            )

            # Stage 1 training: ACNNP learns only Circle -> Triangle.
            pred_triangle = model.predict_triangle(imgs, circle_mask, triangle_mask)
            loss = masked_mse_loss(pred_triangle, imgs, triangle_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_tri_psnr = 0.0
        total_tri_ssim = 0.0
        total_amp_psnr = 0.0

        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)

                total_tri_psnr += masked_psnr(
                    outputs["pred_triangle"],
                    imgs,
                    outputs["triangle_mask"],
                )
                total_tri_ssim += masked_ssim(
                    outputs["pred_triangle"],
                    imgs,
                    outputs["triangle_mask"],
                )
                total_amp_psnr += masked_psnr(
                    outputs["pred_circle"],
                    imgs,
                    outputs["circle_mask"],
                )

        avg_tri_psnr = total_tri_psnr / len(val_loader)
        avg_tri_ssim = total_tri_ssim / len(val_loader)
        avg_amp_psnr = total_amp_psnr / len(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = avg_tri_psnr > best_val_psnr
        if improved:
            best_val_psnr = avg_tri_psnr
            best_epoch = epoch + 1
            torch.save(model.acnnp.state_dict(), output_path)

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"LR: {current_lr:.2e} | "
            f"Train Triangle Loss: {avg_train_loss:.6f} | "
            f"Val ACNNP Triangle PSNR: {avg_tri_psnr:.2f} dB | "
            f"Val ACNNP Triangle SSIM: {avg_tri_ssim:.4f} | "
            f"Val AMP Circle PSNR: {avg_amp_psnr:.2f} dB | "
            f"Best: {best_val_psnr:.2f} dB @ epoch {best_epoch}"
            f"{' | saved best' if improved else ''}"
        )

    print(f"Best ACNNP weights saved to '{output_path}'")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train paper baseline ACNNP + AMP.")
    parser.add_argument("--root-dir", default="imagenette2-320")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--output", default="baseline_acnnp_amp.pth")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {current_device}")

    baseline = train_model_imagenette(
        root_dir=args.root_dir,
        device=current_device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        output_path=args.output,
    )

    print(f"Training complete. Use '{args.output}' for inference.")
