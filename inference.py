import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from training import (
    ACNNP,
    AdaptiveMeanPredictor,
    generate_circle_triangle_masks,
    masked_psnr,
    masked_ssim,
)


class PaperBaselineInference:
    """Inference helper for the paper baseline: ACNNP + AMP.

    Stage 1: ACNNP predicts the Triangle set from the Circle set.
    Stage 2: AMP predicts the Circle set from the Circle set itself.
    AMP is kept as a separate object so it can later be replaced by PixelCNN.
    """

    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.acnnp = ACNNP().to(self.device)
        try:
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Older PyTorch versions do not support weights_only.
            state_dict = torch.load(model_path, map_location=self.device)
        self.acnnp.load_state_dict(state_dict)
        self.acnnp.eval()
        self.amp = AdaptiveMeanPredictor().to(self.device)

    def predict(self, image):
        _, _, height, width = image.shape
        circle_mask, triangle_mask = generate_circle_triangle_masks(
            height, width, image.device
        )

        with torch.no_grad():
            # ACNNP: keep only Circle pixels as input, predict only Triangle pixels.
            circle_image = image * circle_mask
            pred_triangle = self.acnnp(circle_image) * triangle_mask

            # AMP: deterministic 8-bit mean predictor for Circle pixels.
            circle_image_255 = torch.round(image * 255.0) * circle_mask
            pred_circle_255, amp_direction = self.amp(circle_image_255, circle_mask)
            pred_circle = (pred_circle_255 / 255.0).clamp(0.0, 1.0)

            # These images are useful for comparing the baseline with PixelCNN later.
            predicted_full = pred_circle + pred_triangle
            acnnp_reconstruction = circle_image + pred_triangle

        return {
            "circle_mask": circle_mask,
            "triangle_mask": triangle_mask,
            "circle_image": circle_image,
            "triangle_image": image * triangle_mask,
            "pred_triangle": pred_triangle,
            "pred_circle": pred_circle,
            "predicted_full": predicted_full,
            "acnnp_reconstruction": acnnp_reconstruction,
            "amp_direction": amp_direction,
        }


def load_image(img_path, image_size, device):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def resolve_output_dir(out_dir, experiment_name):
    """Save each experiment under inference_results/<experiment_name>/.

    Passing --out-dir inference_results keeps the default layout:
    inference_results/baseline/. Passing --out-dir inference_results/baseline
    is also accepted and will not create baseline/baseline.
    """
    normalized = os.path.normpath(out_dir)
    if os.path.basename(normalized) == experiment_name:
        return normalized
    return os.path.join(normalized, experiment_name)


def inference(
    model_path,
    img_path,
    device="cpu",
    out_dir="inference_results",
    experiment_name="baseline",
    image_size=512,
):
    output_dir = resolve_output_dir(out_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    runner = PaperBaselineInference(model_path=model_path, device=device)
    image = load_image(img_path, image_size=image_size, device=runner.device)
    outputs = runner.predict(image)

    tri_psnr = masked_psnr(outputs["pred_triangle"], image, outputs["triangle_mask"])
    tri_ssim = masked_ssim(outputs["pred_triangle"], image, outputs["triangle_mask"])
    cir_psnr = masked_psnr(outputs["pred_circle"], image, outputs["circle_mask"])
    cir_ssim = masked_ssim(outputs["pred_circle"], image, outputs["circle_mask"])

    print(f"[ACNNP Circle -> Triangle] PSNR: {tri_psnr:.2f} dB | SSIM: {tri_ssim:.4f}")
    print(
        f"[AMP Circle -> Circle] direction: {outputs['amp_direction']} | "
        f"PSNR: {cir_psnr:.2f} dB | SSIM: {cir_ssim:.4f}"
    )

    save_image(image, os.path.join(output_dir, "original.png"))
    save_image(outputs["circle_image"], os.path.join(output_dir, "circle_input.png"))
    save_image(outputs["triangle_image"], os.path.join(output_dir, "triangle_target.png"))
    save_image(outputs["pred_triangle"], os.path.join(output_dir, "pred_triangle_acnnp.png"))
    save_image(outputs["pred_circle"], os.path.join(output_dir, "pred_circle_amp.png"))
    save_image(
        outputs["acnnp_reconstruction"],
        os.path.join(output_dir, "circle_plus_pred_triangle.png"),
    )
    save_image(
        outputs["predicted_full"],
        os.path.join(output_dir, "predicted_full_acnnp_amp.png"),
    )

    print(f"Results saved in {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACNNP + AMP baseline inference.")
    parser.add_argument("--model-path", default="baseline_acnnp_amp.pth")
    parser.add_argument("--img-path", default="lena.bmp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="inference_results")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(
        model_path=args.model_path,
        img_path=args.img_path,
        device=args.device,
        out_dir=args.out_dir,
        experiment_name=args.experiment_name,
        image_size=args.image_size,
    )
