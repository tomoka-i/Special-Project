import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from training import (
    ACNNP,
    AdaptiveMeanPredictor,
    PixelCNNPredictor,
    generate_circle_triangle_masks,
    masked_psnr,
    masked_ssim,
)


def load_state_dict_safely(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class PaperBaselineInference:
    """Inference helper for the paper baseline: ACNNP + AMP.

    Stage 1: ACNNP predicts the Triangle set from the Circle set.
    Stage 2: AMP predicts the Circle set from the Circle set itself.
    AMP is kept as a separate object so it can later be replaced by PixelCNN.
    """

    def __init__(self, model_path, device):
        self.device = torch.device(device)
        self.acnnp = ACNNP().to(self.device)
        self.acnnp.load_state_dict(load_state_dict_safely(model_path, self.device))
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


class PixelCNNImprovementInference:
    """Inference helper for the improvement: ACNNP + PixelCNNPredictor."""

    def __init__(self, acnnp_path, pixelcnn_path, device, hidden_channels=32, num_blocks=3):
        self.device = torch.device(device)

        self.acnnp = ACNNP().to(self.device)
        self.acnnp.load_state_dict(load_state_dict_safely(acnnp_path, self.device))
        self.acnnp.eval()

        self.pixelcnn = PixelCNNPredictor(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
        ).to(self.device)
        self.pixelcnn.load_state_dict(load_state_dict_safely(pixelcnn_path, self.device))
        self.pixelcnn.eval()

    def predict(self, image):
        _, _, height, width = image.shape
        circle_mask, triangle_mask = generate_circle_triangle_masks(
            height, width, image.device
        )

        with torch.no_grad():
            circle_image = image * circle_mask
            pred_triangle = self.acnnp(circle_image) * triangle_mask
            pred_circle = self.pixelcnn(circle_image) * circle_mask
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
            "circle_predictor_name": "PixelCNN",
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
    experiment_name=None,
    mode="pixelcnn",
    pixelcnn_path="pixelcnn_predictor.pth",
    pixelcnn_hidden_channels=32,
    pixelcnn_blocks=3,
    image_size=512,
):
    if experiment_name is None:
        experiment_name = "baseline" if mode == "baseline" else "improvement_v1"
    output_dir = resolve_output_dir(out_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if mode == "baseline":
        runner = PaperBaselineInference(model_path=model_path, device=device)
    else:
        runner = PixelCNNImprovementInference(
            acnnp_path=model_path,
            pixelcnn_path=pixelcnn_path,
            device=device,
            hidden_channels=pixelcnn_hidden_channels,
            num_blocks=pixelcnn_blocks,
        )

    image = load_image(img_path, image_size=image_size, device=runner.device)
    outputs = runner.predict(image)
    circle_predictor_name = outputs.get("circle_predictor_name", "AMP")

    tri_psnr = masked_psnr(outputs["pred_triangle"], image, outputs["triangle_mask"])
    tri_ssim = masked_ssim(outputs["pred_triangle"], image, outputs["triangle_mask"])
    cir_psnr = masked_psnr(outputs["pred_circle"], image, outputs["circle_mask"])
    cir_ssim = masked_ssim(outputs["pred_circle"], image, outputs["circle_mask"])

    print(f"[ACNNP Circle -> Triangle] PSNR: {tri_psnr:.2f} dB | SSIM: {tri_ssim:.4f}")
    if circle_predictor_name == "AMP":
        print(
            f"[AMP Circle -> Circle] direction: {outputs['amp_direction']} | "
            f"PSNR: {cir_psnr:.2f} dB | SSIM: {cir_ssim:.4f}"
        )
    else:
        print(
            f"[{circle_predictor_name} Circle -> Circle] "
            f"PSNR: {cir_psnr:.2f} dB | SSIM: {cir_ssim:.4f}"
        )

    save_image(image, os.path.join(output_dir, "original.png"))
    save_image(outputs["circle_image"], os.path.join(output_dir, "circle_input.png"))
    save_image(outputs["triangle_image"], os.path.join(output_dir, "triangle_target.png"))
    save_image(outputs["pred_triangle"], os.path.join(output_dir, "pred_triangle_acnnp.png"))
    circle_filename = "pred_circle_amp.png" if mode == "baseline" else "pred_circle_pixelcnn.png"
    full_filename = (
        "predicted_full_acnnp_amp.png"
        if mode == "baseline"
        else "predicted_full_acnnp_pixelcnn.png"
    )
    save_image(outputs["pred_circle"], os.path.join(output_dir, circle_filename))
    save_image(
        outputs["acnnp_reconstruction"],
        os.path.join(output_dir, "circle_plus_pred_triangle.png"),
    )
    save_image(
        outputs["predicted_full"],
        os.path.join(output_dir, full_filename),
    )

    print(f"Results saved in {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACNNP baseline or PixelCNN improvement inference.")
    parser.add_argument("--mode", choices=("baseline", "pixelcnn"), default="pixelcnn")
    parser.add_argument("--model-path", default="baseline_acnnp_amp.pth")
    parser.add_argument("--pixelcnn-path", default="pixelcnn_predictor.pth")
    parser.add_argument("--pixelcnn-hidden-channels", type=int, default=32)
    parser.add_argument("--pixelcnn-blocks", type=int, default=3)
    parser.add_argument("--img-path", default="lena.bmp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="inference_results")
    parser.add_argument("--experiment-name", default=None)
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
        mode=args.mode,
        pixelcnn_path=args.pixelcnn_path,
        pixelcnn_hidden_channels=args.pixelcnn_hidden_channels,
        pixelcnn_blocks=args.pixelcnn_blocks,
        image_size=args.image_size,
    )
