# Special Project

## English

### Overview

This repository contains an experimental baseline implementation of the paper method **ACNNP + AMP** from *Novel asymmetric CNN-based and adaptive mean predictors for reversible data hiding in encrypted images*.

The current baseline is organized as follows:

- **ACNNP** predicts the Triangle pixel set from the Circle pixel set.
- **AMP** predicts the Circle pixel set using an adaptive mean predictor.
- The result directory is separated from future improvements so that the next PixelCNN-based predictor can be compared cleanly.

### Baseline Result

| Experiment | Predictor | Target | PSNR | SSIM | Result directory | Log |
| --- | --- | --- | ---: | ---: | --- | --- |
| ACNNP + AMP baseline | ACNNP | Circle -> Triangle | 35.53 dB | 0.9960 | [inference_results/baseline/](inference_results/baseline/) | [logs/acnnp_baseline_20260525.log](logs/acnnp_baseline_20260525.log) |

AMP inference result:

| Predictor | Target | Direction | PSNR | SSIM |
| --- | --- | --- | ---: | ---: |
| AMP | Circle -> Circle | bottom | 30.35 dB | 0.9866 |

### Output Layout

Inference outputs are stored under experiment-specific directories:

```text
inference_results/
├── baseline/           # ACNNP + AMP baseline results
└── improvement_v1/     # Reserved for future PixelCNN results
```

Baseline images are expected under [inference_results/baseline/](inference_results/baseline/), including:

- `original.png`
- `circle_input.png`
- `triangle_target.png`
- `pred_triangle_acnnp.png`
- `pred_circle_amp.png`
- `circle_plus_pred_triangle.png`
- `predicted_full_acnnp_amp.png`


## 日本語

### 概要

このリポジトリは、論文 *Novel asymmetric CNN-based and adaptive mean predictors for reversible data hiding in encrypted images* の手法である **ACNNP + AMP** を再現するための実験ベースラインです。

現在のベースラインは以下の構成です。

- **ACNNP**: Circle画素集合からTriangle画素集合を予測します。
- **AMP**: adaptive mean predictorによりCircle画素集合を自己予測します。
- 今後のPixelCNN版と比較しやすいように、推論結果の保存先を実験ごとに分離しています。

### ベースライン評価結果

| 実験 | 予測器 | 対象 | PSNR | SSIM | 結果ディレクトリ | ログ |
| --- | --- | --- | ---: | ---: | --- | --- |
| ACNNP + AMP baseline | ACNNP | Circle -> Triangle | 35.53 dB | 0.9960 | [inference_results/baseline/](inference_results/baseline/) | [logs/acnnp_baseline_20260525.log](logs/acnnp_baseline_20260525.log) |

AMPの推論結果:

| 予測器 | 対象 | 方向 | PSNR | SSIM |
| --- | --- | --- | ---: | ---: |
| AMP | Circle -> Circle | bottom | 30.35 dB | 0.9866 |

### 出力ディレクトリ構造

推論結果は、実験ごとに以下のディレクトリへ保存します。

```text
inference_results/
├── baseline/           # 今回のACNNP + AMPの結果
└── improvement_v1/     # 将来のPixelCNNの結果用
```

ベースラインの推論画像は [inference_results/baseline/](inference_results/baseline/) に保存されます。

