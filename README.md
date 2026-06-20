# Special Project

## English

### Overview

This repository contains experiments based on *Novel asymmetric CNN-based and adaptive mean predictors for reversible data hiding in encrypted images*.

The first milestone reproduces the paper-style **ACNNP + AMP** baseline. The second milestone, `improvement_v1`, replaces AMP with a grayscale autoregressive **PixelCNNPredictor** while keeping ACNNP fixed.

### Method Summary

- **ACNNP** predicts the Triangle pixel set from the Circle pixel set.
- **AMP** predicts the Circle pixel set using the paper's adaptive mean predictor.
- **PixelCNNPredictor** is the first AMP replacement experiment.
- The current PixelCNN version uses grayscale 1-channel input/output and `masked_mse_loss`, matching the loss style used by ACNNP.

### Results

| Experiment | Triangle Predictor | Circle Predictor | Triangle PSNR | Triangle SSIM | Circle PSNR | Circle SSIM | Output | Log |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | ACNNP | AMP | 35.53 dB | 0.9960 | 30.35 dB | 0.9866 | [inference_results/baseline/](inference_results/baseline/) | [logs/acnnp_baseline_20260525.log](logs/acnnp_baseline_20260525.log) |
| improvement_v1 | ACNNP | PixelCNN | 35.53 dB | 0.9960 | 28.70 dB | 0.9813 | [inference_results/improvement_v1/](inference_results/improvement_v1/) | [logs/pixelcnn_improvement_20260620.log](logs/pixelcnn_improvement_20260620.log) |

### Output Layout

Inference outputs are stored under experiment-specific directories:

```text
inference_results/
├── baseline/           # ACNNP + AMP baseline results
└── improvement_v1/     # ACNNP + PixelCNN results
```

The baseline directory includes:

- `original.png`
- `circle_input.png`
- `triangle_target.png`
- `pred_triangle_acnnp.png`
- `pred_circle_amp.png`
- `circle_plus_pred_triangle.png`
- `predicted_full_acnnp_amp.png`

The PixelCNN improvement directory includes:

- `original.png`
- `circle_input.png`
- `triangle_target.png`
- `pred_triangle_acnnp.png`
- `pred_circle_pixelcnn.png`
- `circle_plus_pred_triangle.png`
- `predicted_full_acnnp_pixelcnn.png`

### Analysis

The first PixelCNN replacement underperformed the AMP baseline on Circle prediction:

```text
AMP Circle PSNR:      30.35 dB
PixelCNN Circle PSNR: 28.70 dB
```

The most likely reason is not the use of MSE itself, but the current PixelCNN direction and structure. The implemented PixelCNN is a single raster-order autoregressive model, mostly using upper-left causal context. In the baseline inference, AMP selected the `bottom` direction, which suggests that a single raster direction is not a fair replacement for the adaptive four-direction AMP.

For this project, MSE is still the right first loss because the target metric is PSNR/SSIM and ACNNP also uses masked MSE. PixelCNN++'s discretized logistic mixture likelihood is worth considering later, but it is more complex and does not directly optimize PSNR.

### Next Steps

1. Implement a four-direction PixelCNN that mirrors AMP's top/left/right/bottom direction choices.
2. Select the best direction by validation Circle MSE or PSNR.
3. Compare the best-direction PixelCNN against AMP.
4. Try larger PixelCNN capacity after the directional issue is fixed.
5. Consider discretized logistic mixture likelihood as a later experiment.

### Branch Workflow

The current workflow is:

1. `feature/baseline-acnnp-amp`: ACNNP + AMP baseline.
2. `feature/pixelcnn-integration`: ACNNP + PixelCNN `improvement_v1`.
3. Next branch idea: four-direction PixelCNN improvement.

## 日本語

### 概要

このリポジトリは、論文 *Novel asymmetric CNN-based and adaptive mean predictors for reversible data hiding in encrypted images* をもとにした実験環境です。

最初のマイルストーンでは、論文の **ACNNP + AMP** baseline を再現しました。次の `improvement_v1` では、ACNNPを固定したまま、AMPをgrayscale自己回帰型の **PixelCNNPredictor** に置き換えました。

### 手法の概要

- **ACNNP**: Circle画素集合からTriangle画素集合を予測します。
- **AMP**: 論文のadaptive mean predictorによりCircle画素集合を自己予測します。
- **PixelCNNPredictor**: AMP置き換えの初回実験です。
- 現在のPixelCNNはgrayscale 1チャンネル入出力で、ACNNPと同じ方針の `masked_mse_loss` を使っています。

### 評価結果

| 実験 | Triangle予測器 | Circle予測器 | Triangle PSNR | Triangle SSIM | Circle PSNR | Circle SSIM | 出力 | ログ |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | ACNNP | AMP | 35.53 dB | 0.9960 | 30.35 dB | 0.9866 | [inference_results/baseline/](inference_results/baseline/) | [logs/acnnp_baseline_20260525.log](logs/acnnp_baseline_20260525.log) |
| improvement_v1 | ACNNP | PixelCNN | 35.53 dB | 0.9960 | 28.70 dB | 0.9813 | [inference_results/improvement_v1/](inference_results/improvement_v1/) | [logs/pixelcnn_improvement_20260620.log](logs/pixelcnn_improvement_20260620.log) |

### 出力ディレクトリ構造

推論結果は、実験ごとに以下のディレクトリへ保存します。

```text
inference_results/
├── baseline/           # ACNNP + AMP baseline results
└── improvement_v1/     # ACNNP + PixelCNN results
```

baselineディレクトリには以下が保存されます。

- `original.png`
- `circle_input.png`
- `triangle_target.png`
- `pred_triangle_acnnp.png`
- `pred_circle_amp.png`
- `circle_plus_pred_triangle.png`
- `predicted_full_acnnp_amp.png`

PixelCNN版のディレクトリには以下が保存されます。

- `original.png`
- `circle_input.png`
- `triangle_target.png`
- `pred_triangle_acnnp.png`
- `pred_circle_pixelcnn.png`
- `circle_plus_pred_triangle.png`
- `predicted_full_acnnp_pixelcnn.png`

### 考察

今回のPixelCNN置き換えでは、Circle予測の性能がAMP baselineを下回りました。

```text
AMP Circle PSNR:      30.35 dB
PixelCNN Circle PSNR: 28.70 dB
```

主な原因候補は、MSEそのものではなく、現在のPixelCNNの予測方向と構造です。今回のPixelCNNは単一のraster順序の自己回帰モデルで、主に左上方向のcausal contextを使います。一方でbaseline推論では、AMPは `bottom` 方向を選択していました。つまり、単一方向PixelCNNは、4方向から適応的に選ぶAMPの置き換えとしては不利です。

このプロジェクトでは、まずMSEを使い続けるのが妥当です。評価指標がPSNR/SSIMであり、ACNNPもmasked MSEで学習しているためです。PixelCNN++本来のdiscretized logistic mixture likelihoodは今後の候補ですが、実装が複雑で、PSNRを直接最大化するlossではありません。

### 次の改善方針

1. AMPと同じ top / left / right / bottom に対応する4方向PixelCNNを実装する。
2. validationのCircle MSEまたはPSNRで最良方向を選ぶ。
3. best-direction PixelCNN と AMP を比較する。
4. 方向問題を解いたあとでPixelCNNの容量を増やす。
5. さらに後の実験としてdiscretized logistic mixture likelihoodを検討する。

### ブランチ運用

現在の流れは以下です。

1. `feature/baseline-acnnp-amp`: ACNNP + AMP baseline。
2. `feature/pixelcnn-integration`: ACNNP + PixelCNN `improvement_v1`。
3. 次の候補: 4方向PixelCNN改善ブランチ。
