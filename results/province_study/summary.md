| run | architecture | strategy | trainable params | best epoch | val acc | **test acc** (n=567) | test acc, clean subset (n=491) | test macro-F1 | train time | hardware |
|---|---|---|---|---|---|---|---|---|---|---|
| A_resnet18_scratch | resnet18 | from-scratch | 11,189,850 | 38 | 93.8 % | **93.3 %** | 94.7 % | 92.9 % | 6.1 min | Tesla T4 |
| B_resnet18_frozen | resnet18 | feature-extraction | 13,338 | 39 | 40.1 % | **42.0 %** | 43.2 % | 37.9 % | 5.3 min | Tesla T4 |
| C_resnet18_finetune | resnet18 | fine-tuning | 11,189,850 | 31 | 92.6 % | **94.0 %** | 95.3 % | 92.4 % | 5.9 min | Tesla T4 |
| D_smallcnn_scratch | small_cnn | from-scratch | 396,058 | 33 | 48.4 % | **46.6 %** | 47.7 % | 40.7 % | 5.3 min | Tesla T4 |
