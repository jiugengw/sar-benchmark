# Training curves

Back to the [README](../README.md). One row per packaged ZIP. Regenerate with
`python tools/plot_training_curves.py`; what each run must save is in [RUN_TEMPLATE.md](RUN_TEMPLATE.md).

How to read them:

- **Still learning:** validation mAP / F1 / Δm still rising at the last epoch; more epochs may help.
- **Overfitting:** training loss keeps falling while validation loss flattens or rises, and validation
  metrics stop improving or drop. Train shorter, regularize, or pick an earlier checkpoint.
- **Plateau:** both losses flat and metrics flat; change the recipe rather than training longer.
- The dotted line marks the epoch with the best validation Δm, when F1 and mAP were both recorded.

| ZIP | Curves | Note |
|---|---|---|
| `nonsubmissions/141026/terramind_pretrained_moelora_box_localization_seed0/terramind_pretrained_moelora_box_localization_seed0_calibrated.zip` | [curves](../nonsubmissions/141026/terramind_pretrained_moelora_box_localization_seed0/terramind_pretrained_moelora_box_localization_seed0_calibrated.curves.png) |  |
| `nonsubmissions/141026/terramind_pretrained_moelora_box_localization_seed0/tm_moelora_boxloc_s0_cal.zip` | [curves](../nonsubmissions/141026/terramind_pretrained_moelora_box_localization_seed0/tm_moelora_boxloc_s0_cal.curves.png) |  |
| `nonsubmissions/151026/improve-48-324733/boxloss_photometric_calibrated.zip` | [curves](../nonsubmissions/151026/improve-48-324733/boxloss_photometric_calibrated.curves.png) |  |
| `nonsubmissions/151026/improve-48-324733/boxloss_photometric_large_anchors_calibrated.zip` | [curves](../nonsubmissions/151026/improve-48-324733/boxloss_photometric_large_anchors_calibrated.curves.png) |  |
| `nonsubmissions/151026/small48-resnet34-326077/resnet34_small_extreme_calibrated.zip` | [curves](../nonsubmissions/151026/small48-resnet34-326077/resnet34_small_extreme_calibrated.curves.png) |  |
| `nonsubmissions/151026/terramind_48ep_boxloss_photometric_large_anchors_seed0_calibrated.zip` | [curves](../nonsubmissions/151026/terramind_48ep_boxloss_photometric_large_anchors_seed0_calibrated.curves.png) |  |
| `nonsubmissions/151026/terramind_pretrained_moelora_calibrated.zip` | [curves](../nonsubmissions/151026/terramind_pretrained_moelora_calibrated.curves.png) | score calibration of the 24-epoch run |
| `nonsubmissions/151026/vit48-325636/boxloss1_photometric_large_anchors_calibrated.zip` | [curves](../nonsubmissions/151026/vit48-325636/boxloss1_photometric_large_anchors_calibrated.curves.png) |  |
| `nonsubmissions/151026/vit48-325636/boxloss2_photometric_large_anchors_calibrated.zip` | [curves](../nonsubmissions/151026/vit48-325636/boxloss2_photometric_large_anchors_calibrated.curves.png) |  |
| `nonsubmissions/151026/vit_48ep_boxloss1_photometric_large_anchors_seed0_calibrated.zip` | [curves](../nonsubmissions/151026/vit_48ep_boxloss1_photometric_large_anchors_seed0_calibrated.curves.png) |  |
| `nonsubmissions/151026/vit_48ep_boxloss2_photometric_large_anchors_seed0_calibrated.zip` | [curves](../nonsubmissions/151026/vit_48ep_boxloss2_photometric_large_anchors_seed0_calibrated.curves.png) |  |
| `nonsubmissions/161026/denodet-v2-329477-epoch32/denodet-v2-329477.zip` | **missing** | DenoDet logs are only in runs/denodet-v2-329477/ on the cluster |
| `nonsubmissions/161026/tm-48baseline-326894/terramind_48epoch_baseline_calibrated.zip` | [curves](../nonsubmissions/161026/tm-48baseline-326894/terramind_48epoch_baseline_calibrated.curves.png) |  |
| `nonsubmissions/161026/vit_lora_sar_cnn_48ep_seed42/submission_val.zip` | **missing** | trained in an external project; no per-epoch history was copied here |
| `nonsubmissions/161026/yolo11s-100-standard-328063/yolo11s_100_standard_328063.zip` | [curves](../nonsubmissions/161026/yolo11s-100-standard-328063/yolo11s_100_standard_328063.curves.png) |  |
| `nonsubmissions/161026/yolo26m-100-p2-328339/yolo26m_100_p2_328339.zip` | [curves](../nonsubmissions/161026/yolo26m-100-p2-328339/yolo26m_100_p2_328339.curves.png) | same E5 detector, earlier packaging |
| `nonsubmissions/171026/E2-328622/E2_328622.zip` | [curves](../nonsubmissions/171026/E2-328622/E2_328622.curves.png) |  |
| `nonsubmissions/171026/E3-328622/E3_328622.zip` | [curves](../nonsubmissions/171026/E3-328622/E3_328622.curves.png) |  |
| `nonsubmissions/171026/E6-328598/E6_328598.zip` | [curves](../nonsubmissions/171026/E6-328598/E6_328598.curves.png) |  |
| `nonsubmissions/171026/E9-329376/E9_329376.zip` | [curves](../nonsubmissions/171026/E9-329376/E9_329376.curves.png) |  |
| `nonsubmissions/171026/YOLO26s-prior-328598/YOLO26s_P2_328338.zip` | [curves](../nonsubmissions/171026/YOLO26s-prior-328598/YOLO26s_P2_328338.curves.png) |  |
| `nonsubmissions/171026/yolo11m-100-p2-329019/yolo11m_100_p2_329019.zip` | [curves](../nonsubmissions/171026/yolo11m-100-p2-329019/yolo11m_100_p2_329019.curves.png) |  |
| `nonsubmissions/171026/yolo26l-100-p2-329019/yolo26l_100_p2_329019.zip` | [curves](../nonsubmissions/171026/yolo26l-100-p2-329019/yolo26l_100_p2_329019.curves.png) |  |
| `nonsubmissions/171026/yolo26m-map-only-groups-20261017/yolo26m_map_only_groups_20261017.zip` | **missing** | per-epoch history was not exported from the cluster run |
| `nonsubmissions/171026/yolo26m_p2_box_bias_20261017/yolo26m_p2_box_bias_20261017.zip` | [curves](../nonsubmissions/171026/yolo26m_p2_box_bias_20261017/yolo26m_p2_box_bias_20261017.curves.png) | box-edge correction only; no extra training |
| `nonsubmissions/171026/yolo26m_p2_finetune_329644/continue_2/deadline-yolo26m-329644_continue_2.zip` | **missing** | second continuation block: results.csv is only in the cluster run folder runs/deadline-yolo26m-329644/continue_2/ |
| `nonsubmissions/171026/yolo26m_p2_finetune_329644/e5_spatial_head/E5_yolo26m_p2_spatial_head_329644.zip` | [curves](../nonsubmissions/171026/yolo26m_p2_finetune_329644/e5_spatial_head/E5_yolo26m_p2_spatial_head_329644.curves.png) |  |
| `nonsubmissions/171026/yolo26m_p2_quick_head_330230/yolo26m_p2_quick_head_330230.zip` | [curves](../nonsubmissions/171026/yolo26m_p2_quick_head_330230/yolo26m_p2_quick_head_330230.curves.png) |  |
| `submissions/E5_yolo26m_p2_328339/E5_yolo26m_p2_328339.zip` | [curves](../submissions/E5_yolo26m_p2_328339/E5_yolo26m_p2_328339.curves.png) |  |
| `submissions/E7_328598/E7_328598.zip` | [curves](../submissions/E7_328598/E7_328598.curves.png) |  |
| `submissions/longer_baseline_calibrated/longer_baseline_calibrated.zip` | [curves](../submissions/longer_baseline_calibrated/longer_baseline_calibrated.curves.png) |  |
| `submissions/longer_boxloss_calibrated/longer_boxloss_calibrated.zip` | [curves](../submissions/longer_boxloss_calibrated/longer_boxloss_calibrated.curves.png) |  |
| `submissions/longer_boxloss_photometric_calibrated/longer_boxloss_photometric_calibrated.zip` | [curves](../submissions/longer_boxloss_photometric_calibrated/longer_boxloss_photometric_calibrated.curves.png) |  |
| `submissions/phase3_yolo26m_joint_20261017_1516/phase3_yolo26m_joint_20261017_1516.zip` | [curves](../submissions/phase3_yolo26m_joint_20261017_1516/phase3_yolo26m_joint_20261017_1516.curves.png) |  |
| `submissions/submission_1/submission_1.zip` | [curves](../submissions/submission_1/submission_1.curves.png) | matched by recipe; the ZIP did not record its run |
| `submissions/terramind_48ep_boxloss_photometric_seed0_calibrated/terramind_48ep_boxloss_photometric_seed0_calibrated.zip` | [curves](../submissions/terramind_48ep_boxloss_photometric_seed0_calibrated/terramind_48ep_boxloss_photometric_seed0_calibrated.curves.png) |  |
| `submissions/terramind_48ep_default_anchors_uncertainty_calibrated/terramind_48ep_default_anchors_uncertainty_calibrated.zip` | [curves](../submissions/terramind_48ep_default_anchors_uncertainty_calibrated/terramind_48ep_default_anchors_uncertainty_calibrated.curves.png) |  |
| `submissions/terramind_48ep_gain_mean3_seed0/terramind_48ep_gain_mean3_seed0.zip` | [curves](../submissions/terramind_48ep_gain_mean3_seed0/terramind_48ep_gain_mean3_seed0.curves.png) |  |
| `submissions/terramind_64epoch_baseline_calibrated/terramind_64epoch_baseline_calibrated.zip` | [curves](../submissions/terramind_64epoch_baseline_calibrated/terramind_64epoch_baseline_calibrated.curves.png) |  |
| `submissions/terramind_pretrained_lora_seed0/terramind_pretrained_lora_seed0.zip` | [curves](../submissions/terramind_pretrained_lora_seed0/terramind_pretrained_lora_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_box_localization_seed0/terramind_pretrained_moelora_box_localization_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_box_localization_seed0/terramind_pretrained_moelora_box_localization_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_class_balance_seed0/terramind_pretrained_moelora_class_balance_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_class_balance_seed0/terramind_pretrained_moelora_class_balance_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_large_anchors_seed0/terramind_pretrained_moelora_large_anchors_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_large_anchors_seed0/terramind_pretrained_moelora_large_anchors_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_preserve_geometry_seed0/terramind_pretrained_moelora_preserve_geometry_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_preserve_geometry_seed0/terramind_pretrained_moelora_preserve_geometry_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_seed0/terramind_pretrained_moelora_seed0.curves.png) |  |
| `submissions/terramind_pretrained_moelora_unfreeze2_seed0_epochs12/terramind_pretrained_moelora_unfreeze2_seed0_epochs12.zip` | [curves](../submissions/terramind_pretrained_moelora_unfreeze2_seed0_epochs12/terramind_pretrained_moelora_unfreeze2_seed0_epochs12.curves.png) |  |
| `submissions/terramind_pretrained_moelora_wide_anchors_seed0/terramind_pretrained_moelora_wide_anchors_seed0.zip` | [curves](../submissions/terramind_pretrained_moelora_wide_anchors_seed0/terramind_pretrained_moelora_wide_anchors_seed0.curves.png) |  |
| `submissions/terramind_small_extreme_calibrated/terramind_small_extreme_calibrated.zip` | [curves](../submissions/terramind_small_extreme_calibrated/terramind_small_extreme_calibrated.curves.png) |  |
| `submissions/tm_48ep_boxloss_photo_large_anchors_s0_cal/tm_48ep_boxloss_photo_large_anchors_s0_cal.zip` | [curves](../submissions/tm_48ep_boxloss_photo_large_anchors_s0_cal/tm_48ep_boxloss_photo_large_anchors_s0_cal.curves.png) |  |
| `submissions/vit_pretrained_moelora_seed0/vit_pretrained_moelora_seed0.zip` | [curves](../submissions/vit_pretrained_moelora_seed0/vit_pretrained_moelora_seed0.curves.png) |  |
| `submissions/yolo11s_100_p2_328058/yolo11s_100_p2_328058.zip` | [curves](../submissions/yolo11s_100_p2_328058/yolo11s_100_p2_328058.curves.png) |  |
| `submissions/yolo26m_p2_candidate_330230/yolo26m_p2_candidate_330230.zip` | [curves](../submissions/yolo26m_p2_candidate_330230/yolo26m_p2_candidate_330230.curves.png) | box-edge correction on the quick-head ZIP; no extra training |
| `submissions/yolo26m_p2_finetune_40ep_329644/yolo26m_p2_finetune_40ep_329644.zip` | [curves](../submissions/yolo26m_p2_finetune_40ep_329644/yolo26m_p2_finetune_40ep_329644.curves.png) |  |
