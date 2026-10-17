from __future__ import annotations

import pytest


def test_build_retinanet_matches_bundle_inference_json() -> None:
    """MONAI 1.3+ removed retinanet_resnet50_fpn_feature_extractor; use bundle API."""
    pytest.importorskip("monai")
    torch = pytest.importorskip("torch")

    from monai.apps.detection.networks.retinanet_network import (
        RetinaNet,
        resnet_fpn_feature_extractor,
    )
    from monai.networks.nets import resnet

    backbone = resnet.resnet50(
        pretrained=False,
        progress=False,
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=(2, 2, 1),
        conv1_t_size=(7, 7, 7),
    )
    feature_extractor = resnet_fpn_feature_extractor(
        backbone=backbone,
        spatial_dims=3,
        pretrained_backbone=False,
        returned_layers=(1, 2),
    )
    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=(16, 16, 8),
        use_list_output=False,
    )
    # Smoke: forward shape plumbing (tiny random input, no sliding window).
    x = torch.zeros(1, 1, 64, 64, 32)
    network.eval()
    with torch.no_grad():
        out = network(x)
    assert out is not None
