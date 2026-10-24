import argparse

import numpy as np
import timm
import torch

from src.eyelid.visionfm import load_encoder


def test_load_encoder_maps_teacher_backbone_safely(tmp_path):
    # Same layout as the VisionFM checkpoint: iBOT teacher with backbone + head keys, plus pickled args and a numpy scalar.
    ref = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    teacher = {f"backbone.{k}": v for k, v in ref.state_dict().items()}
    teacher["backbone.masked_embed"] = torch.zeros(192)
    teacher["head.last_layer.weight_g"] = torch.ones(1)
    path = tmp_path / "w.pth"
    torch.save({"teacher": teacher, "args": argparse.Namespace(arch="vit_tiny"), "loss": np.float64(0.5)}, path)

    model = load_encoder({"weights": str(path), "arch": "vit_tiny_patch16_224"})
    assert torch.equal(model.pos_embed, ref.pos_embed)
    assert torch.equal(model.blocks[0].attn.qkv.weight, ref.blocks[0].attn.qkv.weight)
