import os
from typing import Dict, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(in_ch, out_ch), ConvBNReLU(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, pretrained_path: Optional[str] = None):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)
        self.layer4 = self._make_layer(512, 3, 2)
        self._init_weights()
        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained(self, pretrained_path: str) -> None:
        if not pretrained_path or not os.path.exists(pretrained_path):
            return
        state = torch.load(pretrained_path, map_location="cpu")
        if isinstance(state, dict) and "encoder" in state:
            state = state["encoder"]
        elif isinstance(state, dict) and "model" in state:
            model_state = state["model"]
            state = {k[len("encoder."):]: v for k, v in model_state.items() if k.startswith("encoder.")} or model_state
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        current = self.state_dict()
        adapted = {}
        for key, value in cleaned.items():
            if key not in current:
                continue
            if key == "conv1.weight" and value.ndim == 4 and value.shape[1] == 3 and current[key].shape[1] == 1:
                adapted[key] = value.mean(dim=1, keepdim=True)
            elif value.shape == current[key].shape:
                adapted[key] = value
        self.load_state_dict(adapted, strict=False)

    def forward(self, x: torch.Tensor):
        f1 = self.relu(self.bn1(self.conv1(x)))
        f2 = self.layer1(self.maxpool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return f1, f2, f3, f4, f5


class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch: int = 512, token_ch: int = 256, layers: int = 4, heads: int = 8):
        super().__init__()
        self.proj = ConvBNReLU(in_ch, token_ch, 1, 1, 0)
        enc = nn.TransformerEncoderLayer(
            d_model=token_ch,
            nhead=heads,
            dim_feedforward=token_ch * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                self.branches.append(ConvBNReLU(in_ch, out_ch, 1, 1, 0))
            else:
                self.branches.append(ConvBNReLU(in_ch, out_ch, 3, 1, rate, dilation=rate))
        self.project = ConvBNReLU(len(rates) * out_ch, out_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, mid_ch: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(g_ch, mid_ch, 1), nn.BatchNorm2d(mid_ch))
        self.w_x = nn.Sequential(nn.Conv2d(x_ch, mid_ch, 1), nn.BatchNorm2d(mid_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(mid_ch, 1, 1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(self.w_g(g) + self.w_x(x))


class UpBlockAttn(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, max(16, out_ch // 2))
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attn(x, skip)], dim=1))


class ApexAwareGatedFusion(nn.Module):
    def __init__(self, context_ch: int = 256, out_ch: int = 64):
        super().__init__()
        self.p1 = ConvBNReLU(64, out_ch, 1, 1, 0)
        self.p2 = ConvBNReLU(64, out_ch, 1, 1, 0)
        self.p3 = ConvBNReLU(128, out_ch, 1, 1, 0)
        self.pc = ConvBNReLU(context_ch, out_ch, 1, 1, 0)
        self.gate = nn.Sequential(ConvBNReLU(out_ch * 4, out_ch), nn.Conv2d(out_ch, 4, 1))
        self.apex_prior = nn.Sequential(ConvBNReLU(out_ch * 4, out_ch), nn.Conv2d(out_ch, 1, 1), nn.Sigmoid())
        self.fuse = DoubleConv(out_ch * 2, out_ch)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, context: torch.Tensor):
        target = f2.shape[-2:]
        x1 = F.interpolate(self.p1(f1), size=target, mode="bilinear", align_corners=False)
        x2 = self.p2(f2)
        x3 = F.interpolate(self.p3(f3), size=target, mode="bilinear", align_corners=False)
        xc = F.interpolate(self.pc(context), size=target, mode="bilinear", align_corners=False)
        stacked = torch.cat([x1, x2, x3, xc], dim=1)
        weights = torch.softmax(self.gate(stacked), dim=1)
        fused = weights[:, 0:1] * x1 + weights[:, 1:2] * x2 + weights[:, 2:3] * x3 + weights[:, 3:4] * xc
        apex = self.apex_prior(stacked)
        return self.fuse(torch.cat([fused, fused * apex], dim=1)), apex, weights


class ApexTransNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        encoder_pretrained_path: Optional[str] = None,
        transformer_layers: int = 4,
        use_boundary_head: bool = True,
        use_location_head: bool = True,
    ):
        super().__init__()
        self.use_boundary_head = use_boundary_head
        self.use_location_head = use_location_head
        self.encoder = ResNet34Encoder(in_channels, encoder_pretrained_path)
        self.transformer = TransformerBottleneck(512, 256, transformer_layers)
        self.aspp = ASPP(256, 256)
        self.apex_fusion = ApexAwareGatedFusion(256, 64)
        self.up4 = UpBlockAttn(256, 256, 256)
        self.up3 = UpBlockAttn(256, 128, 128)
        self.up2 = UpBlockAttn(128, 64, 64)
        self.up1 = UpBlockAttn(64, 64, 64)
        self.seg_refine = DoubleConv(64 + 1, 64)
        self.seg_head = nn.Conv2d(64, 1, 1)
        if use_location_head:
            self.loc_conv = DoubleConv(64 + 128, 64)
            self.loc_head = nn.Conv2d(64, 1, 1)
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))
        if use_boundary_head:
            self.boundary_head = nn.Sequential(ConvBNReLU(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f1, f2, f3, f4, f5 = self.encoder(x)
        trans = self.transformer(f5)
        context = self.aspp(trans)
        gated_context, apex_prior, fusion_weights = self.apex_fusion(f1, f2, f3, context)
        d4 = self.up4(context, f4)
        d3 = self.up3(d4, f3)
        d2 = self.up2(d3, f2) + gated_context
        d1 = self.up1(d2, f1)
        full = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        out = {
            "seg_feature": full,
            "seg_logits": None,
            "cls_logits": self.cls_head(context),
            "apex_prior": F.interpolate(apex_prior, size=full.shape[-2:], mode="bilinear", align_corners=False),
            "fusion_weights": F.interpolate(fusion_weights, size=full.shape[-2:], mode="bilinear", align_corners=False),
        }
        if self.use_location_head:
            loc_feat = torch.cat([F.interpolate(d3, size=d2.shape[-2:], mode="bilinear", align_corners=False), d2], dim=1)
            loc_feat = self.loc_conv(loc_feat)
            loc_feat = F.interpolate(loc_feat, size=full.shape[-2:], mode="bilinear", align_corners=False)
            out["loc_logits"] = self.loc_head(loc_feat)
            anatomy_gate = torch.sigmoid(out["loc_logits"])
        else:
            anatomy_gate = out["apex_prior"]
        seg_feat = self.seg_refine(torch.cat([full, anatomy_gate], dim=1))
        out["seg_logits"] = self.seg_head(seg_feat)
        if self.use_boundary_head:
            out["boundary_logits"] = self.boundary_head(seg_feat)
        return out
