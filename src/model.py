import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights

class FinalModel(nn.Module):
    def __init__(self, num_classes=3, fusion_type='late'):
        """
        fusion_type: 'late' (基线) | 'interactive' (创新)
        """
        super(FinalModel, self).__init__()
        self.fusion_type = fusion_type
        
        # 1. 骨干网络 (共享)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(768, 256)
        
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet_base = nn.Sequential(*list(resnet.children())[:-1])
        self.img_proj = nn.Linear(2048, 256)
        
        # 2. 交互层 (只在 interactive 模式下有)
        if self.fusion_type == 'interactive':
            self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(256)
            
        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, mask, images, mode='fusion'):
        """
        mode: 'fusion' | 'text_only' | 'image_only'
        """
        # --- A. 特征提取 ---
        bert_out = self.bert(input_ids=input_ids, attention_mask=mask)
        text_feat = self.text_proj(bert_out.last_hidden_state) # [B, Seq, 256]
        
        img_raw = self.resnet_base(images).squeeze()
        if len(img_raw.shape) == 1: img_raw = img_raw.unsqueeze(0)
        img_feat = self.img_proj(img_raw) # [B, 256]
        
        # --- B. 消融遮蔽 (Ablation Masking) ---
        if mode == 'text_only':
            img_feat = torch.zeros_like(img_feat) # 屏蔽图像
        elif mode == 'image_only':
            text_feat = torch.zeros_like(text_feat) # 屏蔽文本

        # --- C. 融合逻辑 ---
        if self.fusion_type == 'late':
            # === 1. Late Fusion ===
            text_cls = text_feat[:, 0, :] # 取 [CLS]
            combined = torch.cat([text_cls, img_feat], dim=1)
            
        elif self.fusion_type == 'interactive':
            # === 2. Interactive Fusion ===
            img_seq = img_feat.unsqueeze(1) # [B, 1, 256]
            
            # Cross Attention: Text查Image
            # 如果 mode='text_only'，img_seq全是0，查出来也是0，符合逻辑
            # 如果 mode='image_only'，text_feat全是0，查出来也是0，符合逻辑
            attn_out, _ = self.cross_attn(query=text_feat, key=img_seq, value=img_seq)
            text_interacted = self.norm(text_feat + attn_out)
            
            text_cls = text_interacted[:, 0, :]
            combined = torch.cat([text_cls, img_feat], dim=1)
            
        output = self.classifier(combined)
        return output