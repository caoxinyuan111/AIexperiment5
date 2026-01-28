import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import re



class MultimodalDataset(Dataset):
    def __init__(self, data_root, label_file, transform, tokenizer, max_len=128, mode='train'):
        self.data_path = os.path.join(data_root, 'data')
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.df = pd.read_csv(label_file)
        self.label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row['guid'])
        
        # 1. 文本
        txt_path = os.path.join(self.data_path, f"{guid}.txt")
        text = ""
        try:
            with open(txt_path, 'r', encoding='gb18030', errors='ignore') as f:
                text = f.read().strip()
        except:
            text = ""
        
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_len, padding='max_length', truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        mask = inputs['attention_mask'].squeeze(0)
        
        # 2. 图像
        img_path = os.path.join(self.data_path, f"{guid}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'test':
            return input_ids, mask, image, guid
        else:
            label = self.label_map.get(row['tag'], 1)
            return input_ids, mask, image, torch.tensor(label)