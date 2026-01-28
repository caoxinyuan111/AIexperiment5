import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer
# [修正点1] 修复报错，使用 PyTorch 官方优化器
from torch.optim import AdamW 
from src.dataset import MultimodalDataset
from src.model import FinalModel
from tqdm import tqdm
import pandas as pd



# === [修改点2] 工业级参数配置 ===
BATCH_SIZE = 16
EPOCHS = 20        # 给它足够的时间跑
PATIENCE = 3       # 早停耐心值：连续3轮没提升就停
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_task(task_name, fusion_type, mode, train_loader, val_loader):
    """
    运行单个任务（带早停机制）
    """
    print(f"\n{'='*10} Run Task: {task_name} [Type={fusion_type}, Mode={mode}] {'='*10}")
    
    model = FinalModel(fusion_type=fusion_type).to(DEVICE)
    
    # 优化器配置
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': LR},
        {'params': model.resnet_base.parameters(), 'lr': LR},
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        # 只有 interactive 模式才有 cross_attn
        {'params': getattr(model, 'cross_attn', nn.Module()).parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # === 早停相关变量 ===
    best_acc = 0.0
    patience_counter = 0
    # 为每个任务保存一个独立的最佳模型文件
    temp_model_path = f'temp_best_{task_name}.pth'
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        for input_ids, mask, imgs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}"):
            input_ids, mask, imgs, labels = input_ids.to(DEVICE), mask.to(DEVICE), imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, mask, imgs, mode=mode)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Val ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, mask, imgs, labels in val_loader:
                input_ids, mask, imgs, labels = input_ids.to(DEVICE), mask.to(DEVICE), imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(input_ids, mask, imgs, mode=mode)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f" -> Loss: {avg_train_loss:.4f} | Val Acc: {acc:.4f}")
        
        # === [修改点3] 早停逻辑 ===
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0 # 重置计数器
            torch.save(model.state_dict(), temp_model_path)
            # print("    [Save] New Best Model!")
        else:
            patience_counter += 1
            print(f"    [Info] No improve. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"    [Early Stop] Triggered at epoch {epoch+1}!")
                break
    
    print(f"Task {task_name} Best Acc: {best_acc:.4f}")
    
    # 如果是最终的主模型任务，要把最佳权重留下来给后面预测用
    if task_name == 'Interactive_Fusion':
        if os.path.exists(temp_model_path):
            os.rename(temp_model_path, 'best_model_final.pth')
    else:
        # 其他对比任务产生的临时文件删掉，省空间
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
            
    return best_acc

def generate_prediction(test_loader, id_map):
    print("\nGenerating final prediction file...")
    # 加载 Interactive Fusion 模式
    model = FinalModel(fusion_type='interactive').to(DEVICE)
    # 加载早停机制保存下来的那个最好的权重
    model.load_state_dict(torch.load('best_model_final.pth'))
    model.eval()
    
    results = []
    with torch.no_grad():
        for input_ids, mask, imgs, guids in tqdm(test_loader):
            input_ids, mask, imgs = input_ids.to(DEVICE), mask.to(DEVICE), imgs.to(DEVICE)
            # 这里 mode 必须是 fusion
            outputs = model(input_ids, mask, imgs, mode='fusion')
            _, predicted = torch.max(outputs, 1)
            for guid, pred in zip(guids, predicted):
                results.append({'guid': guid, 'tag': id_map[pred.item()]})
                
    pd.DataFrame(results).to_csv('test_result.txt', index=False)
    print("Saved to test_result.txt")

if __name__ == '__main__':
    # 1. 准备数据
    print(f"Using Device: {DEVICE}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    full_ds = MultimodalDataset('data', 'data/train.txt', transform, tokenizer)
    
    # 80% 训练，20% 验证
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 2. 依次运行4个任务 (自动对比 + 自动消融)
    
    # (1) 基线: Late Fusion
    acc_baseline = run_task('Late_Fusion', 'late', 'fusion', train_loader, val_loader)
    
    # (2) 消融: 仅文本
    acc_text = run_task('Interactive_Text', 'interactive', 'text_only', train_loader, val_loader)
    
    # (3) 消融: 仅图像
    acc_image = run_task('Interactive_Image', 'interactive', 'image_only', train_loader, val_loader)
    
    # (4) 主角: Interactive Fusion (这个结果用来生成 test_result.txt)
    acc_final = run_task('Interactive_Fusion', 'interactive', 'fusion', train_loader, val_loader)
    
    # 3. 打印 完美报表
    print("\n" + "="*50)
    print(" REPORT TABLE 1: MODEL COMPARISON (模型对比)")
    print("="*50)
    print(f"| Model Architecture  | Best Val Accuracy |")
    print(f"|---------------------|-------------------|")
    print(f"| Late Fusion (Base)  | {acc_baseline:.4f}            |")
    print(f"| Interactive (Ours)  | {acc_final:.4f}            |")
    print("="*50)
    
    print("\n" + "="*50)
    print(" REPORT TABLE 2: ABLATION STUDY (消融实验)")
    print(" (Based on Interactive Fusion Model)")
    print("="*50)
    print(f"| Input Modality      | Best Val Accuracy |")
    print(f"|---------------------|-------------------|")
    print(f"| Text Only           | {acc_text:.4f}            |")
    print(f"| Image Only          | {acc_image:.4f}            |")
    print(f"| Multimodal (Fusion) | {acc_final:.4f}            |")
    print("="*50 + "\n")
    
    # 4. 预测
    test_ds = MultimodalDataset('data', 'data/test_without_label.txt', transform, tokenizer, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    id_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    
    generate_prediction(test_loader, id_map)