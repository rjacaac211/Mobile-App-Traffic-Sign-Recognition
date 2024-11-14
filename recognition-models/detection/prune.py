import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.50):
    for module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

# Load model
model = YOLO('./runs/detect/tsd_model_7/weights/best.pt')

# Evaluate model
results = model.val(data="../TSR_data/Detection/Traffic-Sign-Detection-15/data.yaml")
print(f"mAP50-95: {results.box.map}")

# Prune model
torch_model = model.model

print("Pruning model...")
pruned_torch_model = prune_model(torch_model, amount=0.50)
print("Model pruned.")

model.model = pruned_torch_model

print("Saving pruned model...")

model.save('./runs/detect/tsd_model_7/weights/best_pruned.pt')

print("Pruned model saved.")

# Evaluate pruned model

model = YOLO('./runs/detect/tsd_model_7/weights/best_pruned.pt')
results = model.val(data="../TSR_data/Detection/Traffic-Sign-Detection-15/data.yaml")
print(f"mAP50-95: {results.box.map}")