import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           f1_score, accuracy_score, cohen_kappa_score,
                           confusion_matrix)
import timm
import copy
import warnings
import argparse
import logging
import pandas as pd
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False", category=FutureWarning)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
EPOCHS = 50
LOCAL_EPOCHS = 5
PATIENCE = 5
CHECKPOINT_PATH = './checkpoints_federated'
CONF_RESULTS_PATH = './conf_results'
WEIGHT_DECAY = 0.01

HOSPITAL_DATASETS = {
    'Hospital_1': [
        'datasets/JSIEC',
        'datasets/IDRiD',
        'datasets/APTOS2019',
        'datasets/MESSIDOR2'
    ],
    'Hospital_2': [
        'datasets/IDRiD',
        'datasets/PAPILA',
        'datasets/Glaucoma_fundus'
    ],
    'Hospital_3': [
        'datasets/JSIEC',
        'datasets/IDRiD',
        'datasets/OCTID',
        'datasets/Retina'
    ]
}

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(CONF_RESULTS_PATH, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_learning_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FineTuneDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None):
        self.root_dir = os.path.join(dataset_path, split)
        self.transform = transform
        self.dataset_name = os.path.basename(dataset_path)
        
        jsiec_path = os.path.join('datasets', 'JSIEC', 'train')
        self.classes = sorted(os.listdir(jsiec_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.class_mappings = {
            'JSIEC': {cls: cls for cls in self.classes},
            'APTOS2019': {
                'anodr': '0.0.Normal',
                'bmilddr': '0.3.DR1',
                'cmoderatedr': '1.0.DR2',
                'dseveredr': '1.1.DR3',
                'eproliferativedr': '29.1.Blur fundus with suspected PDR'
            },
            'MESSIDOR2': {
                'anodr': '0.0.Normal',
                'bmilddr': '0.3.DR1',
                'cmoderatedr': '1.0.DR2',
                'dseveredr': '1.1.DR3',
                'eproliferativedr': '29.1.Blur fundus with suspected PDR'
            },
            'IDRiD': {
                'anoDR': '0.0.Normal',
                'bmildDR': '0.3.DR1',
                'cmoderateDR': '1.0.DR2',
                'dsevereDR': '1.1.DR3',
                'eproDR': '29.1.Blur fundus with suspected PDR'
            },
            'PAPILA': {
                'anormal': '0.0.Normal',
                'bsuspectglaucoma': '10.0.Possible glaucoma',
                'cglaucoma': '10.1.Optic atrophy'
            },
            'Glaucoma_fundus': {
                'anormal_control': '0.0.Normal',
                'bearly_glaucoma': '10.0.Possible glaucoma',
                'cadvanced_glaucoma': '10.1.Optic atrophy'
            },
            'OCTID': {
                'ANormal': '0.0.Normal',
                'CSR': '5.0.CSCR',
                'Diabetic_retinopathy': '1.0.DR2',
                'Macular_Hole': '8.MH',
                'ARMD': '6.Maculopathy'
            },
            'Retina': {
                'anormal': '0.0.Normal',
                'cglaucoma': '10.1.Optic atrophy',
                'bcataract': '29.0.Blur fundus without PDR',
                'ddretina_disease': '6.Maculopathy'
            }
        }
        
        self.samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        
        if self.dataset_name in self.class_mappings:
            mapping = self.class_mappings[self.dataset_name]
            for class_name in os.listdir(self.root_dir):
                if class_name in mapping:
                    mapped_class = mapping[class_name]
                    class_dir = os.path.join(self.root_dir, class_name)
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(valid_extensions):
                            img_path = os.path.join(class_dir, img_name)
                            if os.path.isfile(img_path):
                                self.samples.append((img_path, self.class_to_idx[mapped_class]))
        else:
            logger.warning(f"Dataset {self.dataset_name} not in mapping, skipping")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomAdjustSharpness(2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform

def get_class_mapping_indices(dataset_name):
    dataset = FineTuneDataset(
        dataset_path=os.path.join('datasets', dataset_name),
        split='train'
    )
    
    dataset_classes = sorted(os.listdir(os.path.join('datasets', dataset_name, 'train')))
    dataset_to_jsiec_idx = dict()
    mapping = dataset.class_mappings[dataset_name]
    
    for idx, orig_class in enumerate(dataset_classes):
        if orig_class in mapping:
            jsiec_class = mapping[orig_class]
            jsiec_idx = dataset.class_to_idx[jsiec_class]
            dataset_to_jsiec_idx[idx] = jsiec_idx
    
    return dataset_to_jsiec_idx

def get_inverse_mapping(dataset_name):
    dataset = FineTuneDataset(
        dataset_path=os.path.join('datasets', dataset_name),
        split='train'
    )
    
    dataset_classes = sorted(os.listdir(os.path.join('datasets', dataset_name, 'train')))
    jsiec_to_dataset = {}
    
    mapping = dataset.class_mappings[dataset_name]
    for orig_class in dataset_classes:
        if orig_class in mapping:
            jsiec_class = mapping[orig_class]
            jsiec_idx = dataset.class_to_idx[jsiec_class]
            local_idx = dataset_classes.index(orig_class)
            jsiec_to_dataset[jsiec_idx] = local_idx
    
    return jsiec_to_dataset

def create_vit_model(num_classes):
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    in_features = model.head.in_features
    
    model.head = nn.Sequential(
        nn.Dropout(0.3),
        nn.LayerNorm(in_features),
        nn.Linear(in_features, 2048),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(2048),
        nn.Linear(2048, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.LayerNorm(1024),
        nn.Linear(1024, num_classes)
    )
    
    for module in model.head:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model.to(DEVICE)

def load_initial_model(num_classes):
    model = create_vit_model(num_classes)
    pretrained_dict = torch.load('best_model.pth', map_location=DEVICE)
    model_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and 'head' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def create_hospital_datasets():
    train_transform, eval_transform = get_transforms()
    
    hospital_train_loaders = {}
    hospital_val_loaders = {}
    hospital_mappings = {}

    for hospital, dataset_paths in HOSPITAL_DATASETS.items():
        hospital_train_sets = []
        hospital_val_sets = []
        hospital_mapping = {}
        
        for path in dataset_paths:
            dataset_name = os.path.basename(path)
            try:
                class_mapping = get_class_mapping_indices(dataset_name)
                hospital_mapping.update(class_mapping)
                
                train_dataset = FineTuneDataset(path, 'train', train_transform)
                val_dataset = FineTuneDataset(path, 'val', eval_transform)
                
                if len(train_dataset.samples) > 0:
                    hospital_train_sets.append(train_dataset)
                if len(val_dataset.samples) > 0:
                    hospital_val_sets.append(val_dataset)
                    
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue
        
        if hospital_train_sets:
            hospital_train_loaders[hospital] = DataLoader(
                ConcatDataset(hospital_train_sets),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=8,
                prefetch_factor=2,
                pin_memory=True
            )
            
            if hospital_val_sets:
                hospital_val_loaders[hospital] = DataLoader(
                    ConcatDataset(hospital_val_sets),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=8,
                    prefetch_factor=2,
                    pin_memory=True
                )
            
            hospital_mappings[hospital] = hospital_mapping
    
    logger.info("\nDataset Statistics:")
    for hospital in hospital_train_loaders:
        train_size = len(hospital_train_loaders[hospital].dataset)
        val_size = len(hospital_val_loaders[hospital].dataset) if hospital in hospital_val_loaders else 0
        logger.info(f"{hospital}:")
        logger.info(f"  Train samples: {train_size}")
        logger.info(f"  Validation samples: {val_size}")
    
    return hospital_train_loaders, hospital_val_loaders, hospital_mappings

def aggregate_weights(local_weights, dataset_sizes):
    total_size = sum(dataset_sizes)
    weights = [size/total_size for size in dataset_sizes]
    
    aggregated_weights = {}
    for key in local_weights[0].keys():
        aggregated_weights[key] = torch.zeros_like(local_weights[0][key])
        for client_idx, client_weights in enumerate(local_weights):
            aggregated_weights[key] += client_weights[key] * weights[client_idx]
    
    return aggregated_weights

def update_hospital(model, dataloader, hospital_mapping):
    model.train()
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'head' not in n], 
         'lr': LEARNING_RATE * 0.1,
         'weight_decay': WEIGHT_DECAY},
        {'params': model.head.parameters(), 
         'lr': LEARNING_RATE,
         'weight_decay': WEIGHT_DECAY * 0.1}
    ])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(LOCAL_EPOCHS):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        logger.info(f"Local Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")
    
    return model.state_dict()

def evaluate_model(model, test_loader, label_mapping=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            if label_mapping:
                inv_mapping = {v: k for k, v in label_mapping.items()}
                mapped_preds = torch.tensor([inv_mapping[p.item()] for p in preds])
                preds = mapped_preds
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'confusion_matrix': cm
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError as e:
        logger.warning(f"Could not calculate AUC - {str(e)}")
        metrics['auc'] = 0.0
    
    n_classes = cm.shape[0]
    specificities = []
    
    for i in range(n_classes):
        tn = np.sum(np.delete(np.delete(cm, i, 0), i, 1))
        fp = np.sum(np.delete(cm[:, i], i))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    
    metrics['specificity'] = np.mean(specificities)
    return metrics

def test_on_dataset(model, dataset_name, model_path):
    logger.info(f"\nTesting on {dataset_name}...")
    
    _, eval_transform = get_transforms()
    
    test_dataset = FineTuneDataset(
        os.path.join('datasets', dataset_name),
        'test',
        eval_transform
    )
    
    jsiec_to_dataset = get_inverse_mapping(dataset_name)
    num_classes = len(set(jsiec_to_dataset.values()))
    
    test_model = create_vit_model(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'head.10.weight' in k:
            new_weights = torch.zeros((num_classes, v.size(1)), device=v.device)
            for jsiec_idx, dataset_idx in jsiec_to_dataset.items():
                if jsiec_idx < v.size(0):
                    new_weights[dataset_idx] = v[jsiec_idx]
            new_state_dict[k] = new_weights
        elif 'head.10.bias' in k:
            new_bias = torch.zeros(num_classes, device=v.device)
            for jsiec_idx, dataset_idx in jsiec_to_dataset.items():
                if jsiec_idx < v.size(0):
                    new_bias[dataset_idx] = v[jsiec_idx]
            new_state_dict[k] = new_bias
        else:
            new_state_dict[k] = v
    
    try:
        test_model.load_state_dict(new_state_dict)
    except Exception as e:
        logger.error(f"Failed to load state_dict from {model_path}: {str(e)}")
        raise
    
    test_model.to(DEVICE)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    metrics = evaluate_model(test_model, test_loader, jsiec_to_dataset)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                        index=[f"True_{i}" for i in range(len(metrics['confusion_matrix']))],
                        columns=[f"Pred_{i}" for i in range(len(metrics['confusion_matrix']))])
    cm_csv_path = os.path.join(CONF_RESULTS_PATH, f'confusion_matrix_{dataset_name}.csv')
    cm_df.to_csv(cm_csv_path)
    logger.info(f"Saved confusion matrix to {cm_csv_path}")
    
    logger.info(f"\n{dataset_name} Results:")
    for metric, value in metrics.items():
        if metric == 'confusion_matrix':
            logger.info(f"{metric}:\n{value}")
        else:
            logger.info(f"{metric}: {value*100:.2f}%")
    
    return metrics

def evaluate_all_datasets(model_path):
    logger.info("\nEvaluating all datasets...")
    results = {}
    
    for dataset in sorted(os.listdir('datasets')):
        dataset_path = os.path.join('datasets', dataset)
        if dataset != 'Combined' and os.path.isdir(dataset_path):
            if os.path.exists(os.path.join(dataset_path, 'test')):
                try:
                    metrics = test_on_dataset(None, dataset, model_path)
                    results[dataset] = metrics
                except Exception as e:
                    logger.error(f"Error testing {dataset}: {str(e)}")
                    continue
    
    logger.info("\n=== Final Results Summary ===")
    logger.info(f"{'Dataset':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    logger.info("-" * 80)
    
    for dataset, metrics in results.items():
        logger.info(f"{dataset:<20} "
              f"{metrics['accuracy']*100:>10.2f} "
              f"{metrics['precision']*100:>10.2f} "
              f"{metrics['recall']*100:>10.2f} "
              f"{metrics['f1']*100:>10.2f}")
    
    logger.info("-" * 80)
    
    if results:
        avg_metrics = {
            metric: np.mean([results[dataset][metric] for dataset in results])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa']
        }
        logger.info("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value*100:.2f}%")
    
    return results

def federated_training():
    logger.info("Starting federated learning across hospitals...")
    
    hospital_loaders, hospital_val_loaders, hospital_mappings = create_hospital_datasets()
    dataset_sizes = {hospital: len(loader.dataset) for hospital, loader in hospital_loaders.items()}
    num_classes = 39
    best_val_accs = {hospital: 0 for hospital in hospital_loaders.keys()}
    patience_counters = {hospital: 0 for hospital in hospital_loaders.keys()}
    global_best_acc = 0
    
    initial_model = load_initial_model(num_classes)
    global_weights = initial_model.state_dict()
    del initial_model
    
    for round in range(EPOCHS):
        logger.info(f"\nFederated Round {round+1}/{EPOCHS}")
        local_weights = []
        local_sizes = []
        early_stop = []
        
        for hospital, loader in hospital_loaders.items():
            logger.info(f"\nTraining {hospital} (Dataset size: {dataset_sizes[hospital]})")
            
            client_model = create_vit_model(num_classes=num_classes)
            client_model.load_state_dict(global_weights)
            
            client_weights = update_hospital(
                client_model, 
                loader,
                hospital_mappings[hospital]
            )
            
            if hospital in hospital_val_loaders:
                val_metrics = evaluate_model(client_model, hospital_val_loaders[hospital])
                val_acc = val_metrics['accuracy'] * 100
                logger.info(f"{hospital} Local Validation Accuracy: {val_acc:.2f}%")
                
                if val_acc > best_val_accs[hospital]:
                    best_val_accs[hospital] = val_acc
                    patience_counters[hospital] = 0
                else:
                    patience_counters[hospital] += 1
                
                if patience_counters[hospital] >= PATIENCE:
                    early_stop.append(hospital)
            
            local_weights.append(client_weights)
            local_sizes.append(dataset_sizes[hospital])
            
            del client_model
            torch.cuda.empty_cache()
        
        if len(early_stop) == len(hospital_loaders):
            logger.info("\nEarly stopping triggered for all hospitals!")
            break
        
        global_weights = aggregate_weights(local_weights, local_sizes)
        aggregated_path = os.path.join(CHECKPOINT_PATH, f'aggregated_model_round_{round+1}.pth')
        torch.save(global_weights, aggregated_path)
        
        avg_acc = 0
        test_results = evaluate_all_datasets(aggregated_path)
        if test_results:
            avg_acc = np.mean([metrics['accuracy'] for metrics in test_results.values()])
        
        if avg_acc > global_best_acc:
            global_best_acc = avg_acc
            best_model_path = os.path.join(CONF_RESULTS_PATH, f'best_federated_model_round_{round+1}.pth')
            torch.save(global_weights, best_model_path)
            logger.info(f"Saved new best model with average accuracy: {avg_acc*100:.2f}%")
    
    logger.info("\nFinal Best Local Validation Accuracies:")
    for hospital, acc in best_val_accs.items():
        logger.info(f"{hospital}: {acc:.2f}%")
        
    return global_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning for Medical Imaging')
    parser.add_argument('--test-only', action='store_true', help='Run in test-only mode')
    parser.add_argument('--model-path', type=str, default=os.path.join(CONF_RESULTS_PATH, 'best_federated_model_final.pth'))
    parser.add_argument('--dataset', type=str, help='Specific dataset to test')
    
    args = parser.parse_args()
    
    if args.test_only:
        if not os.path.exists(args.model_path):
            logger.error(f"Error: Model file {args.model_path} not found")
            exit(1)
        
        if args.dataset:
            test_on_dataset(None, args.dataset, args.model_path)
        else:
            evaluate_all_datasets(args.model_path)
    else:
        try:
            model = federated_training()
            best_model_path = os.path.join(CONF_RESULTS_PATH, 'best_federated_model_final.pth')
            torch.save(model, best_model_path)
            evaluate_all_datasets(best_model_path)
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise e
