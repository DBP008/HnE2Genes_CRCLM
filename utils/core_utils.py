import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
from PIL import Image
import h5py

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_attention_heatmap(attention_scores, grid_size=None):
    """
    Create a heatmap visualization from attention scores
    
    Args:
        attention_scores: torch.Tensor of attention scores (N,) where N is number of patches
        grid_size: tuple (height, width) for arranging patches in grid. If None, infers square grid
    
    Returns:
        PIL Image of the heatmap
    """
    attention_np = attention_scores.cpu().numpy().squeeze()
    
    # Handle edge case where all attention scores are the same
    if np.max(attention_np) == np.min(attention_np):
        attention_np = np.ones_like(attention_np) * 0.5
    else:
        # Normalize attention scores to [0, 1]
        attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
    
    # If grid_size not provided, assume square grid
    if grid_size is None:
        n_patches = len(attention_np)
        grid_h = grid_w = int(np.ceil(np.sqrt(n_patches)))
    else:
        grid_h, grid_w = grid_size
    
    # Pad attention scores if necessary
    total_patches = grid_h * grid_w
    if len(attention_np) < total_patches:
        padded_attention = np.zeros(total_patches)
        padded_attention[:len(attention_np)] = attention_np
        attention_np = padded_attention
    elif len(attention_np) > total_patches:
        attention_np = attention_np[:total_patches]
    
    # Reshape to grid
    attention_grid = attention_np.reshape(grid_h, grid_w)
    
    # Create heatmap with proper figure handling
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attention_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Attention Score', fraction=0.046, pad=0.04)
    ax.set_title('Attention Heatmap', fontsize=14)
    ax.axis('off')
    
    # Convert plot to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Explicitly close the figure to free memory
    
    return img

def create_spatial_attention_heatmap(attention_scores, coords, patch_size=256, downsample_factor=32, slide_name=None):
    """
    Create a spatial heatmap visualization from attention scores and patch coordinates
    
    Args:
        attention_scores: torch.Tensor of attention scores (N,) where N is number of patches
        coords: numpy array of patch coordinates (N, 2) - (x, y) coordinates
        patch_size: size of each patch in pixels
        downsample_factor: factor to downsample the final heatmap for visualization
        slide_name: name of the slide for title display
    
    Returns:
        PIL Image of the spatial heatmap
    """
    attention_np = attention_scores.cpu().numpy().squeeze()
    
    # Handle edge case where all attention scores are the same
    if np.max(attention_np) == np.min(attention_np):
        attention_np = np.ones_like(attention_np) * 0.5
    else:
        # Normalize attention scores to [0, 1]
        attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
    
    # Calculate the extent of the WSI based on coordinates
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # Create a canvas size based on the coordinate extent
    canvas_width = int((max_x - min_x + patch_size) // downsample_factor)
    canvas_height = int((max_y - min_y + patch_size) // downsample_factor)
    
    # Initialize canvas
    attention_canvas = np.zeros((canvas_height, canvas_width))
    count_canvas = np.zeros((canvas_height, canvas_width))
    
    # Calculate patch size on canvas
    patch_canvas_size = max(1, patch_size // downsample_factor)
    
    # Map each patch to its position on the canvas
    for i, (x, y) in enumerate(coords):
        # Convert coordinates to canvas coordinates
        canvas_x = int((x - min_x) // downsample_factor)
        canvas_y = int((y - min_y) // downsample_factor)
        
        # Handle boundary conditions
        end_x = min(canvas_x + patch_canvas_size, canvas_width)
        end_y = min(canvas_y + patch_canvas_size, canvas_height)
        
        # Add attention score to canvas - fill the entire patch area
        attention_canvas[canvas_y:end_y, canvas_x:end_x] += attention_np[i]
        count_canvas[canvas_y:end_y, canvas_x:end_x] += 1
    
    # Average overlapping regions
    mask = count_canvas > 0
    attention_canvas[mask] /= count_canvas[mask]
    
    # Apply Gaussian smoothing to reduce pixelation
    try:
        from scipy import ndimage
        # Smooth the attention map to create more continuous appearance
        sigma = max(1.0, patch_canvas_size / 3.0)  # Adaptive smoothing based on patch size
        attention_canvas = ndimage.gaussian_filter(attention_canvas, sigma=sigma)
    except ImportError:
        # Fallback if scipy is not available - manual smoothing
        print("Warning: scipy not available, using basic smoothing")
        # Simple manual smoothing by averaging with neighbors
        if patch_canvas_size > 1:
            # Create a simple kernel for smoothing
            kernel_size = max(3, int(patch_canvas_size))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make sure kernel size is odd
            
            # Pad the array for convolution
            pad_size = kernel_size // 2
            padded_canvas = np.pad(attention_canvas, pad_size, mode='constant', constant_values=0)
            smoothed_canvas = np.zeros_like(attention_canvas)
            
            # Manual convolution with uniform kernel
            for i in range(attention_canvas.shape[0]):
                for j in range(attention_canvas.shape[1]):
                    window = padded_canvas[i:i+kernel_size, j:j+kernel_size]
                    smoothed_canvas[i, j] = np.mean(window[window > 0]) if np.any(window > 0) else 0
            
            attention_canvas = smoothed_canvas
    
    # Create custom blue-to-red colormap
    colors = ['#000080', '#0000FF', '#4169E1', '#87CEEB', '#FFFFFF', '#FFB6C1', '#FF6347', '#FF0000', '#8B0000']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('blue_red', colors, N=n_bins)
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use better interpolation to smooth the visualization
    im = ax.imshow(attention_canvas, cmap=cmap, interpolation='bicubic', origin='upper', 
                   aspect='equal', vmin=0, vmax=1)
    
    # Create colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attention Score', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    
    # Set title with slide name if provided
    if slide_name:
        title = f'Spatial Attention Heatmap - {slide_name}'
    else:
        title = 'Spatial Attention Heatmap'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Customize axes
    ax.set_xlabel('WSI Width (downsampled)', fontsize=14)
    ax.set_ylabel('WSI Height (downsampled)', fontsize=14)
    ax.tick_params(labelsize=12)
    
    # Add grid for better reference (subtle)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Tight layout to prevent clipping
    plt.tight_layout()
    
    # Convert plot to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Explicitly close the figure to free memory
    
    return img

def get_coords_from_slide_data(slide_data, batch_idx, data_dir):
    """
    Get coordinates for a specific slide from slide data
    
    Args:
        slide_data: DataFrame containing slide information
        batch_idx: index of the current batch/slide
        data_dir: directory containing h5 files
    
    Returns:
        numpy array of coordinates or None if not found
    """
    try:
        slide_id = slide_data['slide_id'].iloc[batch_idx]
        h5_path = os.path.join(data_dir, 'h5_files', f'{slide_id}.h5')
        
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                coords = f['coords'][:]
                return coords
        else:
            # Try alternative path structure
            h5_path = os.path.join(data_dir, f'{slide_id}.h5')
            if os.path.exists(h5_path):
                with h5py.File(h5_path, 'r') as f:
                    coords = f['coords'][:]
                    return coords
    except Exception as e:
        print(f"Warning: Could not load coordinates for slide {batch_idx}: {e}")
        return None
    
    return None

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if epoch < self.stop_epoch:
                print(f'EarlyStopping: epoch {epoch} is below stop epoch {self.stop_epoch}, not stopping.')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    
    # Flags to capture attention and coordinates from first sample for visualization
    first_sample_attention = None
    first_sample_coords = None
    
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)      
            logits, Y_prob, Y_hat, A_raw, instance_dict = model(data, label=label, instance_eval=True)
            
            # Capture attention and coordinates from first validation sample for TensorBoard visualization
            if batch_idx == 0 and writer is not None and isinstance(model, CLAM_SB) and n_classes == 2:
                first_sample_attention = A_raw.clone()
                # Try to get coordinates from the dataset
                if hasattr(loader.dataset, 'slide_data') and hasattr(loader.dataset, 'data_dir'):
                    first_sample_coords = get_coords_from_slide_data(
                        loader.dataset.slide_data, batch_idx, loader.dataset.data_dir
                    )
            
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        
        # Add attention heatmap visualization for CLAM_SB binary classification
        if first_sample_attention is not None and isinstance(model, CLAM_SB) and n_classes == 2:
            try:
                # For CLAM_SB, A_raw has shape (1, N) where N is number of patches
                attention_scores = first_sample_attention.squeeze(0)  # Remove batch dimension, shape: (N,)
                
                # Get slide name for title
                slide_name = None
                if hasattr(loader.dataset, 'slide_data'):
                    try:
                        slide_name = loader.dataset.slide_data['slide_id'].iloc[0]
                    except:
                        pass
                
                # Create spatial heatmap if coordinates are available and match attention scores
                if first_sample_coords is not None and len(first_sample_coords) == len(attention_scores):
                    print(f"Creating spatial attention heatmap with {len(attention_scores)} patches and {len(first_sample_coords)} coordinates")
                    heatmap_img = create_spatial_attention_heatmap(attention_scores, first_sample_coords, slide_name=slide_name)
                    heatmap_tag = 'val/spatial_attention_heatmap'
                elif first_sample_coords is not None:
                    print(f"Warning: Coordinate count ({len(first_sample_coords)}) doesn't match attention count ({len(attention_scores)}), using grid layout")
                    heatmap_img = create_attention_heatmap(attention_scores)
                    heatmap_tag = 'val/grid_attention_heatmap'
                else:
                    print(f"Creating grid attention heatmap with {len(attention_scores)} patches")
                    heatmap_img = create_attention_heatmap(attention_scores)
                    heatmap_tag = 'val/grid_attention_heatmap'
                
                # Convert PIL image to numpy array for TensorBoard
                heatmap_np = np.array(heatmap_img)
                # Convert to format expected by TensorBoard: (C, H, W)
                if len(heatmap_np.shape) == 3:
                    heatmap_np = np.transpose(heatmap_np, (2, 0, 1))
                
                writer.add_image(heatmap_tag, heatmap_np, epoch)
                
            except Exception as e:
                print(f"Warning: Could not create attention heatmap visualization: {e}")


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
