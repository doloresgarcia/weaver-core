import wandb
import numpy as np

def log_wandb_init(args):
    """log information about the run in the config section of wandb
    Currently wandb is only initialized in training mode

    Args:
        args (_type_): parsed args from training
    """    
    if args.regression_mode:
        wandb.config.regression_mode = True
    else:
        wandb.config.classification_mode = True
    wandb.config.num_epochs = args.num_epochs

def log_confussion_matrix_wandb(y_true, y_score, epoch):
    """function to log confussion matrix in the wandb.ai website 

    Args:
        y_true (_type_): labels (B,)
        y_score (_type_): probabilities (B,num_classes)
        epoch (_type_): epoch of training so that maybe we can build slider in wandb
    """    
    if y_score.ndim == 1:
        y_pred = y_score > 0.5
    else:
        y_pred = y_score.argmax(1)
    cm = wandb.plot.confusion_matrix(y_score, y_true=y_true)
    wandb.log({'confussion matrix': cm})
    # we could also log multiple cm during training but no sliding for now.