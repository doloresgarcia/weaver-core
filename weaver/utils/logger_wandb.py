import wandb
import numpy as np
import torch
from sklearn.metrics import roc_curve

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


def log_roc_curves(y_true, y_score, epoch):
    
    # 5 classes G(0),Q(1),S(2),C(3),B(4)
    # b tagging  (b/g, b/ud, b/c)
    fpr_bg, tpr_bg = create_binary_rocs(4, 0, y_true, y_score)
    fpr_bud, tpr_bud = create_binary_rocs(4, 1, y_true, y_score)
    fpr_bc, tpr_bc = create_binary_rocs(4, 3, y_true, y_score)
    columns = ['b vs g', 'b vs ud', 'b vs c']
    xs = [tpr_bg, tpr_bud, tpr_bc]
    ys = [fpr_bg, fpr_bud, fpr_bc]
    title_log = "roc b"
    title_plot = "b tagging"
    wanb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)

    # c tagging (c/g, c/ud, c/b)
    fpr_cg, tpr_cg = create_binary_rocs(3, 0, y_true, y_score)
    fpr_cud, tpr_cud = create_binary_rocs(3, 1, y_true, y_score)
    fpr_cb, tpr_cb = create_binary_rocs(3, 4, y_true, y_score)
    columns = ['c vs g', 'c vs ud', 'c vs b']
    xs = [tpr_cg, tpr_cud, tpr_cb]
    ys = [fpr_cg, fpr_cud, fpr_cb]
    title_log = "roc c"
    title_plot = "c tagging"
    wanb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)

    # s tagging (s/g, s/ud, s/c, s/b)
    fpr_sg, tpr_sg = create_binary_rocs(2, 0, y_true, y_score)
    fpr_sud, tpr_sud = create_binary_rocs(2, 1, y_true, y_score)
    fpr_sb, tpr_sb = create_binary_rocs(2, 4, y_true, y_score)
    columns = ['s vs g', 's vs ud', 's vs b']
    xs = [tpr_sg, tpr_sud, tpr_sb]
    ys = [fpr_sg, fpr_sud, fpr_sb]
    title_log = "roc s"
    title_plot = "s tagging"
    wanb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)

    # g tagging (g/ud, g/s, g/c, g/b)
    fpr_gud, tpr_gud = create_binary_rocs(0, 1, y_true, y_score)
    fpr_gs, tpr_gs = create_binary_rocs(0, 2, y_true, y_score)
    fpr_gc, tpr_gc = create_binary_rocs(0, 3, y_true, y_score)
    fpr_gb, tpr_gb = create_binary_rocs(0, 4, y_true, y_score)
    columns = ['g vs ud', 'g vs s', 'g vs c', 'g vs b']
    xs = [tpr_gud, tpr_gs, tpr_gc, tpr_gb]
    ys = [fpr_gud, fpr_gs, fpr_gc, fpr_gb]
    title_log = "roc g"
    title_plot = "g tagging"
    wanb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
    
    


def wanb_log_multiline_rocs(xs, ys, title_log, title_plot, columns):
    wandb.log({title_log : wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=columns,
        title=title_plot, 
        xname="jet tagging efficiency")})


def create_binary_rocs(positive, negative, y_true, y_score):
    mask_positive = y_true==positive
    mask_negative = y_true==negative 
    number_positive = len(y_true[mask_positive])
    number_negative = len(y_true[mask_negative])
    y_true_positive = torch.reshape(torch.ones([number_positive]),(-1,))    
    y_true_negative = torch.reshape(torch.zeros([number_negative]),(-1,))   
    y_true_ = torch.cat((y_true_positive,y_true_negative), dim=0) 
    y_score_positive = torch.tensor(y_score[mask_positive])
    y_score_negative = torch.tensor(y_score[mask_negative])
    indices = torch.tensor([negative, positive])
    y_score_positive_ =torch.index_select(y_score_positive, 1, indices)
    y_score_negative_ =torch.index_select(y_score_negative, 1, indices)
    
    y_scores_pos_prob = torch.exp(y_score_positive_)/torch.sum(torch.exp(y_score_positive_), keepdim=True, dim=1)
    y_scores_neg_prob = torch.exp(y_score_negative_)/torch.sum(torch.exp(y_score_negative_), keepdim=True, dim=1)

    y_prob_positiveclass = torch.reshape(y_scores_pos_prob[:,1],(-1,))
    y_prob_positiveclass_neg = torch.reshape(y_scores_neg_prob[:,1],(-1,))

    y_prob_positive = torch.cat((y_prob_positiveclass,y_prob_positiveclass_neg), dim=0)
    
    fpr, tpr, thrs = roc_curve(y_true_.numpy(), y_prob_positive.numpy(), pos_label=1)
    return fpr, tpr
