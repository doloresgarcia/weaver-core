import wandb
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

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
    fpr_bg, tpr_bg, auc_bg= create_binary_rocs(4, 0, y_true, y_score)
    fpr_bud, tpr_bud, auc_bud  = create_binary_rocs(4, 1, y_true, y_score)
    fpr_bc, tpr_bc, auc_bc = create_binary_rocs(4, 3, y_true, y_score)
    calculate_and_log_tpr_1_10_percent(fpr_bg,tpr_bg, "b", "g")
    calculate_and_log_tpr_1_10_percent(fpr_bud,tpr_bud, "b", "ud")
    calculate_and_log_tpr_1_10_percent(fpr_bc,tpr_bc, "b", "c")
    columns = ['b vs g', 'b vs ud', 'b vs c']
    xs = [tpr_bg, tpr_bud, tpr_bc]
    ys = [fpr_bg, fpr_bud, fpr_bc]
    auc_ = [auc_bc, auc_bud, auc_bc]
    title_log = "roc b"
    title_plot = "b tagging"
    wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
    wandb_log_auc(auc_, ["b_g", "b_ud", "b_c"])

    # c tagging (c/g, c/ud, c/b)
    fpr_cg, tpr_cg, auc_cg = create_binary_rocs(3, 0, y_true, y_score)
    fpr_cud, tpr_cud, auc_cud= create_binary_rocs(3, 1, y_true, y_score)
    fpr_cb, tpr_cb, auc_cb = create_binary_rocs(3, 4, y_true, y_score)
    calculate_and_log_tpr_1_10_percent(fpr_cg,tpr_cg, "c", "g")
    calculate_and_log_tpr_1_10_percent(fpr_cud,tpr_cud, "c", "ud")
    calculate_and_log_tpr_1_10_percent(fpr_cb,tpr_cb, "c", "b")
    columns = ['c vs g', 'c vs ud', 'c vs b']
    xs = [tpr_cg, tpr_cud, tpr_cb]
    ys = [fpr_cg, fpr_cud, fpr_cb]
    auc_ = [auc_cg, auc_cud, auc_cb]
    title_log = "roc c"
    title_plot = "c tagging"
    wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
    wandb_log_auc(auc_, ["c_g", "c_ud", "c_b"])

    # s tagging (s/g, s/ud, s/c, s/b)
    fpr_sg, tpr_sg, auc_sg = create_binary_rocs(2, 0, y_true, y_score)
    fpr_sud, tpr_sud, auc_sud  = create_binary_rocs(2, 1, y_true, y_score)
    fpr_sc, tpr_sc, auc_sc  = create_binary_rocs(2, 3, y_true, y_score)
    fpr_sb, tpr_sb, auc_sb  = create_binary_rocs(2, 4, y_true, y_score)
    calculate_and_log_tpr_1_10_percent(fpr_sg,tpr_sg, "s", "g")
    calculate_and_log_tpr_1_10_percent(fpr_sud,tpr_sud, "s", "ud")
    calculate_and_log_tpr_1_10_percent(fpr_sc,tpr_sc, "s", "c")
    calculate_and_log_tpr_1_10_percent(fpr_sb,tpr_sb, "s", "b")
    columns = ['s vs g', 's vs ud', 's vs c', 's vs b']
    xs = [tpr_sg, tpr_sud, tpr_sc, tpr_sb]
    ys = [fpr_sg, fpr_sud, fpr_sc, fpr_sb]
    auc_ = [auc_sg, auc_sud, auc_sb]
    title_log = "roc s"
    title_plot = "s tagging"
    wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
    wandb_log_auc(auc_, ["s_g", "s_ud", "s_c", "s_b"])

    # g tagging (g/ud, g/s, g/c, g/b)
    fpr_gud, tpr_gud, auc_gud  = create_binary_rocs(0, 1, y_true, y_score)
    fpr_gs, tpr_gs, auc_gs  = create_binary_rocs(0, 2, y_true, y_score)
    fpr_gc, tpr_gc, auc_gc  = create_binary_rocs(0, 3, y_true, y_score)
    fpr_gb, tpr_gb, auc_gb  = create_binary_rocs(0, 4, y_true, y_score)
    calculate_and_log_tpr_1_10_percent(fpr_gud,tpr_gud, "g", "ud")
    calculate_and_log_tpr_1_10_percent(fpr_gs,tpr_gs, "g", "s")
    calculate_and_log_tpr_1_10_percent(fpr_gc,tpr_gc, "g", "c")
    calculate_and_log_tpr_1_10_percent(fpr_gb,tpr_gb, "g", "b")
    columns = ['g vs ud', 'g vs s', 'g vs c', 'g vs b']
    xs = [tpr_gud, tpr_gs, tpr_gc, tpr_gb]
    ys = [fpr_gud, fpr_gs, fpr_gc, fpr_gb]
    auc_ = [auc_gud, auc_gs, auc_gc, auc_gb]
    title_log = "roc g"
    title_plot = "g tagging"
    wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
    wandb_log_auc(auc_, ["g_ud", "g_s", "g_c", "g_b"])
    
    
#def tagging_at_xpercent_misstag():


def wandb_log_auc(auc_, names ):
    for i in range(0,len(auc_)):
        name = "auc/" + names[i]
        # logging 1-auc because we are looking at roc with flipped axis
        wandb.log({name: 1-auc_[i]}) 




    return auc_

def wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns):
    ys_log = [np.log10(j+1e-8) for j in ys]
    wandb.log({title_log : wandb.plot.line_series(
        xs=xs,
        ys=ys_log,
        keys=columns,
        title=title_plot, 
        xname="jet tagging efficiency")})

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx

def create_binary_rocs(positive, negative, y_true, y_score):
    mask_positive = y_true==positive
    mask_negative = y_true==negative 
    number_positive = len(y_true[mask_positive])
    number_negative = len(y_true[mask_negative])
    print('s',positive,negative,number_positive,number_negative)
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

    auc_score = roc_auc_score(y_true_.numpy(), y_prob_positive.numpy())


    return fpr, tpr, auc_score

def calculate_and_log_tpr_1_10_percent(fpr,tpr, name_pos, name_neg):
    idx_10_percent = find_nearest(fpr,0.1)
    idx_1_percent = find_nearest(fpr,0.01)

    tpr_10_percent = tpr[idx_10_percent]
    tpr_1_percent = tpr[idx_1_percent]

    name_10 = "tageff/"+ name_pos + "-tagging eff-10%-" + name_neg + "misstag rate"
    name_1 = "tageff/"+ name_pos + "-tagging eff-1%" + name_neg + "misstag rate"
    wandb.log({name_10: tpr_10_percent, name_1: tpr_1_percent})