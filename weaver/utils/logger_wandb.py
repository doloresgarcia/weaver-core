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
    print('EPOCH', epoch, y_true.shape, y_score.shape)
    _bg = create_binary_rocs(4, 0, y_true, y_score)
    _bud  = create_binary_rocs(4, 1, y_true, y_score)
    _bc = create_binary_rocs(4, 3, y_true, y_score)
    if len(_bg)>0 and len(_bud)>0 and len(_bc)>0:
        # this if checks if all elements are not of the same class
        calculate_and_log_tpr_1_10_percent(_bg[0],_bg[1], "b", "g")
        calculate_and_log_tpr_1_10_percent(_bud[0],_bud[1], "b", "ud")
        calculate_and_log_tpr_1_10_percent(_bc[0],_bc[1], "b", "c")
        columns = ['b vs g', 'b vs ud', 'b vs c']
        xs = [_bg[1], _bud[1], _bc[1]]
        ys = [_bg[0], _bud[0], _bc[0]]
        auc_ = [_bc[2], _bud[2], _bc[2]]
        title_log = "roc b"
        title_plot = "b tagging"
        wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
        wandb_log_auc(auc_, ["b_g", "b_ud", "b_c"])
    else:
        print('all batch from the same class in b',len(_bg),len(_bud),len(_bc))

    # c tagging (c/g, c/ud, c/b)
    _cg = create_binary_rocs(3, 0, y_true, y_score)
    _cud = create_binary_rocs(3, 1, y_true, y_score)
    _cb = create_binary_rocs(3, 4, y_true, y_score)
    if len(_cg)>0 and len(_cud)>0 and len(_cb)>0:
        calculate_and_log_tpr_1_10_percent(_cg[0], _cg[1], "c", "g")
        calculate_and_log_tpr_1_10_percent(_cud[0], _cud[1], "c", "ud")
        calculate_and_log_tpr_1_10_percent(_cb[0],_cb[1], "c", "b")
        columns = ['c vs g', 'c vs ud', 'c vs b']
        xs = [_cg[1], _cud[1], _cb[1]]
        ys = [_cg[0], _cud[0], _cb[0]]
        auc_ = [_cg[2], _cud[2], _cb[2]]
        title_log = "roc c"
        title_plot = "c tagging"
        wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
        wandb_log_auc(auc_, ["c_g", "c_ud", "c_b"])
    else:
        print('all batch from the same class in c',len(_cg) ,len(_cud), len(_cb))

    # s tagging (s/g, s/ud, s/c, s/b)

    _sg = create_binary_rocs(2, 0, y_true, y_score)
    _sud = create_binary_rocs(2, 1, y_true, y_score)
    _sc  = create_binary_rocs(2, 3, y_true, y_score)
    _sb  = create_binary_rocs(2, 4, y_true, y_score)
    if len(_sg)>0 and len(_sud)>0 and len(_sc)>0 and len(_sb)>0:
        calculate_and_log_tpr_1_10_percent(_sg[0],_sg[1], "s", "g")
        calculate_and_log_tpr_1_10_percent(_sud[0],_sud[1], "s", "ud")
        calculate_and_log_tpr_1_10_percent(_sc[0],_sc[1], "s", "c")
        calculate_and_log_tpr_1_10_percent(_sb[0],_sb[1], "s", "b")
        columns = ['s vs g', 's vs ud', 's vs c', 's vs b']
        xs = [_sg[1], _sud[1], _sc[1], _sb[1]]
        ys = [_sg[0], _sud[0], _sc[0], _sb[0]]
        auc_ = [_sg[2], _sud[2], _sb[2]]
        title_log = "roc s"
        title_plot = "s tagging"
        wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
        wandb_log_auc(auc_, ["s_g", "s_ud", "s_c", "s_b"])
    else:
        print('all batch from the same class in s',len(_sg),len(_sud),len(_sc), len(_sb))

    # g tagging (g/ud, g/s, g/c, g/b)
    _gud  = create_binary_rocs(0, 1, y_true, y_score)
    _gs  = create_binary_rocs(0, 2, y_true, y_score)
    _gc  = create_binary_rocs(0, 3, y_true, y_score)
    _gb  = create_binary_rocs(0, 4, y_true, y_score)
    if len(_gud)>0 and len(_gs)>0 and len(_gc)>0 and len(_gb)>0:
        calculate_and_log_tpr_1_10_percent(_gud[0],_gud[1], "g", "ud")
        calculate_and_log_tpr_1_10_percent(_gs[0],_gs[1], "g", "s")
        calculate_and_log_tpr_1_10_percent(_gc[0],_gc[1], "g", "c")
        calculate_and_log_tpr_1_10_percent(_gb[0],_gb[1], "g", "b")
        columns = ['g vs ud', 'g vs s', 'g vs c', 'g vs b']
        xs = [_gud[1], _gs[1], _gc[1], _gb[1]]
        ys = [_gud[0], _gs[0], _gc[0], _gb[0]]
        auc_ = [_gud[2], _gs[2], _gc[2], _gb[2]]
        title_log = "roc g"
        title_plot = "g tagging"
        wandb_log_multiline_rocs(xs, ys, title_log, title_plot, columns)
        wandb_log_auc(auc_, ["g_ud", "g_s", "g_c", "g_b"])
    else:
        print('all batch from the same class in g',len(_gud),len(_gs),len(_gc),len(_gb))
    
    
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
    print(y_true.shape, np.sum(mask_positive),  np.sum(mask_negative), positive, negative)
    number_positive = len(y_true[mask_positive])
    number_negative = len(y_true[mask_negative])
    if number_positive>0 and number_negative>0:
        #print('s',positive,negative,number_positive,number_negative)
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
        return [fpr, tpr, auc_score]
    else:
        return []

    

def calculate_and_log_tpr_1_10_percent(fpr,tpr, name_pos, name_neg):
    idx_10_percent = find_nearest(fpr,0.1)
    idx_1_percent = find_nearest(fpr,0.01)

    tpr_10_percent = tpr[idx_10_percent]
    tpr_1_percent = tpr[idx_1_percent]

    name_10 = "tageff/"+ name_pos + "-tagging eff-10%-" + name_neg + "misstag rate"
    name_1 = "tageff/"+ name_pos + "-tagging eff-1%" + name_neg + "misstag rate"
    wandb.log({name_10: tpr_10_percent, name_1: tpr_1_percent})