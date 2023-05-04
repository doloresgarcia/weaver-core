import numpy as np
import awkward as ak
import tqdm
import time
import torch
from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def train_classification(model, loss_func, opt, scheduler, train_loader, dev, epoch, 
                         steps_per_epoch=None, grad_scaler=None, tb_helper=None, logwandb=False):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for batch_g in tq:
            label = torch.tensor(batch_g[1]).long().to(dev)
            num_examples = label.shape[0]
            label_counter.update(label.cpu().numpy())
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                batch_g_ = batch_g[0].to(dev)
                batch_x = batch_g_.ndata['feat'].to(dev)
                batch_e = batch_g_.edata['feat'].to(dev)

                batch_EigVecs = batch_g_.ndata['EigVecs'].to(dev)
                #random sign flipping
                sign_flip = torch.rand(batch_EigVecs.size(1)).to(dev)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                
                batch_EigVals = batch_g_.ndata['EigVals'].to(dev)
                model_output = model.forward(batch_g_, batch_x, batch_e, batch_EigVecs, batch_EigVals)
                logits = model_output
                loss = loss_func(logits, label)

            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            #loss_log = loss_log.item()
            loss = loss.item()

            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct
           
            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')
            
            if logwandb and (num_batches % 50):
                import wandb
                wandb.log({"loss classification": loss})
                #wandb.log({"loss energy": loss})

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if logwandb:
        wandb.log({"loss_epoch_end": total_loss / num_batches})
        wandb.log({"acc_epoch_end": total_correct / count})

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, 
                            logwandb=False):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels_all=[]
    if logwandb:
        counts_particles = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch_g in tq:
               
                label = torch.tensor(batch_g[1]).long().to(dev)
                entry_count += label.shape[0]
                
                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)
                batch_g_ = batch_g[0].to(dev) 
                batch_x = batch_g_.ndata['feat'].to(dev)
                batch_e = batch_g_.edata['feat'].to(dev)
                batch_EigVecs = batch_g_.ndata['EigVecs'].to(dev)
                batch_EigVals = batch_g_.ndata['EigVals'].to(dev)
                model_output = model.forward(batch_g_, batch_x, batch_e, batch_EigVecs, batch_EigVals)


                logits = model_output

                scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                labels_all.append(label.detach().cpu().numpy())

                _, preds = logits.max(1)
                
      
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')
                if logwandb:
                    import wandb
                    wandb.log({"loss val classification": loss})
                    #wandb.log({"loss val energy": loss})

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break
    

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))
    

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)

    if logwandb:
        counts_particles = np.concatenate(counts_particles)
        from ..logger_wandb import log_confussion_matrix_wandb, log_roc_curves, log_histograms
        y_true_wandb = labels[data_config.label_names[0]]
        scores_wandb = scores
        if len(y_true_wandb)>10000:
            scores_wandb = scores_wandb[0:10000]
            y_true_wandb = y_true_wandb[0:10000]
            counts_particles = counts_particles[0:10000]

        log_confussion_matrix_wandb(y_true_wandb, scores_wandb, epoch)
        log_roc_curves(y_true_wandb, scores_wandb, epoch)
        print('logging HISTOGRAMS')
        log_histograms(y_true_wandb, scores_wandb, counts_particles, epoch)
        
    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert(count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers
