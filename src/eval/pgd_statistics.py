from tqdm import tqdm
import torch
import importlib
import time
import numpy as np
from sklearn.metrics import roc_auc_score,confusion_matrix,f1_score

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
pgd_whitebox = importlib.import_module("src.utils.adv").pgd_whitebox


def test(model, criterion, dataloader, opt, device, logger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    adv_losses = AverageMeter("Adv_Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    adv_top1 = AverageMeter("Adv-Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, adv_losses, top1, adv_top1],
        logger,
        prefix="Test: ",
    )
    end = time.time()

    # statistics
    y_true = []
    y_pred_clean = []
    y_logits_clean = []
    y_pred_adv = []
    y_logits_adv = []

    # Set model to evaluation mode
    model.eval()
    # Iterate over data.
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)
        # clean images
        outputs = model(images)
        # Predict
        y_pred_clean_batch = torch.max(outputs,dim=1)[1]
        # The loss value here is the average of a batch
        loss = criterion(outputs, labels)
        # Statistics
        acc1 = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0][0].item(), images.size(0))

        # adversarial images
        adv_images = pgd_whitebox(
            model,
            images,
            labels,
            opt.evl_epsilon,
            opt.evl_num_steps,
            opt.evl_step_size,
            opt.clip_min,
            opt.clip_max,
            device=device,
            is_random=not opt.const_init,
        )
        adv_images.to(device)
        # compute output
        adv_outputs = model(adv_images)
        # predict
        y_pred_adv_batch = torch.max(adv_outputs, dim=1)[1]
        adv_loss = criterion(adv_outputs, labels)
        # Statistics
        acc1 = accuracy(adv_outputs, labels, topk=(1, ))
        adv_losses.update(adv_loss.item(), adv_outputs.size(0))
        adv_top1.update(acc1[0][0].item(), adv_outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # accumulate statistics
        y_true.extend(labels.tolist())
        y_pred_clean.extend(y_pred_clean_batch.tolist())
        y_logits_clean.extend(outputs[:,1].tolist())
        y_pred_adv.extend(y_pred_adv_batch.tolist())
        y_logits_adv.extend(adv_outputs[:,1].tolist())
    if opt.num_classes == 2:
        clean_auc = roc_auc_score(y_true, y_logits_clean, multi_class='ovo')
        adv_auc = roc_auc_score(y_true, y_logits_adv, multi_class='ovo')

        clean_cm = confusion_matrix(y_true,y_pred_clean)
        clean_sensitivity = clean_cm[1, 1] / (clean_cm[1, 0] + clean_cm[1, 1])
        clean_specificity = clean_cm[0, 0] / (clean_cm[0, 0] + clean_cm[0, 1])
        clean_f1_score = f1_score(y_true, y_pred_clean)

        adv_cm = confusion_matrix(y_true, y_pred_adv)
        adv_sensitivity = adv_cm[1, 1] / (adv_cm[1, 0] + adv_cm[1, 1])
        adv_specificity = adv_cm[0, 0] / (adv_cm[0, 0] + adv_cm[0, 1])
        adv_f1_score = f1_score(y_true, y_pred_adv)

        y_true = torch.Tensor(y_true)
        y_pred_clean = torch.Tensor(y_pred_clean)
        y_pred_adv = torch.Tensor(y_pred_adv)
        index = torch.eq(y_true, y_pred_clean)

        y_true_subset = y_true[index]
        y_pred_adv_subset = y_pred_adv[index]
        adv_success = 100. * torch.sum(1. - torch.eq(y_true_subset, y_pred_adv_subset).float() / len(y_true_subset))

        logger.info(f"clean_auc:{clean_auc:.3f} | "
                    f"clean_sensitivity:{100. * clean_sensitivity:.1f}% | "
                    f"clean_specificity:{100. * clean_specificity:.1f}% | "
                    f"clean_acc:{top1.avg:.1f}% | "
                    f"clean_f1_score:{clean_f1_score:.3f} | "
                    f"adv_auc:{adv_auc:.3f} | "
                    f"adv_sensitivity:{100. * adv_sensitivity:.1f}% | "
                    f"adv_specificity:{100. * adv_specificity:.1f}% | "
                    f"adv_acc:{adv_top1.avg:.1f}% | "
                    f"adv_f1_score:{adv_f1_score:.3f} |\n"
                    f"adv_success:{adv_success:.3f}")
    else:
        muti_analyze(y_true, y_pred_clean, y_pred_adv, opt.num_classes)
    progress.display(i)

    return model, top1.avg, losses.avg, adv_top1.avg, adv_losses.avg


def sen(Y_test, Y_pred, n):
    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def pre(Y_test, Y_pred, n):
    pre = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:, i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)

    return pre


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def acc(Y_test, Y_pred, n):
    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return acc


def muti_analyze(y_true, y_logits, y_adv_logits, num_classes=5):
    y_true = np.array(y_true)
    y_logits = np.array(y_logits)
    y_adv_logits = np.array(y_adv_logits)
    sensitivity = sen(y_true, y_logits, num_classes)
    precision = pre(y_true, y_logits, num_classes)
    specificity = spe(y_true, y_logits, num_classes)
    # f1 = 2.0 * sensitivity*precision/(sensitivity+precision)
    f1=[]
    for i in range(num_classes):
        f1.append(2.0 * sensitivity[i] * precision[i] / (sensitivity[i] + precision[i]))
    accuracy = acc(y_true, y_logits, num_classes)


    adv_sensitivity = sen(y_true, y_adv_logits, num_classes)
    adv_precision = pre(y_true, y_adv_logits, num_classes)
    adv_specificity = spe(y_true, y_adv_logits, num_classes)
    # adv_f1 = 2.0 * adv_sensitivity * adv_precision / (adv_sensitivity + adv_precision)
    adv_f1=[]
    for i in range(num_classes):
        adv_f1.append(2.0 * adv_sensitivity[i] * adv_precision[i] / (adv_sensitivity[i] + adv_precision[i]))
    adv_accuracy = acc(y_true, y_adv_logits, num_classes)
    print(f"clean sensitivity:{sensitivity} || clean precision:{precision} || cleanspecificity:{specificity} "
          f"||clean f1: || adv sensitivity:{adv_sensitivity} || adv precision:{adv_precision} "
          f"||adv specificity:{adv_specificity} ||adv f1:")
