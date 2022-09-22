import importlib
import time

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
    # Set model to evaluation mode
    model.eval()
    # Iterate over data.
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        labels = data[1].to(device)

        # get_attn(model, images)
        # clean images
        outputs = model(images)
        # Predict
        # The loss value here is the average of a batch
        loss = criterion(outputs, labels)
        # Statistics
        acc1, correct = accuracy(outputs, labels, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

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
        # get_attn(model, adv_images)
        adv_outputs = model(adv_images)
        adv_loss = criterion(adv_outputs, labels)
        # Statistics
        acc1, _ = accuracy(adv_outputs, labels, topk=(1, ))
        adv_losses.update(adv_loss.item(), adv_outputs.size(0))
        adv_top1.update(acc1[0].item(), adv_outputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i)

    return model, top1.avg, losses.avg, adv_top1.avg, adv_losses.avg
