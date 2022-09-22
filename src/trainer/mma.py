#  MMA Training: Direct Input Space Margin Maximization through Adversarial Training
import time
import importlib

AverageMeter = importlib.import_module(f"src.utils.metrics").AverageMeter
ProgressMeter = importlib.import_module(f"src.utils.metrics").ProgressMeter
accuracy = importlib.import_module(f"src.utils.metrics").accuracy
show_tensor = importlib.import_module("src.utils.visualize").show_tensor


def train(model, dataloader, optimizer, args, epoch, device, logger, mma_trainer, es, AttackPolicy):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, top1],
        logger,
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    model.train()
    for i, data in enumerate(dataloader):
        images = data[0].to(device)
        idx = data[1]
        labels = dataloader.target[idx].to(device)
        idx = idx.to(device)

        outputs = model(images)
        # calculate robust loss
        loss = mma_trainer.train_one_batch(images, idx, labels)

        # measure accuracy and record loss
        acc1, _ = accuracy(outputs, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    progress.display(i)

    return es, model, losses.avg, 0, 0, top1.avg
