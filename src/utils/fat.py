import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def earlystop(model, data, target, step_size, epsilon, perturb_steps, clip_min, clip_max, device,
              tau, randominit_type, loss_fn, args, rand_init=True, omega=0):
    '''
    The implematation of early-stopped PGD
    Following the Alg.1 in our FAT paper <https://arxiv.org/abs/2002.11242>
    :param step_size: the PGD step size
    :param epsilon: the perturbation bound
    :param perturb_steps: the maximum PGD step
    :param tau: the step controlling how early we should stop interations when wrong adv data is found
    :param randominit_type: To decide the type of random inirialization (random start for searching adv data)
    :param rand_init: To decide whether to initialize adversarial sample with random noise (random start for searching adv data)
    :param omega: random sample parameter for adv data generation (this is for escaping the local minimum.)
    :return: output_adv (friendly adversarial data) output_target (targets), output_natural (the corresponding natrual data), count (average backword propagations count)
    '''
    model.eval()

    K = perturb_steps
    count = 0
    output_target = []
    output_adv = []
    output_natural = []

    control = (torch.ones(len(target)) * tau).to(device)

    # Initialize the adversarial data with random noise
    if rand_init:
        if randominit_type == "normal_distribution_randominit":
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
        if randominit_type == "uniform_randominit":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device)
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
    else:
        iter_adv = data.to(device).detach()

    iter_clean_data = data.to(device).detach()
    iter_target = target.to(device).detach()
    output_iter_clean_data = model(data)

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        # Calculate the indexes of adversarial data those still needs to be iterated
        for idx in range(len(pred)):
            if pred[idx] != iter_target[idx]:
                if control[idx] == 0:
                    output_index.append(idx)
                else:
                    control[idx] -= 1
                    iter_index.append(idx)
            else:
                iter_index.append(idx)

        # Add adversarial data those do not need any more iteration into set output_adv
        if len(output_index) != 0:
            if len(output_target) == 0:
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, args.input_size, args.input_size).to(device)
                output_natural = iter_clean_data[output_index].reshape(-1, 3, args.input_size, args.input_size).to(device)
                output_target = iter_target[output_index].reshape(-1).to(device)
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, args.input_size, args.input_size).to(device)), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, args.input_size, args.input_size).to(device)), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).to(device)), dim=0)

        # calculate gradient
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(output, iter_target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).to(device)
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        # update iter adv
        if len(iter_index) != 0:
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()

            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().to(device)
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, clip_min, clip_max)
            count += len(iter_target)
        else:
            output_adv = output_adv.detach()
            return output_adv, output_target, output_natural, count
        K = K-1

    if len(output_target) == 0:
        output_target = iter_target.reshape(-1).squeeze().to(device)
        output_adv = iter_adv.reshape(-1, 3, args.input_size, args.input_size).to(device)
        output_natural = iter_clean_data.reshape(-1, 3, args.input_size, args.input_size).to(device)
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, args.input_size, args.input_size)), dim=0).to(device)
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().to(device)
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, args.input_size, args.input_size).to(device)),dim=0).to(device)
    output_adv = output_adv.detach()
    return output_adv, output_target, output_natural, count
