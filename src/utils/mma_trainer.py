import copy
import torch.optim as optim


class GradCloner(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.clone_model = copy.deepcopy(model)
        self.clone_optimizer = optim.SGD(self.clone_model.parameters(), lr=0.)

    def copy_and_clear_grad(self):
        self.clone_optimizer.zero_grad()
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            ## important
            if pvalue.grad is not None:
                cvalue.grad = pvalue.grad.clone()
        self.optimizer.zero_grad()

    def combine_grad(self, alpha=1, beta=1):
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            ## important
            if cvalue.grad is not None:
                pvalue.grad.data = \
                    alpha * pvalue.grad.data + beta * cvalue.grad.data


class TrainEvalMixin(object):
    def __init__(self, model, device, loss_fn,
                 dataname, adversary):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.adversary = adversary
        self.dataname = dataname

        self.dct_eps = {}

    def update_eps(self, eps, idx):
        for jj, ii in enumerate(idx):
            ii = ii.item()
            curr_epsval = eps[jj].item()

            self.dct_eps[ii] = curr_epsval

    def get_eps(self, idx, data):
        lst_eps = []
        for ii in idx:
            ii = ii.item()
            lst_eps.append(max(
                self.adversary.mineps,
                self.dct_eps.setdefault(ii, self.adversary.mineps)
            ))
        return data.new_tensor(lst_eps)


class MMATrainer(TrainEvalMixin):
    def __init__(self, model, device, loss_fn, optimizer,
                 margin_loss_fn, hinge_maxeps, clean_loss_coeff=1. / 3,
                 adversary=None, dataname="train"):
        # lr_by_steps: dict with steps as key, and lr as value
        # lr_by_epochs: dict with epochs as key, and lr as value
        TrainEvalMixin.__init__(
            self, model, device, loss_fn, dataname,
            adversary)

        self.optimizer = optimizer
        self.hinge_maxeps = hinge_maxeps
        self.margin_loss_fn = margin_loss_fn
        self.clean_loss_coeff = clean_loss_coeff
        self.add_clean_loss = clean_loss_coeff > 0

        if self.add_clean_loss:
            self.grad_cloner = GradCloner(self.model, self.optimizer)

    def train_one_batch(self, data, idx, target):
        # clean prediction and save clean gradient
        clnoutput = self.model(data)
        clnloss = self.loss_fn(clnoutput, target)

        if self.add_clean_loss:
            self.optimizer.zero_grad()
            clnloss.backward()
            self.grad_cloner.copy_and_clear_grad()

        # anpgd on correct examples
        search_loss = self.adversary.search_loss_fn(clnoutput, target)
        cln_correct = (search_loss < 0)
        cln_wrong = (search_loss >= 0)

        data_correct = data[cln_correct]
        target_correct = target[cln_correct]
        idx_correct = idx[cln_correct]

        num_correct = cln_correct.sum().item()
        num_wrong = cln_wrong.sum().item()

        curr_eps = data.new_zeros(len(data))
        if num_correct > 0:
            prev_eps = self.get_eps(idx_correct, data)

            advdata_correct, curr_eps_correct = self.adversary(
                data_correct, target_correct, prev_eps)

            data[cln_correct] = advdata_correct
            curr_eps[cln_correct] = curr_eps_correct

        # mma loss and gradient
        mmaoutput = self.model(data)
        if num_correct == 0:
            marginloss = mmaoutput.new_zeros(size=(1,))
        else:
            marginloss = self.margin_loss_fn(
                mmaoutput[cln_correct], target[cln_correct])
        if num_wrong == 0:
            clsloss = 0.
        else:
            clsloss = self.loss_fn(mmaoutput[cln_wrong], target[cln_wrong])

        if num_correct > 0:
            marginloss = marginloss[self.hinge_maxeps > curr_eps_correct]

        mmaloss = (marginloss.sum() + clsloss * num_wrong) / len(data)
        self.optimizer.zero_grad()
        mmaloss.backward()

        # combine gradient from both clean loss and mma loss
        if self.add_clean_loss:
            self.grad_cloner.combine_grad(
                1 - self.clean_loss_coeff, self.clean_loss_coeff)

        self.optimizer.step()

        self.update_eps(curr_eps, idx)
        return clnloss + mmaloss
