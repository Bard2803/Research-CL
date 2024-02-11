"""
 Strategy callbacks provide access before/after each phase of the training
 and evaluation loops. Subclasses can override the desired callbacks to
 customize the loops. In Avalanche, callbacks are used by
 :class:`StrategyPlugin` to implement continual strategies, and
 :class:`StrategyLogger` for automatic logging.

 For each method of the training and evaluation loops, `StrategyCallbacks`
 provide two functions `before_{method}` and `after_{method}`, called
 before and after the method, respectively.

 As a reminder, `BaseStrategy` loops follow the structure shown below:

 **Training loop**
 The training loop is organized as follows::
     train
         train_exp  # for each experience
             adapt_train_dataset
             train_dataset_adaptation
             make_train_dataloader
             train_epoch  # for each epoch
                 # forward
                 # backward
                 # model update

 **Evaluation loop**
 The evaluation loop is organized as follows::
     eval
         eval_exp  # for each experience
             adapt_eval_dataset
             eval_dataset_adaptation
             make_eval_dataloader
             eval_epoch  # for each epoch
                 # forward
                 # backward
                 # model update
 """
from avalanche.core import SupervisedPlugin
from avalanche.core import BasePlugin
import torch


class argsRMN:
    def __init__(self, n_epochs):
        self.name = "RMN_"
        self.savename = "RMN_save_"
        self.prune_epoch = 30
        self.train_epochs = n_epochs # 50
        self.wt_para = 1.e-5
        self.prune_para = 0.999999
        self.print_ev = 10

    def getOldModel(self, currTask):
        return '{}{}{}.pt'.format(self.name, self.savename, currTask - 1)

    def saveModel(self, model, currTask):
        torch.save(model, '{}{}{}.pt'.format(self.name, self.savename, currTask))

class RMNPlugin(SupervisedPlugin):
    """ A customized strategy needed to implement the Relevance Mapping Networks (RMN) approach."""
    def __init__(self, n_epochs):
        super().__init__()
        self.currTask_ = -1
        self.argsRMN_ = argsRMN(n_epochs)
        self.epochCounter_ = 0

    def updateTaskLabel(self, strategy):
        self.currTask_ = strategy.experience.task_label

    def bTurnOffAdjx(self, strategy: 'BasePlugin'):
        if self.currTask_ > 0:
            for ix in range(self.currTask_):
                strategy.model = self.turn_off_adjx(strategy.model, ix, bn_off=True)

    def before_training_exp(self, strategy: 'BasePlugin', **kwargs):
        self.epochCounter_ = 0
        self.updateTaskLabel(strategy)
        self.bTurnOffAdjx(strategy)

        if self.currTask_ > 0:
            free_wts = self.learnable_weights(strategy.model, strategy.model.nTasks, wt_para=self.argsRMN_.wt_para, verbose=True)
            print("\n Parameters that are still trainable = {} %".format(free_wts * 100))

            for ix in range(self.currTask_):
                """Checking whether old adjx and fixed weights are preserved"""
                error_signal = self.check_model_version(strategy.model, old_path='{}{}{}.pt'.format(self.argsRMN_.name, self.argsRMN_.savename, self.currTask_ - 1), task=ix)
                assert (error_signal == False)

    def before_eval_exp(self, strategy: 'BasePlugin', **kwargs):
        self.updateTaskLabel(strategy)
        self.bTurnOffAdjx(strategy)

    def before_forward(self, strategy: 'BasePlugin', **kwargs):
        self.updateTaskLabel(strategy)
        # if self.argsRMN_.datasetName == 'pmnist':
        #     strategy.mbatch[0] = strategy.mbatch[0].view(-1,28*28)[:,self.argsRMN_.getPermuations()[self.currTask_]]
        #     #strategy.mbatch[0] = strategy.mbatch[0].view(-1,1,28,28)
        #     strategy.mbatch[1] = strategy.mbatch[1].view(-1)

    def before_eval_forward(self, strategy: 'BasePlugin', **kwargs):
        self.updateTaskLabel(strategy)
        # if self.argsRMN_.datasetName == 'pmnist':# and not self.isPermuted:
        #     strategy.mbatch[0] = strategy.mbatch[0].view(-1,28*28)[:,self.argsRMN_.getPermuations()[self.currTask_]]
        #     #strategy.mbatch[0] = strategy.mbatch[0].view(-1,1,28,28)
        #     strategy.mbatch[1] = strategy.mbatch[1].view(-1)

    def before_training_epoch(self, strategy: 'BasePlugin', **kwargs):
        # switch off task related masks of all previous tasks
        # -> assumes the tasks to be presented increasing integer numbers!
        self.epochCounter_ += 1
        self.updateTaskLabel(strategy)
        if self.currTask_ > 0:
            for ix in range(self.currTask_):
                strategy.model = self.turn_off_adjx(strategy.model, ix, bn_off=True)
    def isinstanceCustom(self, module, compStr='ALinear'):
        return type(module).__name__ == compStr

    def turn_off_adjx(self, model, task, bn_off=False):
            for name, module in model.named_children():
                if self.isinstanceCustom(module, 'ALinear') or self.isinstanceCustom(module, 'AConv2d'):
                    module.adjx[task].requires_grad = False
                elif bn_off == True:
                    """Turn off batch norm as well for the task/adjx"""
                    if self.isinstanceCustom(module, 'ModuleList'):#isinstance(module, torch.nn.ModuleList):  # dependent on how you defined batch norm -not robust needs fixing!
                        for bn_name, bn_param in module[task].named_parameters():
                            bn_param.requires_grad = False
                        module[task].eval()
            return model

    def after_update(self, strategy: 'BasePlugin', **kwargs):
        self.updateTaskLabel(strategy)
        if self.currTask_ > 0:
            # manually keeping weights same for old adjx - needs to be implemented through hooks
            strategy.model=self.change_weights(strategy.model, old_model=self.argsRMN_.getOldModel(self.currTask_), present_task=self.currTask_, path=True)

    def change_weights(self, model, old_model, present_task=0, path=False):
        """change net weights to previous tasks weights where adjx is 1"""
        assert (present_task > 0)
        if path:
            old_model = torch.load(old_model)
        for (n, p), (on, op) in zip(model.named_children(), old_model.named_children()):
            if self.isinstanceCustom(p, 'ALinear') or self.isinstanceCustom(p, 'AConv2d'):
                #assert (isinstance(op, ALinear))
                mask = (op.adjx[0] > 0).data
                for m in range(1, present_task):
                    mask += (op.adjx[m] > 0).data
                mask[mask > 0] = 1.
                with torch.no_grad():
                    p.weight[mask == 1] = op.weight[mask == 1]

        return model

    def after_training_epoch(self, strategy: 'BasePlugin', **kwargs):
        self.updateTaskLabel(strategy)
        if (self.epochCounter_ >= (self.argsRMN_.train_epochs - self.argsRMN_.prune_epoch) and self.argsRMN_.train_epochs > self.argsRMN_.prune_epoch):
            print("Relevance Mapping Pruning...")
            self._prune(strategy.model, task=self.currTask_, wt_sp=True)
            #self.pruned_ = True

    def _prune(self, module, task, wt_sp=False, name=None):
            alive, total = 0.0, 0.
            if self.isinstanceCustom(module, 'ALinear'):#isinstance(module, ALinear):
                if not module.multi:
                    mask = (module.soft_round(module.adjx[task]) > self.argsRMN_.prune_para).data
                    if wt_sp == True:
                        """remove adjacencies for small weights as well - wt_para"""
                        mask_ = (module.soft_round(module.adjx[task]).float() * module.weight.abs() > self.argsRMN_.wt_para).data
                        # mask = torch.logical_and(mask, mask_)
                        #logging.debug("Mask removes adj for weights less than {} - additional:{}%".format \
                        #                  (wt_para, (1 - torch.sum(mask * mask_) / torch.sum(mask)) * 100.))
                        mask = mask * mask_
                    #logging.debug("{} Adjx [{}] Params alive:{}%".format(name, task,
                    #                                                     (torch.count_nonzero(mask) * 1.) / (
                    #                                                                 1. * torch.numel(mask)) * 100.))
                    l = module.adjx[task] * mask.float()
                    ####
                    # In case adjx is all zeros - bad, btw!
                    if (l.sum().item() == 0):
                        #logging.error("adjx[{}] for {} has no activations".format(task, name))
                        if wt_sp == True:
                            mask = (module.soft_round(module.adjx[task]) > self.argsRMN_.prune_para).data
                            mask_ = (module.soft_round(module.adjx[task]) * module.weight.abs() > self.argsRMN_.wt_para / 10).data
                            #logging.debug("Modified Mask removes adj for weights less than {} - additional:{}%".format \
                            #                  (wt_para, torch.sum(mask * mask_) / torch.sum(mask) * 100.))
                            mask = mask * mask_
                            l = module.adjx[task] * mask.float()
                        else:
                            raise AssertionError("{} adjx[{}] is empty!".format(name, task))
                    #####
                    module.adjx[task].data.copy_(l.data)
                    # A = module.soft_round(module.adjx[task]).byte().float() * module.adjx[task]
                    # module.adjx[task].data.copy_(A.data)
                    # print("Params alive:",module.soft_round(module.adjx[task]).byte().sum().float()/np.prod(module.adjx[task].shape))

            if hasattr(module, 'children'):
                for subname, submodule in module.named_children():
                    self._prune(submodule, task, wt_sp, name=subname)

    def turnOffRMaps(self, strategy):
        self.updateTaskLabel(strategy)
        for ix in range(self.currTask_ + 1):
            strategy.model = self.turn_off_adjx(strategy.model, ix, bn_off=True)  # turns off adjacency and BN for the task by requires_grad=False

    def after_training_exp(self, strategy: 'BasePlugin', **kwargs):
        self.turnOffRMaps(strategy)
        self.argsRMN_.saveModel(strategy.model, self.currTask_)

    def after_eval_exp(self, strategy: 'BasePlugin', **kwargs):
        self.turnOffRMaps(strategy)

    def check_model_version(self, present_model, old_path, task, new_model_pth=False, old_model_pth=True):
        """Check if adjacencies and fixed weights don't change over tasks"""
        assert (task > -1)
        error_signal = False  # if true means conditions not met for further training
        print("\nComparing if adjacencies and fixed weights are same over tasks\n")
        if old_model_pth:
            old_model = torch.load(old_path)
        else:
            old_model = old_path
        if new_model_pth:
            present_model = torch.load(present_model)

        for (name, module), (old_name, old_module) in zip(present_model.named_children(), old_model.named_children()):
            if self.isinstanceCustom(module, 'ALinear') or self.isinstanceCustom(module, 'AConv2d'):#isinstance(module, ALinear):
                #assert (isinstance(old_module, ALinear))
                if (module.adjx[task] != old_module.adjx[task]).sum().item() != 0:
                    print("CAUTION: new {}.adjx.{} are not equal to {}.adjx.{}".format(name, task, old_name, task))
                    error_signal = True
                if (module.adjx[task].requires_grad == True) or (old_module.adjx[task].requires_grad == True):
                    print("CAUTION: adjx {} requires grad = True".format(task))
                    error_signal = True
                ### Checking weights as well
                if (module.soft_round(module.adjx[task]).round() * module.weight != \
                    old_module.soft_round(old_module.adjx[task]).round() * old_module.weight).sum().item() != 0:
                    print("\nCAUTION: new {} weights are not equal to old {} weights for task {}". \
                                  format(name, old_name, task))
                    error_signal = True
                    # print(module.soft_round(module.adjx[task]).round()*module.weight)
                    # print(old_module.soft_round(old_module.adjx[task]).round()*old_module.weight)
                    # exit()

            elif self.isinstanceCustom(module, 'ModuleList'): #isinstance(module, torch.nn.ModuleList):
                #assert (isinstance(old_module, torch.nn.ModuleList))
                for (bn_n, bn_p), (old_bn_n, old_bn_p) in zip(module[task].named_parameters(),
                                                              old_module[task].named_parameters()):
                    if (bn_p != old_bn_p).sum().item() != 0:
                        print("CAUTION: new {}.{} are not equal to old {}.{} for task {}".format(name, bn_n, \
                                                                                                         old_name,
                                                                                                         old_bn_n,
                                                                                                         task))
                        error_signal = True
                    if (bn_p.requires_grad == True) or (old_bn_p.requires_grad == True):
                        print("CAUTION: {} BN {} requires grad".format(name, task))
                        error_signal = True
                if module[task].training == True:
                    print("CAUTION: new model BN {} {} is in train()".format(name, task))
                    error_signal = True
                if old_module[task].training == True:
                    print("CAUTION: old model BN {} {} is in train()".format(old_name, task))
                    error_signal = True
            else:
                print("{},{} is not being compared!\n".format(name, old_name))

        return error_signal

    def learnable_weights(self, model, tasks, wt_para=0.22, verbose=False):
        """Returns %age of weights in the model which can be changed - should confirm in hooks"""
        # TODO: in original approach the gradient masks below are uncommented -> check!
        Total_nodes, Free_nodes = 0., 0.
        for name, module in model.named_children():
            if self.isinstanceCustom(module, 'ALinear') or self.isinstanceCustom(module, 'AConv2d'): #isinstance(module, ALinear):
                assert (module.adjx[0].requires_grad == False)
                gradient_mask = (module.soft_round(module.adjx[0]).round().float()*module.weight.abs()<=wt_para).data
                #gradient_mask = (module.adjx[0] == 0).data
                for k in range(1, tasks):
                    if module.adjx[k].requires_grad == False:
                        gradient_mask = gradient_mask * (module.soft_round(module.adjx[k]).round().float()*module.weight.abs() <= wt_para).data
                        #gradient_mask = gradient_mask * (module.adjx[k] == 0).data
                    else:
                        print("Adjacencies of {} for task <= {} are fixed\n".format(name, k - 1))
                        break
                free_lr_nodes = torch.count_nonzero(gradient_mask) * 1.
                total_lr_nodes = torch.numel(gradient_mask) * 1.
                if verbose:
                    print("Free nodes in {} = {}".format(name, free_lr_nodes / total_lr_nodes))
                Free_nodes += free_lr_nodes
                Total_nodes += total_lr_nodes
        return Free_nodes / Total_nodes
