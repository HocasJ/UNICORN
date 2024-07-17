# from abc import ABC
# import torch
# import torch.nn.functional as F
# from torch import nn, autograd
# from scipy.optimize import linear_sum_assignment
# from util.loss.supcontrast import SupConLoss
# import numpy as np

# class CM(autograd.Function):

#     @staticmethod
#     def forward(ctx, inputs_rgb, inputs_ir, targets_rgb, targets_ir, features, momentum):
#         ctx.features = features
#         ctx.momentum = momentum

#         ctx.save_for_backward(inputs_rgb, inputs_ir, targets_rgb, targets_ir)
#         outputs_rgb = inputs_rgb.mm(ctx.features.t())
#         outputs_ir = inputs_ir.mm(ctx.features.t())

#         return outputs_rgb, outputs_ir

#     @staticmethod
#     def backward(ctx, grad_outputs1, grad_outputs2):
#         inputs_rgb, inputs_ir, targets_rgb, targets_ir = ctx.saved_tensors
#         grad_inputs1 = None
#         grad_inputs2 = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs1 = grad_outputs1.mm(ctx.features)
#         if ctx.needs_input_grad[1]:
#             grad_inputs2 = grad_outputs2.mm(ctx.features)

#         # momentum update
#         inputs = torch.cat((inputs_rgb, inputs_ir), dim=0)
#         targets = torch.cat((targets_rgb, targets_ir), dim=0)
#         for x,y in zip(inputs, targets):
#             ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
#             ctx.features[y] /= ctx.features[y].norm()

#         return grad_inputs1, grad_inputs2, None, None, None, None


# # outputs = cm(inputs, targets, rgb_size, self.features, self.num_samples_rgb, self.num_samples_ir, self.momentum)
# def cm(inputs_rgb, inputs_ir, indexes_rgb, indexes_ir, features, momentum):
#     return CM.apply(inputs_rgb, inputs_ir, indexes_rgb ,indexes_ir, features, torch.Tensor([momentum]).to(inputs_rgb.device))


# class ClusterMemory(nn.Module, ABC):
#     def __init__(self, num_features, num_samples, prototype_labels, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
#         super(ClusterMemory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples

#         self.momentum = momentum
#         self.change_scale = change_scale
#         self.temp = temp
#         self.use_hard = use_hard

#         self.register_buffer('features', torch.zeros(num_samples, num_features))

#         self.prototype_labels = prototype_labels


#     def forward(self, inputs_rgb, inputs_ir, targets_rgb, targets_ir):
#         inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
#         inputs_ir = F.normalize(inputs_ir, dim=1).cuda()

#         outputs_rgb, outputs_ir = cm(inputs_rgb, inputs_ir, targets_rgb, targets_ir, self.features,  self.momentum)
#         outputs_rgb /= self.temp
#         outputs_ir /= self.temp
#         loss_rgb = F.cross_entropy(outputs_rgb, targets_rgb)
#         loss_ir = F.cross_entropy(outputs_ir, targets_ir)

#         return loss_rgb, loss_ir

from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from scipy.optimize import linear_sum_assignment
from util.loss.supcontrast import SupConLoss
import numpy as np

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_ir, targets_rgb, targets_ir, features, momentum):
        ctx.features = features
        ctx.momentum = momentum

        ctx.save_for_backward(inputs_rgb, inputs_ir, targets_rgb, targets_ir)
        outputs_rgb = inputs_rgb.mm(ctx.features.t())
        outputs_ir = inputs_ir.mm(ctx.features.t())

        return outputs_rgb, outputs_ir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs_rgb, inputs_ir, targets_rgb, targets_ir = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None
        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features)

        # momentum update
        inputs = torch.cat((inputs_rgb, inputs_ir), dim=0)
        targets = torch.cat((targets_rgb, targets_ir), dim=0)
        for x,y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs1, grad_inputs2, None, None, None, None


# outputs = cm(inputs, targets, rgb_size, self.features, self.num_samples_rgb, self.num_samples_ir, self.momentum)
def cm(inputs_rgb, inputs_ir, indexes_rgb, indexes_ir, features, momentum):
    return CM.apply(inputs_rgb, inputs_ir, indexes_rgb ,indexes_ir, features, torch.Tensor([momentum]).to(inputs_rgb.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, prototype_labels, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

        self.prototype_labels = prototype_labels


    def forward(self, inputs_rgb, inputs_ir, targets_rgb, targets_ir):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_ir = F.normalize(inputs_ir, dim=1).cuda()

        outputs_rgb, outputs_ir = cm(inputs_rgb, inputs_ir, targets_rgb, targets_ir, self.features,  self.momentum)
        outputs_rgb /= self.temp
        outputs_ir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets_rgb)
        loss_ir = F.cross_entropy(outputs_ir, targets_ir)

        return loss_rgb, loss_ir


# from abc import ABC
# import torch
# import torch.nn.functional as F
# from torch import nn, autograd
# from scipy.optimize import linear_sum_assignment
# from util.loss.supcontrast import SupConLoss
# import numpy as np

class CM_double(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_ir, targets_rgb, targets_ir, features_rgb, features_ir,prototype_labels_rgb
        ,prototype_labels_ir, momentum, scale):
        ctx.features_rgb = features_rgb
        ctx.features_ir = features_ir
        ctx.momentum = momentum
        ctx.scale = scale

        ctx.save_for_backward(inputs_rgb, inputs_ir, targets_rgb, targets_ir,prototype_labels_rgb,prototype_labels_ir)
        outputs_rgb = inputs_rgb.mm(ctx.features_rgb.t())
        outputs_ir = inputs_ir.mm(ctx.features_ir.t())

        return outputs_rgb, outputs_ir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs_rgb, inputs_ir, targets_rgb, targets_ir, prototype_labels_rgb,prototype_labels_ir = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None
        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features_rgb)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features_ir)

        # momentum update
        for x,y in zip(inputs_rgb, targets_rgb):
            ctx.features_rgb[y] = ctx.momentum * ctx.features_rgb[y] + (1. - ctx.momentum) * x
            ctx.features_rgb[y] /= ctx.features_rgb[y].norm()

        for x, y in zip(inputs_ir, targets_ir):
            ctx.features_ir[y] = ctx.momentum * ctx.features_ir[y] + (1. - ctx.momentum) * x
            ctx.features_ir[y] /= ctx.features_ir[y].norm()

        
        if ctx.scale != 1 :
            rgb_idx = prototype_labels_rgb
            ir_idx = prototype_labels_ir
            temp_rgb = ctx.features_rgb
            for r_idx, i_idx in zip(rgb_idx,ir_idx):
                if r_idx in targets_rgb:
                    i_idx =torch.where(ir_idx == r_idx)[0][0]
                    ctx.features_rgb[r_idx] = ctx.scale * ctx.features_rgb[r_idx] + (1. - ctx.scale) * ctx.features_ir[i_idx]
                    ctx.features_rgb[r_idx] /= ctx.features_rgb[r_idx].norm()
                if i_idx in targets_ir:
                    r_idx = torch.where(rgb_idx == i_idx)[0][0]
                    ctx.features_ir[i_idx] =  ctx.scale * ctx.features_ir[i_idx] + (1. - ctx.scale) * temp_rgb[r_idx]
                    ctx.features_ir[i_idx] /= ctx.features_ir[i_idx].norm()

            del temp_rgb

        return grad_inputs1, grad_inputs2, None, None, None, None, None, None, None, None


# outputs = cm(inputs, targets, rgb_size, self.features, self.num_samples_rgb, self.num_samples_ir, self.momentum)
def cm_double(inputs_rgb, inputs_ir, indexes_rgb, indexes_ir, features_rgb, features_ir, prototype_labels_rgb,prototype_labels_ir, momentum, change_scale):
    return CM_double.apply(inputs_rgb, inputs_ir, indexes_rgb ,indexes_ir, features_rgb, features_ir,prototype_labels_rgb,prototype_labels_ir, torch.Tensor([momentum]).to(inputs_rgb.device), torch.Tensor([change_scale]).to(inputs_rgb.device))


class ClusterMemory_double(nn.Module, ABC):
    def __init__(self, num_features, num_samples_rgb, num_samples_ir, prototype_labels_rgb, prototype_labels_ir,temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory_double, self).__init__()
        self.num_features = num_features
        self.num_samples_rgb = num_samples_rgb
        self.num_samples_ir = num_samples_ir

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features_rgb', torch.zeros(num_samples_rgb, num_features))
        self.register_buffer('features_ir', torch.zeros(num_samples_ir, num_features))

        self.prototype_labels_rgb = prototype_labels_rgb
        self.prototype_labels_ir = prototype_labels_ir


    def forward(self, inputs_rgb, inputs_ir, targets_rgb, targets_ir):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_ir = F.normalize(inputs_ir, dim=1).cuda()

        outputs_rgb, outputs_ir = cm_double(inputs_rgb, inputs_ir, targets_rgb, targets_ir, self.features_rgb, self.features_ir, self.prototype_labels_rgb,
        self.prototype_labels_ir, self.momentum, self.change_scale)
        outputs_rgb /= self.temp
        outputs_ir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets_rgb)
        loss_ir = F.cross_entropy(outputs_ir, targets_ir)

        device = 'cuda'
        xent = SupConLoss(device)

        loss_contr_rgb = xent(inputs_rgb, self.features_rgb, targets_rgb, self.prototype_labels_rgb)
        loss_contr_ir = xent(inputs_ir, self.features_ir, targets_ir, self.prototype_labels_ir)

        loss_contr = loss_contr_rgb + loss_contr_ir


        loss_contr_rgb_cross = xent(inputs_rgb, self.features_ir, targets_rgb, self.prototype_labels_ir)
        loss_contr_ir_cross = xent(inputs_ir, self.features_rgb, targets_ir, self.prototype_labels_rgb)

        loss_contr_cross = loss_contr_rgb_cross + loss_contr_ir_cross

        return loss_contr, loss_contr_cross#loss_rgb, loss_ir







class CM2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs_rgb, inputs_ir, targets_rgb, targets_ir, features, momentum):
        ctx.features = features
        ctx.momentum = momentum

        ctx.save_for_backward(inputs_rgb, inputs_ir, targets_rgb, targets_ir)
        outputs_rgb = inputs_rgb.mm(ctx.features.t())
        outputs_ir = inputs_ir.mm(ctx.features.t())

        return outputs_rgb, outputs_ir

    @staticmethod
    def backward(ctx, grad_outputs1, grad_outputs2):
        inputs_rgb, inputs_ir, targets_rgb, targets_ir = ctx.saved_tensors
        grad_inputs1 = None
        grad_inputs2 = None
        if ctx.needs_input_grad[0]:
            grad_inputs1 = grad_outputs1.mm(ctx.features)
        if ctx.needs_input_grad[1]:
            grad_inputs2 = grad_outputs2.mm(ctx.features)

        # momentum update
        inputs = torch.cat((inputs_rgb, inputs_ir), dim=0)
        targets = torch.cat((targets_rgb, targets_ir), dim=0)
        for x,y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs1, grad_inputs2, None, None, None, None


# outputs = cm(inputs, targets, rgb_size, self.features, self.num_samples_rgb, self.num_samples_ir, self.momentum)
def cm2(inputs_rgb, inputs_ir, indexes_rgb, indexes_ir, features, momentum):
    return CM2.apply(inputs_rgb, inputs_ir, indexes_rgb ,indexes_ir, features, torch.Tensor([momentum]).to(inputs_rgb.device))


class ClusterMemory2(nn.Module, ABC):
    def __init__(self, num_features, num_samples, prototype_labels, temp=0.05, momentum=0.9, use_hard=False, change_scale=0.9):
        super(ClusterMemory2, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.change_scale = change_scale
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

        self.prototype_labels = prototype_labels


    def forward(self, inputs_rgb, inputs_ir, targets_rgb, targets_ir):
        inputs_rgb = F.normalize(inputs_rgb, dim=1).cuda()
        inputs_ir = F.normalize(inputs_ir, dim=1).cuda()

        outputs_rgb, outputs_ir = cm2(inputs_rgb, inputs_ir, targets_rgb, targets_ir, self.features,  self.momentum)
        outputs_rgb /= self.temp
        outputs_ir /= self.temp
        loss_rgb = F.cross_entropy(outputs_rgb, targets_rgb)
        loss_ir = F.cross_entropy(outputs_ir, targets_ir)

        device = 'cuda'
        xent = SupConLoss(device)

        loss_contr_rgb = xent(inputs_rgb, self.features, targets_rgb, self.prototype_labels)
        loss_contr_ir = xent(inputs_rgb, self.features, targets_rgb, self.prototype_labels)

        loss_contr = loss_contr_rgb + loss_contr_ir

        return loss_contr#loss_contr_rgb, loss_contr_ir