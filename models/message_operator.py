import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter, ModuleList, Linear
import torch.nn.functional as F

from models.base_op import MessageOp
from models.simple_models import MultiLayerPerceptron
from models.utils import one_dim_weighted_add, two_dim_weighted_add


class LastMessageOp(MessageOp):
    def __init__(self):
        super(LastMessageOp, self).__init__()
        self._aggr_type = "last"

    def _combine(self, feat_list):
        return feat_list[-1]


class SumMessageOp(MessageOp):
    def __init__(self, start, end):
        super(SumMessageOp, self).__init__(start, end)
        self._aggr_type = "sum"

    def _combine(self, feat_list):
        return sum(feat_list[self._start:self._end])


class MeanMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MeanMessageOp, self).__init__(start, end)
        self._aggr_type = "mean"

    def _combine(self, feat_list):
        return sum(feat_list[self._start:self._end]) / (self._end - self._start)


class MaxMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MaxMessageOp, self).__init__(start, end)
        self._aggr_type = "max"

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).max(dim=0)[0]


class MinMessageOp(MessageOp):
    def __init__(self, start, end):
        super(MinMessageOp, self).__init__(start, end)
        self._aggr_type = "min"

    def _combine(self, feat_list):
        return torch.stack(feat_list[self._start:self._end], dim=0).min(dim=0)[0]


class ConcatMessageOp(MessageOp):
    def __init__(self, start, end):
        super(ConcatMessageOp, self).__init__(start, end)
        self._aggr_type = "concat"

    def _combine(self, feat_list):
        return torch.hstack(feat_list[self._start:self._end])


class ProjectedConcatMessageOp(MessageOp):
    def __init__(self, start, end, feat_dim, hidden_dim, num_layers):
        super(ProjectedConcatMessageOp, self).__init__(start, end)
        self._aggr_type = "proj_concat"

        self.__learnable_weight = ModuleList()
        for _ in range(end - start):
            self.__learnable_weight.append(MultiLayerPerceptron(
                feat_dim, hidden_dim, num_layers, hidden_dim))

    def _combine(self, feat_list):
        adopted_feat_list = feat_list[self._start:self._end]

        concat_feat = self.__learnable_weight[0](adopted_feat_list[0])
        for i in range(1, self._end - self._start):
            transformed_feat = F.relu(
                self.__learnable_weight[i](adopted_feat_list[i]))
            concat_feat = torch.hstack((concat_feat, transformed_feat))

        return concat_feat


class SimpleWeightedMessageOp(MessageOp):

    # 'alpha' needs one additional parameter 'alpha';
    # 'hand_crafted' needs one additional parameter 'weight_list'
    def __init__(self, start, end, combination_type, *args):
        super(SimpleWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = "simple_weighted"

        if combination_type not in ["alpha", "hand_crafted"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'alpha' or 'hand_crafted'.")
        self.__combination_type = combination_type

        if len(args) != 1:
            raise ValueError(
                "Invalid parameter numbers for the simple weighted aggregator!")
        self.__alpha, self.__weight_list = None, None
        if combination_type == "alpha":
            self.__alpha = args[0]
            if not isinstance(self.__alpha, float):
                raise TypeError("The alpha must be a float!")
            elif self.__alpha > 1 or self.__alpha < 0:
                raise ValueError("The alpha must be a float in [0,1]!")

        elif combination_type == "hand_crafted":
            self.__weight_list = args[0]
            if isinstance(self.__weight_list, list):
                self.__weight_list = torch.FloatTensor(self.__weight_list)
            elif not isinstance(self.__weight_list, (list, Tensor)):
                raise TypeError(
                    "The input weight list must be a list or a tensor!")

    def _combine(self, feat_list):
        if self.__combination_type == "alpha":
            self.__weight_list = [self.__alpha]
            for _ in range(len(feat_list) - 1):
                self.__weight_list.append(
                    (1 - self.__alpha) * self.__weight_list[-1])
            self.__weight_list = torch.FloatTensor(
                self.__weight_list[self._start:self._end])

        elif self.__combination_type == "hand_crafted":
            pass
        else:
            raise NotImplementedError

        weighted_feat = one_dim_weighted_add(
            feat_list[self._start:self._end], weight_list=self.__weight_list)
        return weighted_feat


class LearnableWeightedMessageOp(MessageOp):

    # 'simple' needs one additional parameter 'prop_steps';
    # 'simple_weighted' allows negative weights, all else being the same as 'simple';
    # 'gate' needs one additional parameter 'feat_dim';
    # 'ori_ref' needs one additional parameter 'feat_dim';
    # 'jk' needs two additional parameter 'prop_steps' and 'feat_dim'
    def __init__(self, start, end, combination_type, *args):
        super(LearnableWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = "learnable_weighted"

        if combination_type not in ["simple", "simple_allow_neg", "gate", "ori_ref", "jk"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'simple', 'simple_allow_neg', 'gate', 'ori_ref' or 'jk'.")
        self.__combination_type = combination_type

        self.__learnable_weight = None
        if combination_type == "simple" or combination_type == "simple_allow_neg":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the simple learnable weighted aggregator!")
            prop_steps = args[0]
            # a 2d tensor is required to use xavier_uniform_.
            tmp_2d_tensor = torch.FloatTensor(1, prop_steps + 1)
            nn.init.xavier_normal_(tmp_2d_tensor)
            self.__learnable_weight = Parameter(tmp_2d_tensor.view(-1))

        elif combination_type == "gate":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the gate learnable weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim, 1)

        elif combination_type == "ori_ref":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the ori_ref learnable weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim + feat_dim, 1)

        elif combination_type == "jk":
            if len(args) != 2:
                raise ValueError(
                    "Invalid parameter numbers for the jk learnable weighted aggregator!")
            prop_steps, feat_dim = args[0], args[1]
            self.__learnable_weight = Linear(
                feat_dim + (prop_steps + 1) * feat_dim, 1)

    def _combine(self, feat_list):
        weight_list = None
        if self.__combination_type == "simple":
            weight_list = F.softmax(torch.sigmoid(
                self.__learnable_weight[self._start:self._end]), dim=0)

        elif self.__combination_type == "simple_allow_neg":
            weight_list = self.__learnable_weight[self._start:self._end]

        elif self.__combination_type == "gate":
            adopted_feat_list = torch.vstack(feat_list[self._start:self._end])
            weight_list = F.softmax(
                torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(self._end - self._start, -1).T), dim=1)

        elif self.__combination_type == "ori_ref":
            reference_feat = feat_list[0].repeat(self._end - self._start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self._start:self._end])))
            weight_list = F.softmax(
                torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(-1, self._end - self._start)), dim=1)

        elif self.__combination_type == "jk":
            reference_feat = torch.hstack(feat_list).repeat(
                self._end - self._start, 1)
            adopted_feat_list = torch.hstack(
                (reference_feat, torch.vstack(feat_list[self._start:self._end])))
            weight_list = F.softmax(
                torch.sigmoid(self.__learnable_weight(adopted_feat_list).view(-1, self._end - self._start)), dim=1)

        else:
            raise NotImplementedError

        weighted_feat = None
        if self.__combination_type == "simple" or self.__combination_type == "simple_allow_neg":
            weighted_feat = one_dim_weighted_add(
                feat_list[self._start:self._end], weight_list=weight_list)
        elif self.__combination_type in ["gate", "ori_ref", "jk"]:
            weighted_feat = two_dim_weighted_add(
                feat_list[self._start:self._end], weight_list=weight_list)
        else:
            raise NotImplementedError

        return weighted_feat


class IterateLearnableWeightedMessageOp(MessageOp):

    # 'recursive' needs one additional parameter 'feat_dim'
    def __init__(self, start, end, combination_type, *args):
        super(IterateLearnableWeightedMessageOp, self).__init__(start, end)
        self._aggr_type = "iterate_learnable_weighted"

        if combination_type not in ["recursive"]:
            raise ValueError(
                "Invalid weighted combination type! Type must be 'recursive'.")
        self.__combination_type = combination_type

        self.__learnable_weight = None
        if combination_type == "recursive":
            if len(args) != 1:
                raise ValueError(
                    "Invalid parameter numbers for the recursive iterate weighted aggregator!")
            feat_dim = args[0]
            self.__learnable_weight = Linear(feat_dim + feat_dim, 1)

    def _combine(self, feat_list):
        weight_list = None
        if self.__combination_type == "recursive":
            weighted_feat = feat_list[self._start]
            for i in range(self._start, self._end):
                weights = torch.sigmoid(self.__learnable_weight(
                    torch.hstack((feat_list[i], weighted_feat))))
                if i == self._start:
                    weight_list = weights
                else:
                    weight_list = torch.hstack((weight_list, weights))
                weight_list = F.softmax(weight_list, dim=1)

                weighted_feat = torch.mul(
                    feat_list[self._start], weight_list[:, 0].view(-1, 1))
                for j in range(1, i + 1):
                    weighted_feat = weighted_feat + \
                        torch.mul(feat_list[self._start + j],
                                  weight_list[:, j].view(-1, 1))

        else:
            raise NotImplementedError

        return weighted_feat
