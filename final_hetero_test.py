from typing import Dict, List, Tuple
from functools import reduce
import time

import torch

from dataset.dblp import Dblp
from models.hetero_SGAP_models import NARS_SIGN, Fast_NARS_SGC_WithLearnableWeights
from tasks.node_classification import HeteroNodeClassification
from auto_choose_gpu import GpuWithMaxFreeMem

# Hyperparameters
PROP_STEPS = 3
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_EPOCHS = 50
LR = 0.01
WEIGHT_DECAY = 0.0
BATCH_SIZE = 10000


def GenerateSubgraphsWithSameEdgeTypeNum(dataset, random_subgraph_num: int, subgraph_edge_type_num: int) -> Dict:
    return dataset.nars_preprocess(edge_types=dataset.EDGE_TYPES,
                                   predict_class=dataset.TYPE_OF_NODE_TO_PREDICT,
                                   random_subgraph_num=random_subgraph_num,
                                   subgraph_edge_type_num=subgraph_edge_type_num)


# Input format: [(random_subgraph_num, subgraph_edge_type_num), ...]
# Each element is a tuple of (random_subgraph_num, subgraph_edge_type_num)
def GenerateSubgraphDict(dataset, subgraph_config: List) -> Dict:
    subgraph_list = [GenerateSubgraphsWithSameEdgeTypeNum(
        dataset, random_subgraph_num, subgraph_edge_type_num)
        for random_subgraph_num, subgraph_edge_type_num
        in subgraph_config]

    return reduce(lambda x, y: {**x, **y}, subgraph_list)


def Dict2List(dict: Dict) -> List:
    return [(key, dict[key]) for key in dict]


# Input format: [(random_subgraph_num, subgraph_edge_type_num), ...]
# Each element is a tuple of (random_subgraph_num, subgraph_edge_type_num)
def GenerateSubgraphList(dataset, subgraph_config: List) -> List:
    return Dict2List(GenerateSubgraphDict(dataset, subgraph_config))


# Input format: [(random_subgraph_num, subgraph_edge_type_num), ...]
# Each element is a tuple of (random_subgraph_num, subgraph_edge_type_num)
def OneTrialWithSubgraphConfig(dataset, subgraph_config: List, num_epochs: int) -> Tuple[
        float, List, torch.torch.Tensor]:
    subgraph_list = GenerateSubgraphList(dataset, subgraph_config)

    predict_class = dataset.TYPE_OF_NODE_TO_PREDICT

    model = Fast_NARS_SGC_WithLearnableWeights(prop_steps=PROP_STEPS,
                                               feat_dim=dataset.data.num_features[predict_class],
                                               num_classes=dataset.data.num_classes[predict_class],
                                               hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                               random_subgraph_num=len(subgraph_list))

    device = torch.device(
        f"cuda:{GpuWithMaxFreeMem()}" if torch.cuda.is_available() else "cpu")
    classification = HeteroNodeClassification(dataset, predict_class, model,
                                              lr=LR, weight_decay=WEIGHT_DECAY,
                                              epochs=num_epochs, device=device,
                                              train_batch_size=BATCH_SIZE,
                                              eval_batch_size=BATCH_SIZE,
                                              subgraph_list=subgraph_list,
                                              seed=int(time.time()),
                                              record_subgraph_weight=True)
    return classification.test_acc


def main():
    dataset = Dblp(root='.', path_of_zip='./dataset/DBLP_processed.zip')

    SUBGRAPH_CONFIG = [(3, 2), (3, 3)]
    test_acc = OneTrialWithSubgraphConfig(dataset,
                                          SUBGRAPH_CONFIG,
                                          num_epochs_to_train=NUM_EPOCHS)

    print('test_acc:', test_acc)


if __name__ == '__main__':
    main()
