import math


def scenario(src_id, trg_id):
    """Normalize场景ID为字符串元组，避免'几to几'写错或类型不一致。"""
    return str(src_id), str(trg_id)

def backbone_scenario(backbone, src_id, trg_id):
    """Helper to tag overrides with a specific backbone while keeping keys hashable."""
    return str(backbone), str(src_id), str(trg_id)

NUSTAR_DEFAULTS_HAR = {
    "adv_sigma": 0.1,
    "adv_ctrl_points": 10,
    "adv_num_candidates": 16,
    "sem_thresh": 0.5,
    "cons_thresh": 0.5,
    "stat_quantile": 0.7,
    "stat_window": 512,
    "stat_min_history": 32,
    "stat_min_entropy": 0.0,
    "proto_momentum": 0.9,
}

NUSTAR_DEFAULTS_EEG = dict(NUSTAR_DEFAULTS_HAR)
NUSTAR_DEFAULTS_FD = dict(NUSTAR_DEFAULTS_HAR)

def _apply_nustar_defaults(overrides, defaults):
    for _, params in overrides.items():
        for key, value in defaults.items():
            params.setdefault(key, value)


HAR_ACCUP_SCENARIO_OVERRIDES = {
    backbone_scenario('CNN', 2, 11): {
        'batch_size': 18,
        'fisher_alpha': 3000,
        'freeze_bn_stats': False,
        'grad_clip': 1.5,
        'grad_clip_value': 0.25,
        'learning_rate': 2.1085227849137455e-05,
        'lr_decay': 0.4,
        'max_fisher_updates': 128,
        'momentum': 0.4,
        'num_epochs': 8,
        'online_fisher': True,
        'pre_learning_rate': 0.000512563293439829,
        'step_size': 32,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.000663887199726086,
        'adv_sigma': 0.1,
        'sem_thresh': 0.4,
        'cons_thresh': 0.6,
        'stat_quantile': 0.5,
        'stat_window': 256,
        'stat_min_history': 16,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.5,
    },

    backbone_scenario('CNN', 6, 23): {
        'batch_size': 3, #3
        'fisher_alpha': 2000,
        'freeze_bn_stats': False,
        'grad_clip': 0.8,
        'grad_clip_value': None,
        'learning_rate': 3.5e-06,
        'lr_decay': 0.1,
        'max_fisher_updates': 64,
        'momentum': 0.6835535372453551,
        'num_epochs': 15, #15
        'online_fisher': True,
        'pre_learning_rate': 0.00068,
        'step_size': 40,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 3.811117047127126e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.5,
        'cons_thresh': 0.5,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 7, 13): {
        'batch_size': 25, #25
        'fisher_alpha': 1000,
        'freeze_bn_stats': True,
        'grad_clip': 0.5,
        'grad_clip_value': 0.1,
        'learning_rate': 9e-06,
        'lr_decay': 0.2,
        'max_fisher_updates': 32,
        'momentum': 0.3,
        'num_epochs': 15, #15
        'online_fisher': True,
        'pre_learning_rate': 0.0001,
        'step_size': 20,
        'steps': 1,
        'train_backbone_modules':None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.00037599766990728935,
        'adv_sigma': 0.1,
        'sem_thresh': 0.3,
        'cons_thresh': 0.5,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 9, 18): {
        'batch_size': 4, #4
        'fisher_alpha': 2000,
        'freeze_bn_stats': False,
        'grad_clip': 0.6,
        'grad_clip_value': 0.2,
        'learning_rate': 7e-06,
        'lr_decay': 0.2,
        'max_fisher_updates': 128,
        'momentum': 0.8499147261518536,
        'num_epochs': 15,
        'online_fisher': True,
        'pre_learning_rate': 0.0007,
        'step_size': 42,
        'steps': 4,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.0009919512457586324,
        'adv_sigma': 0.1,
        'sem_thresh': 0.3,
        'cons_thresh': 0.5,
        'stat_quantile': 0.4,
        'stat_window': 256,
        'stat_min_history': 64,
        'stat_min_entropy': 0.0,
        'proto_momentum': 1.5,
    },

    backbone_scenario('CNN', 12, 16): {
        'batch_size': 17, #17
        'fisher_alpha': 3233.768103119852,
        'freeze_bn_stats': False,
        'grad_clip': 1.4,
        'grad_clip_value': 0.5,
        'learning_rate': 2.1873782249655115e-06,
        'lr_decay': 0.6634806241909338,
        'max_fisher_updates': 128,
        'momentum': 0.8791299958620375,
        'num_epochs': 14,
        'online_fisher': True,
        'pre_learning_rate': 0.0008,
        'step_size': 57,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.0005745937675896207,
        'adv_sigma': 0.1,
        'sem_thresh': 0.5,
        'cons_thresh': 0.5,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },


}

_apply_nustar_defaults(HAR_ACCUP_SCENARIO_OVERRIDES, NUSTAR_DEFAULTS_HAR)

FD_ACCUP_SCENARIO_OVERRIDES = {
    backbone_scenario('CNN', 0, 1): {
        'batch_size': 105,
        'fisher_alpha': 1000,
        'freeze_bn_stats': True,
        'grad_clip': 0.3891867331952878,
        'grad_clip_value': 1.0,
        'learning_rate': 0.00015, #0.00015
        'lr_decay': 0.5,
        'max_fisher_updates': 32,
        'momentum': 0.8078763519687022,
        'num_epochs': 33,
        'online_fisher': True,
        'pre_learning_rate': 0.0009450856698883171, #0.0009450856698883171
        'step_size': 22,
        'steps': 2,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.0008793200616704788,
        'adv_sigma': 0.2,
        'sem_thresh': 0.2,
        'cons_thresh': 0.6,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.5,
    },

    backbone_scenario('CNN', 1, 0): {
        'batch_size': 120, #96
        'fisher_alpha': 2000,
        'freeze_bn_stats': True,
        'grad_clip': 0.8,
        'grad_clip_value': 0.5,
        'learning_rate': 0.0000003, #0.000012
        'lr_decay': 0.4,
        'max_fisher_updates': 512,
        'momentum': 0.3,
        'num_epochs': 54, #54
        'online_fisher': True,
        'pre_learning_rate': 0.03,
        'step_size': 29,
        'steps': 4,
        'train_backbone_modules':None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.00001,
        'adv_sigma': 0.1,
        'sem_thresh': 0.3,
        'cons_thresh': 0.7,
        'stat_quantile': 0.8,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },
    backbone_scenario('CNN', 1, 2): {
        'batch_size': 138,
        'fisher_alpha': 2000,
        'freeze_bn_stats': True,
        'grad_clip': 0.3,
        'grad_clip_value': None,
        'learning_rate': 0.0003, #0.0003
        'lr_decay': 0.3,
        'max_fisher_updates': 64,
        'momentum': 0.3,
        'num_epochs': 25,
        'online_fisher': True,
        'pre_learning_rate': 0.003,
        'step_size': 28,
        'steps': 3,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.00015905963734717395,
        'adv_sigma': 0.1,
        'sem_thresh': 0.1,
        'cons_thresh': 0.5,
        'stat_quantile': 0.9,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 2, 3): {
        'batch_size': 128,
        'fisher_alpha': 2000,
        'freeze_bn_stats': False,
        'grad_clip': 0.2,
        'grad_clip_value': 0.1,
        'learning_rate': 0.0000009,
        'lr_decay': 0.2629447502891978,
        'max_fisher_updates': 128,
        'momentum': 0.2,
        'num_epochs': 25,
        'online_fisher': True,
        'pre_learning_rate': 0.0009990135722027596,
        'step_size': 25,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.0009204145717746104,
        'adv_sigma': 0.1,
        'sem_thresh': 0.2,
        'cons_thresh': 0.8,
        'stat_quantile': 0.8,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.5,
    },

    backbone_scenario('CNN', 3, 1): {
        'batch_size': 89,
        'fisher_alpha': 1000,
        'freeze_bn_stats': False,
        'grad_clip': 0.4,
        'grad_clip_value': 0.8,
        'learning_rate': 0.0000001,
        'lr_decay': 0.4,
        'max_fisher_updates': 256,
        'momentum': 0.4,
        'num_epochs': 25,
        'online_fisher': True,
        'pre_learning_rate': 0.001,
        'step_size': 30,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 0.00003,
        'adv_sigma': 0.1,
        'sem_thresh': 0.5,
        'cons_thresh': 0.5,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.4,
    },
}

_apply_nustar_defaults(FD_ACCUP_SCENARIO_OVERRIDES, NUSTAR_DEFAULTS_FD)

EEG_ACCUP_SCENARIO_OVERRIDES = {
    backbone_scenario('CNN', 0, 11): {
        'batch_size': 97, #97
        'fisher_alpha': 2000,
        'freeze_bn_stats': True,
        'grad_clip': 0.8,
        'grad_clip_value': None,
        'learning_rate': 3e-05,
        'lr_decay': 0.5,
        'max_fisher_updates': 256,
        'momentum': 0.6,
        'num_epochs': 30,
        'online_fisher': True,
        'step_size': 10,
        'steps': 1,
        'pre_learning_rate': 0.0005, #3
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 5e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.5,
        'cons_thresh': 0.5,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 7, 18): {
        'batch_size': 128,
        'fisher_alpha': 2000,
        'freeze_bn_stats': True,
        'grad_clip': 0.1,
        'grad_clip_value': 0.5,
        'learning_rate': 0.000005, #0.0005
        'lr_decay': 0.6,
        'max_fisher_updates': 512,
        'momentum': 0.7,
        'num_epochs': 30,
        'online_fisher': True,
        'pre_learning_rate': 0.0006, #6
        'step_size': 14,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 4.157333141396952e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.25,
        'cons_thresh': 0.5,
        'stat_quantile': 0.3,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 9, 14): {
        'batch_size': 235,
        'fisher_alpha': 900,
        'freeze_bn_stats': True,
        'grad_clip': 0.8,
        'grad_clip_value': None,
        'learning_rate': 0.000004, #= 65.19
        'lr_decay': 0.5,
        'max_fisher_updates': 256,
        'momentum': 0.4,
        'num_epochs': 25,
        'pre_learning_rate': 0.0023, #0.0015 0.0023
        'step_size': 30,
        'steps': 2,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 4.395625457432446e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.2,
        'cons_thresh': 0.6,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.9,
    },

    backbone_scenario('CNN', 12, 5): {
        'batch_size': 89, #89
        'fisher_alpha': 1200,
        'freeze_bn_stats': True,
        'grad_clip': 0.3,
        'grad_clip_value': 0.1,
        'learning_rate': 0.0002,
        'lr_decay': 0.5,
        'max_fisher_updates': 256,
        'momentum': 0.9,
        'num_epochs': 34, #34
        'pre_learning_rate': 0.0005, #5
        'step_size': 25,
        'steps': 1,
        'train_backbone_modules': None,
        'train_classifier': True,
        'train_full_backbone': True,
        'weight_decay': 6.5e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.25,
        'cons_thresh': 0.8,
        'stat_quantile': 0.7,
        'stat_window': 512,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.6,
    },

    backbone_scenario('CNN', 16, 1): {
        'batch_size': 128,
        'fisher_alpha': 2000,
        'freeze_bn_stats': True,
        'grad_clip': 0.3,
        'grad_clip_value': 0.1,
        'learning_rate': 0.0003, #0.0003
        'lr_decay': 0.3,
        'max_fisher_updates': 256,
        'momentum': 0.8079469220282296,
        'num_epochs': 30,
        'pre_learning_rate': 0.002, #2
        'step_size': 20,
        'steps': 2,
        'train_backbone_modules': ['conv_block1', 'conv_block2'],
        'train_classifier': True,
        'train_full_backbone': False,
        'weight_decay': 2.0322884602495502e-05,
        'adv_sigma': 0.1,
        'sem_thresh': 0.2,
        'cons_thresh': 0.7,
        'stat_quantile': 0.4,
        'stat_window': 256,
        'stat_min_history': 32,
        'stat_min_entropy': 0.0,
        'proto_momentum': 0.2,
    },

}

_apply_nustar_defaults(EEG_ACCUP_SCENARIO_OVERRIDES, NUSTAR_DEFAULTS_EEG)

def get_hparams_class(dataset_name):  # 根据给定数据集名称字符串返回对应的数据集配置类
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'step_size': 30,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.5,
            'grad_clip_value': None
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 3e-4,
                'learning_rate': 1e-4,
                'adv_sigma': 0.1,
                'adv_ctrl_points': 10,
                'adv_num_candidates': 16,
                'sem_thresh': 0.5,
                'cons_thresh': 0.5,
                'stat_quantile': 0.7,
                'stat_window': 512,
                'stat_min_history': 32,
                'stat_min_entropy': 0.0,
                'proto_momentum': 0.9,

                # Regularization
                'fisher_alpha': 2000.0,
                'max_fisher_updates': -1,
                'freeze_bn_stats': False,

                'scenario_overrides': dict(FD_ACCUP_SCENARIO_OVERRIDES),

                'grad_clip': 0.5,
                'grad_clip_value': None
            },
            'NoAdap': {'pre_learning_rate': 5e-4}
        }

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 96,        # ↑ 从 64 提到 96：更稳定的 BN/统计，CPU 也扛得住
            'weight_decay': 5e-5,
            'step_size': 20,         # ↓ 让预训练在第20轮衰减一次学习率
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.5,        # ↑ 统一为 0.5，避免双处设置相互打架
            'grad_clip_value': None
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 5e-4,
                'learning_rate': 3e-4,
                'adv_sigma': 0.1,
                'adv_ctrl_points': 10,
                'adv_num_candidates': 16,
                'sem_thresh': 0.5,
                'cons_thresh': 0.5,
                'stat_quantile': 0.7,
                'stat_window': 512,
                'stat_min_history': 32,
                'stat_min_entropy': 0.0,
                'proto_momentum': 0.9,

                # Regularization
                'fisher_alpha': 2000.0,
                'max_fisher_updates': -1,
                'freeze_bn_stats': False,

                'scenario_overrides': dict(EEG_ACCUP_SCENARIO_OVERRIDES),

                'grad_clip': 0.5,
                'grad_clip_value': None
            },
            'NoAdap': {'pre_learning_rate': 5e-4}
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 15,  # 30 -> 16（避免源域过拟合，利于 TTA）
            'batch_size': 16,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.1,
            'grad_clip_value': None
        }
        # 关键：NuSTAR 的核心超参
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 5e-4,
                'learning_rate': 3e-5,  # TTA 基础 lr；如果代码支持分组，BN 用 5e-5
                'adv_sigma': 0.1,
                'adv_ctrl_points': 10,
                'adv_num_candidates': 16,
                'sem_thresh': 0.5,
                'cons_thresh': 0.5,
                'stat_quantile': 0.7,
                'stat_window': 512,
                'stat_min_history': 32,
                'stat_min_entropy': 0.0,
                'proto_momentum': 0.9,

                # Regularization
                'fisher_alpha': 2000.0,
                'max_fisher_updates': -1,
                'freeze_bn_stats': False,

                'scenario_overrides': dict(HAR_ACCUP_SCENARIO_OVERRIDES),

                'grad_clip': 1.0,
                'grad_clip_value': 0.5
            },
            'NoAdap': {'pre_learning_rate': 1e-3}
        }
