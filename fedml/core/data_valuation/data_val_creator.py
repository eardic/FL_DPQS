from fedml.core.data_valuation.feature_valuator import FeatureValuator
from fedml.core.data_valuation.fedbal_data_valuator import FedBalLossValuation, FedBalGradNormValuation
from fedml.core.data_valuation.gradient_shapley import GShapley
from fedml.core.data_valuation.leave_one_out import LOO
from fedml.core.data_valuation.tmc_group_shapley import TMCGroupShapley
from fedml.core.data_valuation.tmc_shapley import TMCShapley
from fedml.core.data_valuation.anomaly_valuator import AnomalyBasedValuator


def create_data_valuator(trainer, args):
    if not hasattr(args, "data_val_algo"):
        return None
    if "loo" == args.data_val_algo:
        return LOO(trainer, args)
    elif "gshap" in args.data_val_algo:
        return GShapley(trainer, args)
    elif "tmc" in args.data_val_algo:
        return TMCShapley(trainer, args)
    elif "tmc_group" in args.data_val_algo:
        return TMCGroupShapley(trainer, args)
    elif "fedbal_loss" in args.data_val_algo:
        return FedBalLossValuation(trainer, args)
    elif "fedbal_gnorm" in args.data_val_algo:
        return FedBalGradNormValuation(trainer, args)
    elif "ftr_sim" in args.data_val_algo:
        return FeatureValuator(trainer, args)
    elif "anomaly" in args.data_val_algo:
        return AnomalyBasedValuator(trainer, args)

    return None
