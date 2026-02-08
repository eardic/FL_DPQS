import numpy as np
import threading
import wandb


class FedBalThreshController(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.min_list = []
        self.max_list = []
        self.sum = 0
        self.total_sample_num = 0
        self.utility = []
        self.ltr = 0
        self.new_thresh = 0
        self.w = int(args.fedbal_update_period)
        self.ss = args.fedbal_step_size
        self.lock = threading.Lock()

    def get_thresh(self):
        return self.new_thresh

    def update(self, round_ix):
        self.lock.acquire(blocking=True)
        try:
            if self.total_sample_num == 0:
                print("FedBalancer: No samples ! Skipped thresh update !")
                return self.new_thresh
            stat_util = self.sum / self.total_sample_num
            wandb.log({"FedBal/StatUtility": stat_util, "round": round_ix})
            wandb.log({"FedBal/TotalSamples": self.total_sample_num, "round": round_ix})
            wandb.log({"FedBal/Sum": self.sum, "round": round_ix})
            self.utility.append(stat_util)
            R = round_ix + 1
            if R % self.w == 0 and len(self.utility) >= 2 * self.w:
                past_util = np.sum(self.utility[-2 * self.w: -self.w])
                recent_util = np.sum(self.utility[-self.w:])
                if past_util > recent_util:
                    self.ltr = min(self.ltr + self.ss, 1.0)
                else:
                    self.ltr = max(self.ltr - self.ss, 0.0)
            ll = np.min(self.min_list)
            lh = np.mean(self.max_list)
            self.new_thresh = ll + (lh-ll) * self.ltr
            wandb.log({
                "FedBal/MinLoss": ll, 
                "FedBal/Thresh": self.new_thresh, 
                "FedBal/MeanMaxLoss": lh, 
                "round": round_ix
            })
            self.reset()
        except Exception as ex:
            print("FedBalThreshController_update:", ex)
        finally:
            self.lock.release()
        return self.new_thresh

    def reset(self):
        self.min_list = []
        self.max_list = []
        self.sum = 0
        self.total_sample_num = 0

    def add_meta(self, meta):
        self.lock.acquire(blocking=True)
        try:
            self.min_list.append(meta[0])
            self.max_list.append(meta[1])
            self.sum += meta[2]
            self.total_sample_num += meta[3]
        except Exception as ex:
            print("FedBalThreshController_add_meta:", ex)
        finally:
            self.lock.release()
