from typing import Union
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils.metric_utils.calc_metric import calc_acc, calc_conf_matrix, calc_q_acc

class IterativeModel:
    def __init__(self, model_params) -> None:
        self.rng = torch.Generator()
        self.model_params = model_params


    def run(self, train_ts, test_ts, val_ts, data_df, meta_df, save=False, plot=False, step_size=25):
        
        seed, rate, iters = self.model_params['seed'], self.model_params['rate'], self.model_params['iters']
        self.rng.manual_seed(seed)
        
        S, Q = data_df.shape[0], data_df.shape[1] # student param dimension; question param dimension
        params, history, last_epoch = self.train(train_ts, val_ts, test_ts, S, Q, rate, iters, step_size)

        probabilities, predictions = self.predict(test_ts, params)
        performance = self.calc_performance(test_ts, predictions)
        correctness_ts = self.get_correctness(test_ts, predictions)

        results = {'res': performance, 'hyperparams': {'seed': seed, 'rate': rate, 'iters': last_epoch},
                            'params': params, 'history': history, 'correctness': correctness_ts, 'probit': probabilities, 'test_ts': test_ts}

        if plot:
            self.plot(history, last_epoch, step_size, plot)

        if save:
            torch.save(results, save)

        return results


    def train(self, train_ts: torch.Tensor):
        ...


    def predict(self, test_ts: torch.Tensor, params_ts) -> Union[torch.Tensor, torch.Tensor]:
        ...


    def calc_performance(self, test_ts, predictions):
        acc = calc_acc(test_ts[0], predictions)
        conf_matrix = calc_conf_matrix(test_ts[0], predictions)
        q_acc = calc_q_acc(test_ts[0], predictions, test_ts[2])
        performance = {'acc': acc, 'conf': conf_matrix, 'q_acc': q_acc}
        return performance


    def get_correctness(self, data_ts, predictions):
        correctness = torch.eq(data_ts[0], predictions)
        correctness_ts = torch.clone(data_ts)
        correctness_ts[0] = correctness
        return correctness_ts


    def plot(self, history, iters, step_size, plot):
        plt.plot(range(iters), history['avg train nll'])
        plt.plot(np.arange(0, iters, step_size), history['avg test nll'])
        plt.plot(np.arange(0, iters, step_size), history['avg val nll'])
        plt.plot(np.arange(0, iters, step_size), history['train acc']/100)
        plt.plot(np.arange(0, iters, step_size), history['test acc']/100)
        plt.plot(np.arange(0, iters, step_size), history['val acc']/100)
        plt.title('nll and acc')
        plt.xlabel('epoch')
        plt.legend(['Train nll', 'Test nll', 'Validation nll', 'Train acc', 'Test acc', 'Val acc'])
        # plt.legend(['Train nll', 'Validation nll', 'Train acc', 'Val acc'])
        plt.show()
        plt.savefig(plot)


    def print_iter_res(self, epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc):
        dp = 3
        print(f'{epoch} train: {round(train_nll.item())} {round(train_acc,dp)}; '
                            f'val: {round(val_nll.item())} {round(val_acc,dp)}; '
                            f'test: {round(test_nll.item())} {round(test_acc,dp)}')


    def print_iter_res_wo_val(self, epoch, train_nll, test_nll, train_acc, test_acc):
        dp = 5
        print(f'{epoch} train: {round(train_nll.item(),dp)} {round(train_acc,dp)}; '
                            f'test: {round(test_nll.item(),dp)} {round(test_acc,dp)}')
