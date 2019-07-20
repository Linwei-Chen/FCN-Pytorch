import os
import os.path as osp
import json
import torch

"""
Logger is built to log the information during the model train like loss, accuracy
ModelSaver is design to save the state_dict of model, optimizer, scheduler etc safetly
Author: Charles Chen
"""


class Logger:
    def __init__(self, save_path, json_name):
        self.create_dir(save_path)
        self.save_path = save_path
        self.json_name = json_name
        self.json_path = osp.join(save_path, json_name + '.json')
        # if .json file not exist create one
        if not osp.exists(self.json_path):
            with open(self.json_path, 'a') as f:
                json.dump({}, f)
        self.state = json.load(open(self.json_path, 'r'))
        # if 'max' not in self.state: self.state['max'] = {}

    def get_data(self, key):
        """
        :param key: key word of data
        :return: data[key]
        """
        if key not in self.state:
            print(f'*** find no {key} data!')
            return []
        else:
            return self.state[key]

    def get_max(self, key):
        """

        :param key:
        :return:
        """
        if key not in self.state:
            print(f'*** find no {key} data!')
            return float('-inf')
        try:
            # if key not in self.state['max']:
            return max(self.state[key])
        except Exception:
            print(f'sorry, cannot get the max of data {key}')
            return float('-inf')

    def log(self, key, data, show=False):
        if key not in self.state:
            self.state[key] = []
        else:
            self.state[key].append(data)
        if show:
            print(f'log key:{key} -> data:{data}')

    def save_log(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.state, f)
            print('*** Save log safely!')

    def visualize(self, key=None, range=None):
        """

        :param key: the key for dict to find data
        :param range: tuple, to get the range of data[key]
        :return:
        """
        if key is None:
            for key in self.state:
                data = self.state[key]
                self.save_training_pic(data=data,
                                       path=self.save_path, name=self.json_name,
                                       ylabel=key, xlabel='iteration')
        elif key not in self.state:
            print(f'*** find no data of {key}!')
        else:
            if range == None or not isinstance(range, tuple):
                self.save_training_pic(data=self.state[key],
                                       path=self.save_path, name=self.json_name,
                                       ylabel=key, xlabel='iteration')
            else:
                self.save_training_pic(data=self.state[key][range[0]:range[1]],
                                       path=self.save_path, name=self.json_name,
                                       ylabel=key, xlabel='iteration')

    @staticmethod
    def save_training_pic(data, path, name, ylabel, xlabel, smooth=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def moving_average_filter(data, n):
            assert smooth % 2 == 1
            res = [0 for i in data]
            length = len(data)
            for i in range(length):
                le = max(0, i - n // 2)
                ri = min(i + n // 2 + 1, length)
                s = sum(data[le:ri])
                l = ri - le
                res[i] = s / float(l)
            return res

        if isinstance(smooth, int):
            data = moving_average_filter(data, n=smooth)

        x = range(1, len(data) + 1)
        y = data
        fig, ax = plt.subplots()
        ax.plot(x, y)

        ax.set(xlabel=xlabel, ylabel=ylabel,
               title='{} {}'.format(name, ylabel))
        ax.grid()
        fig.savefig(path + '/{}_{}.png'.format(name, ylabel), dpi=330)
        plt.cla()
        plt.clf()
        plt.close('all')

    @staticmethod
    def create_dir(dir_path):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)


class ModelSaver:
    def __init__(self, save_path, name_list, strict_mode=False):
        self.create_dir(save_path)
        self.save_path = save_path
        self.name_dict = {name: osp.join(save_path, name + '.pkl') for name in name_list}
        self.strict_mode = strict_mode

    def load(self, name, model):
        if self.strict_mode:
            model.load_state_dict(torch.load(self.name_dict[name], map_location='cpu'))
            print(f'*** Loading {name} successfully')
        else:
            try:
                model.load_state_dict(torch.load(self.name_dict[name], map_location='cpu'))
            except Exception:
                print(f'*** Loading {name} fail!')
            else:
                print(f'*** Loading {name} successfully')

    def save(self, name, model):
        if self.strict_mode:
            self.save_safely(model.state_dict(), self.save_path, file_name=name + '.pkl')
            print(f'*** Saving {name} successfully')
        else:
            try:
                self.save_safely(model.state_dict(), self.save_path, file_name=name + '.pkl')
            except Exception:
                print(f'*** Saving {name} fail!')
            else:
                print(f'*** Saving {name} successfully')

    @staticmethod
    def save_safely(file, dir_path, file_name):
        """
        save the file safely, if detect the file name conflict,
        save the new file first and remove the old file
        """
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
            print('*** dir not exist, created one')
        save_path = osp.join(dir_path, file_name)
        if osp.exists(save_path):
            temp_name = save_path + '.temp'
            torch.save(file, temp_name)
            os.remove(save_path)
            os.rename(temp_name, save_path)
            print('*** find the file conflict while saving, saved safely')
        else:
            torch.save(file, save_path)

    @staticmethod
    def create_dir(dir_path):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)


if __name__ == '__main__':
    test_dict = {
        'lr': [0.1, 0.1, ],
        'time': [1, 1]
    }
    with open('./test_dict.json', 'w') as f:
        json.dump(test_dict, f)
    with open('./test_dict.json', 'r') as f:
        temp_dict = json.load(f)
        print(temp_dict)

    logger = Logger(save_path='./', json_name='test')
    logger.log(key='lr', data=0.1)
    print(logger.get_data('lr'))
    logger.save_log()
    logger.visualize()
