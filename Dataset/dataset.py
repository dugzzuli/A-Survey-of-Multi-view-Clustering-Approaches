import numpy as np
import linecache
from utils.utils import *

class Dataset(object):

    def __init__(self, config):
        self.Weight=config['Weight']
        self.View = config['View']
        self.WF = config['W']


        self.label_file = config['label_file']

        self.ViewData,self.W, self.Y,self.node_map  = self._load_data()

        self.num_nodes = self.ViewData[0].shape[0]
        self.num_classes = self.Y.shape[1]



        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False


    def _load_data(self):
        lines = linecache.getlines(self.label_file)
        lines = [line.rstrip('\n') for line in lines]


        #===========load label============
        node_map = {}
        label_map = {}
        Y = []
        cnt = 0
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[0]] = idx
            y = []
            for label in line[1:]:
                if label not in label_map:
                    label_map[label] = cnt
                    cnt += 1
                y.append(label_map[label])
            Y.append(y)
        self.num_classes = len(label_map)
        self.num_nodes = len(node_map)

        L = np.zeros((self.num_nodes, self.num_classes), dtype=np.int32)
        for idx, y in enumerate(Y):
            L[idx][y] = 1

        # =========load feature==========

        viewData=[]
        for i in range(len(self.View)):
            if(self.View[i].find("edges")>0 or self.View[i].find("walks")>0):
                VData = np.zeros((self.num_nodes, self.num_nodes))
                lines = linecache.getlines(self.View[i])
                lines = [line.rstrip('\n') for line in lines]
                for line in lines:
                    line = line.split(' ')
                    idx1 = node_map[line[0]]
                    idx2 = node_map[line[1]]
                    VData[idx2, idx1] = 1.0
                    VData[idx1, idx2] = 1.0
            else:
                VData=self.load_attr(node_map,self.View[i])
            viewData.append(VData)

            # =========load feature==========

        WF = []
        for i in range(len(self.WF)):
            W1 = np.zeros((self.num_nodes, self.num_nodes))
            lines = linecache.getlines(self.WF[i])
            lines = [line.rstrip('\n') for line in lines]
            for line in lines:
                line = line.split(' ')
                idx1 = node_map[line[0]]
                idx2 = node_map[line[1]]
                if(self.Weight):
                    W1[idx2, idx1] = float(line[2])
                    W1[idx1, idx2] = float(line[2])
                else:
                    W1[idx2, idx1] = 1.0
                    W1[idx1, idx2] = 1.0
            WF.append(W1)

        return viewData,WF, L,node_map

    def load_attr(self,node_map,feature_file,name='view'):
        lines = linecache.getlines(feature_file)
        lines = [line.rstrip('\n') for line in lines]

        num_features = len(lines[0].split(' ')) - 1
        Z = np.zeros((self.num_nodes, num_features), dtype=np.float32)
        print(name+' #,nodes {}, features {}, classes {}'.format(self.num_nodes,num_features, self.num_classes))

        for line in lines:
            line = line.split(' ')
            node_id = node_map[line[0]]
            Z[node_id] = np.array([float(x) for x in line[1:]])
        return Z



    def sample(self, batch_size, do_shuffle=True, with_label=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self._order)
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()
        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]

        for i in range(len(self.View)):
            mini_batch["V"+str(i+1)] = self.ViewData[i][cur_index]

        for i in range(len(self.W)):
            mini_batch["adj" + str(i + 1)] = self.W[i][cur_index][:, cur_index]

        mini_batch.cur_index = cur_index

        if with_label:
            mini_batch.Y = self.Y[cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True
        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):

        mini_batch = Dotdict()
        for i in range(len(self.View)):
            mini_batch["V"+str(i+1)] = self.ViewData[i][idx]

        for i in range(len(self.W)):
            mini_batch["adj" + str(i + 1)] = self.W[i][idx][:, idx]

        mini_batch.idx = idx
        return mini_batch
