#usage: l4_run.py
import tempfile
import os
import itertools
import logging
import numpy as np
from time import time
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, load, dump
import matplotlib.pylab as plt
import sys

from project_tools import *

class Kmeans(object):
    def __init__(self, k=1):
        self.k = k

    def getAverageDistance(self, data):
        dists = np.zeros((len(self.centroids),))
        for ix, centroid in enumerate(self.centroids):
            temp = data[(self.labels == ix)[:data.shape[0]]]
            dist = 0
            for i in temp:
                dist += np.linalg.norm(i - centroid)
            dists[ix] = dist/len(temp)
            
        return dists

    def predict(self, data):
        predictions = []
        for d in data:
            distance = []
            for centroid in self.centroids:
                distance.append(np.linalg.norm(d - centroid))
            predictions.append(np.argmin(distance))

        return predictions

    def getLabels(self):
        return self.labels

    def train(self, data, verbose=1):
        shape = data.shape
        ranges = np.zeros((shape[1], 2))
        centroids = np.zeros((shape[1], 2))
        for dim in range(shape[1]):
            ranges[dim, 0] = np.min(data[:,dim])
            ranges[dim, 1] = np.max(data[:,dim])
        if verbose == 1:
        centroids = np.zeros((self.k, shape[1]))
        for i in range(self.k):
            for dim in range(shape[1]):
                centroids[i, dim] = np.random.uniform(ranges[dim, 0], ranges[dim, 1], 1)
        if verbose == 1:
            plt.scatter(data[:,0], data[:,1])
            plt.scatter(centroids[:,0], centroids[:,1], c = 'r')
            plt.show()
        count = 0
        while count < 100:
            count += 1
            if verbose == 1:
            distances = np.zeros((shape[0],self.k))
            for ix, i in enumerate(data):
                for ic, c in enumerate(centroids):
                    distances[ix, ic] = np.sqrt(np.sum((i-c)**2))
            labels = np.argmin(distances, axis = 1)
            new_centroids = np.zeros((self.k, shape[1]))
            for centroid in range(self.k):
                temp = data[labels == centroid]
                if len(temp) == 0:
                    return 0
                for dim in range(shape[1]): 
                    new_centroids[centroid, dim] = np.mean(temp[:,dim])
            if verbose == 1:
                plt.scatter(data[:,0], data[:,1], c = labels)
                plt.scatter(new_centroids[:,0], new_centroids[:,1], c = 'r')
                plt.show()
            if np.linalg.norm(new_centroids - centroids) < np.finfo(float).eps:
                break
            centroids = new_centroids
        self.centroids = centroids
        self.labels = labels

        return 1

class ComponentNamesError(Exception):
    pass

class LabelsError(Exception):
    pass

from sompy.decorators import timeit
from sompy.codebook import Codebook
from sompy.neighborhood import NeighborhoodFactory
from sompy.normalization import NormalizerFactory

class SOM_factory(object):
    @staticmethod
    def build(data,
              mapsize=None,
              mask=None,
              mapshape='planar',
              lattice='rect',
              normalization='var',
              initialization='pca',
              neighborhood='gaussian',
              training='batch',
              name='sompy',
              component_names=None):
        if normalization:
            normalizer = NormalizerFactory.build(normalization)
        else:
            normalizer = None
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        return SOM(data, neighborhood_calculator, normalizer, mapsize, mask,
                   mapshape, lattice, initialization, training, name, component_names)


class SOM(object):

    def __init__(self,
                 data,
                 neighborhood,
                 normalizer=None,
                 mapsize=None,
                 mask=None,
                 mapshape='planar',
                 lattice='rect',
                 initialization='pca',
                 training='batch',
                 name='sompy',
                 component_names=None):
        self._data = normalizer.normalize(data) if normalizer else data
        self._normalizer = normalizer
        self._dim = data.shape[1]
        self._dlen = data.shape[0]
        self._dlabel = None
        self._bmu = None

        self.name = name
        self.data_raw = data
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization
        self.mask = mask or np.ones([1, self._dim])
        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize
        self.codebook = Codebook(mapsize, lattice)
        self.training = training
        self._component_names = self.build_component_names() if component_names is None else [component_names]
        self._distance_matrix = self.calculate_map_dist()

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('dims error')

    def build_component_names(self):
        cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
        return np.asarray(cc)[np.newaxis, :]

    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._dlabel = label

    def build_data_labels(self):
        cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
        return np.asarray(cc)[:, np.newaxis]

    def calculate_map_dist(self):
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix

    @timeit()
    def train(self,
              n_job=1,
              shared_memory=False,
              verbose='info',
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf):
        logging.root.setLevel(
            getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info("START TRAINING LOOP")

        if self.initialization == 'random':
            self.codebook.random_initialization(self._data)

        elif self.initialization == 'pca':
            self.codebook.pca_linear_initialization(self._data)

        self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        logging.debug(
            " --------------------------------------------------------------")
        logging.info(" FIN ERROR: %f" % np.mean(self._bmu[1]))

    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Rough approx...")

        ms, mpd = self._calculate_ms_and_mpd()

        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen
        
        trainlen=int(trainlen*trainlen_factor)
        
        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info(" Detail approx...")

        ms, mpd = self._calculate_ms_and_mpd()

        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin # from radius fin in rough training
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin 
            
        trainlen=int(trainlen_factor*trainlen)
            
        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False):
        radius = np.linspace(radiusin, radiusfin, trainlen)

        if shared_memory:
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None

        fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

        logging.info(" init r: %f , final r: %f, samples count %d\n" %
                     (radiusin, radiusfin, trainlen))

        for i in range(trainlen):
            t1 = time()
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            bmu = self.find_bmu(data, njb=njob)
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu,
                                                                neighborhood)

            qerror = (i + 1, round(time() - t1, 3),
                      np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2))) 
            logging.info(
                " iteration: %d | time:  %f, error: %f\n" %
                qerror)
            
        bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)
        self._bmu = bmu

    @timeit(logging.DEBUG)
    def find_bmu(self, input_matrix, njb=1, nth=1):
        dlen = input_matrix.shape[0]
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        if njb == -1:
            njb = cpu_count()

        pool = Pool(njb)
        chunk_bmu_finder = _chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part+1)*dlen // njb, dlen)

        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]
        b = pool.map(lambda chk: chunk_bmu_finder(chk, self.codebook.matrix, y2, nth=nth), chunks)
        pool.close()
        pool.join()
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu

    @timeit(logging.DEBUG)
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        row = bmu[0].astype(int)
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))
        S = P.dot(training_data)

        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)

    def project_data(self, data):
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        labels = np.arange(0, self.codebook.matrix.shape[0])
        clf.fit(self.codebook.matrix, labels)

        data = self._normalizer.normalize_by(self.data_raw, data)

        return clf.predict(data)

    def predict_by(self, data, target, k=5, wt='distance'):
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != target]
        x = self.codebook.matrix[:, indX]
        y = self.codebook.matrix[:, target]
        n_neighbors = k
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(x, y)

        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indX]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indX], data)

        predicted_values = clf.predict(data)
        predicted_values = self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)
        return predicted_values

    def predict(self, x_test, k=5, wt='distance'):
        target = self.data_raw.shape[1]-1
        x_train = self.codebook.matrix[:, :target]
        y_train = self.codebook.matrix[:, target]
        clf = neighbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        x_test = self._normalizer.normalize_by(
            self.data_raw[:, :target], x_test)
        predicted_values = clf.predict(x_test)

        return self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)

    def find_k_nodes(self, data, k=5):
        from sklearn.neighbors import NearestNeighbors

        neighbor = NearestNeighbors(n_neighbors=k)
        neighbor.fit(self.codebook.matrix)

        return neighbor.kneighbors(
            self._normalizer.normalize_by(self.data_raw, data))

    def bmu_ind_to_xy(self, bmu_ind):
        rows = self.codebook.mapsize[0]
        cols = self.codebook.mapsize[1]
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def cluster(self, n_clusters=8):
        import sklearn.cluster as clust
        cl_labels = clust.KMeans(n_clusters=n_clusters).fit_predict(
            self._normalizer.denormalize_by(self.data_raw,
                                            self.codebook.matrix))
        self.cluster_labels = cl_labels

        return cl_labels

    def predict_probability(self, data, target, k=5):
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indx = ind[ind != target]
        x = self.codebook.matrix[:, indx]
        y = self.codebook.matrix[:, target]

        clf = neighbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indx]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indx], data)

        weights, ind = clf.kneighbors(data, n_neighbors=k,
                                      return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self.codebook.matrix[ind, target])
        labels[labels >= 0] = 1
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_activation(self, data, target=None, wt='distance'):
        weights, ind = None, None

        if not target:
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.codebook.nnodes)
            labels = np.arange(0, self.codebook.matrix.shape[0])
            clf.fit(self.codebook.matrix, labels)

            data = self._normalizer.normalize_by(self.data_raw, data)
            weights, ind = clf.kneighbors(data)

            weights = 1./weights

        return weights, ind

    def calculate_topographic_error(self):
        bmus1 = self.find_bmu(self.data_raw, njb=1, nth=1)
        bmus2 = self.find_bmu(self.data_raw, njb=1, nth=2)
        topographic_error = None
        if self.codebook.lattice=="rect":
            bmus_gap = np.abs((self.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - self.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
            topographic_error = np.mean(bmus_gap != 1)
        elif self.codebook.lattice=="hexa":
            dist_matrix_1 = self.codebook.lattice_distances[bmus1[0].astype(int)].reshape(len(bmus1[0]), -1)
            topographic_error = (np.array(
                [distances[bmu2] for bmu2, distances in zip(bmus2[0].astype(int), dist_matrix_1)]) > 2).mean()
        return(topographic_error)

    def calculate_quantization_error(self):
        neuron_values = self.codebook.matrix[self.find_bmu(self._data)[0].astype(int)]
        quantization_error = np.mean(np.abs(neuron_values - self._data))
        return quantization_error

    def calculate_map_size(self, lattice):
        D = self.data_raw.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        VS = np.linalg.eig(A)
        eigval = sorted(np.linalg.eig(A)[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            size1 = min(munits, round(np.sqrt(munits / ratio)))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]


def _chunk_based_bmu_find(input_matrix, codebook, y2, nth=1):
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    bmu = np.empty((dlen, 2))

    blen = min(50, dlen)
    i0 = 0

    while i0+1 <= dlen:
        low = i0
        high = min(dlen, i0+blen)
        i0 = i0+blen
        ddata = input_matrix[low:high+1]
        d = np.dot(codebook, ddata.T)
        d *= -2
        d += y2.reshape(nnodes, 1)
        bmu[low:high+1, 0] = np.argpartition(d, nth, axis=0)[nth-1]
        bmu[low:high+1, 1] = np.partition(d, nth, axis=0)[nth-1]
        del ddata

    return bmu


dlen = 200
n_cluster = 4

Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] - .22*np.random.rand(dlen,1))[:,0]
Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
Data2.values[:,1] = (-.9*Data2.values[:,0][:,np.newaxis] + .992*np.random.rand(dlen,1))[:,0]
Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
Data3.values[:,1] = (.3*Data3.values[:,0][:,np.newaxis] - 10*.099*np.random.rand(dlen,1))[:,0]
Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + +.1*np.random.rand(dlen,1))[:,0]

Data1 = np.concatenate((Data1,Data2,Data3,Data4))
plt.scatter(Data1[:,0], Data1[:,1])

mapsize = [20,20]
som = SOM_factory.build(Data1, mapsize) 
som.train(n_job=1, verbose='info')
topographic_error = som.calculate_topographic_error()
quantization_error = np.mean(som._bmu[1])
print ("topographic_error = %s; quantization_error = %s" % (topographic_error, quantization_error))
