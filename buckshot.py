import pandas as pd
import numpy as np
import json
import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
import os
import random
import math
import matplotlib as plt

np.set_printoptions(precision=10)
pd.set_option('display.precision',10)

try:
    with open("buckshot.cfg") as f:
        config = json.load(f)
    DPATH = config.get("data_path")
    DATASET = config.get("dataset")
except Exception as e:
    print(e)
    print("Make sure the config file data_path is setup.")


class Cluster(object):
    """Singular cluster."""

    def __init__(self, values, right_cluster=None, left_cluster=None, verbose=False):
        self.right = right_cluster
        self.left = left_cluster

        self.values = pd.DataFrame([])
        if type(values) != type(self.values):
            data = np.array([values.values])
            self.values = pd.DataFrame(data, columns=values.index.values, index=[values.name])
            self.values = self.values.convert_objects(convert_numeric=True)
        else:
            self.values = values.copy()

        self.nominal = []
        self.continuous = []
        for col in self.values.columns:
            current = self.values[col]
            if current.dtype == 'float64':
                self.continuous.append(col)
            else:
                self.nominal.append(col)
        self.calculate_centroid()

        if verbose is True:
            print("Centroid:", self.centroid)

    def merge_clusters(self, new_cluster):
        old_left = self.left
        self.left = self.__class__(self.values,
                                    right_cluster=self.right,
                                    left_cluster=old_left,
                                    verbose=True)
        self.right = new_cluster
        self.add(new_cluster.get_values())
        #self.mean = pd.concat([self.values[self.continuous].mean(), self.values[self.nominal].mode()])

    def add(self, new_values):
        self.values = pd.concat([self.values, new_values])

    def calculate_centroid(self):
        mean = self.values[self.continuous].mean()
        mode = self.values[self.nominal].mode()
        if len(mode.index) == 0:
            mode = self.values[self.nominal].iloc[0]
        #concat into dataframe and grab the only value in series
        self.centroid = pd.concat([mean,mode], axis=1)
        print(mean,mode)

    def get_values(self):
        return self.values

    def intra_distance(self):
        """Finds the intra-cluster variance using squared euclidean distance."""
        dist = 0
        for i in range(0, len(self)):
            row = self.values.iloc[i]
            dist += distance(row, self.centroid)

    @classmethod
    def len(cls):
        cls.length = len(cls.values)
        return cls.length


class Buckshot(object):
    """Performs Buckshot clustering (HAC + K-Means)

    Linkage = Centroid


    Attributes
        k (int): Number of clusters.
    """

    data_name = 'adult-big'
    output = 'adult.out'

    def __init__(self, k=10):
        self.dataset = DATASET
        self.k = k

    def run(self):
        """Runs preprocessing, hac, and k_means."""
        self.preprocess()
        self.hac()
        self.k_means()

    def preprocess(self):
        """Runs all preprocessing functions."""
        self.load_df()
        del self.df['fnlwgt']
        self.replace_missing_values()
        self.normalize_data()

    def load_df(self):
        """Converts the dataset to pandas and numpy friendly file format."""
        if 'arff' in self.dataset:
            self.df = load_arff_to_df(self.dataset)
            self.df = self.df.rename(columns=lambda x: x.replace(':', ''))
            self.df = self.df.rename(columns=lambda x: x.replace('-', '_'))

    def replace_missing_values(self):
        """Replaces missing values with mean and mode."""
        self.continuous = []
        self.nominal = []
        for col in self.df.columns:
            current = self.df[col]
            if current.dtype == 'int64':
                current.fillna(current.mean(), inplace=True)
                self.continuous.append(col)
            else:
                current.fillna(self.df[col].mode()[0], inplace=True)
                self.nominal.append(col)
            self.df[col] = current

    def normalize_data(self):
        """Normalizes all continuous values in dataFrame."""
        #self.df['class'] = self.df['class'].apply(convert_class)
        int_df = self.df[self.continuous]
        int_df = (int_df - int_df.mean()) / (int_df.max() - int_df.min())
        self.df[self.continuous] = int_df

    def get_random_samples(self):
        """Creates random sampled dataframe equal to sqrt(n)."""
        n = len(self.df.index)
        sqrt_n = int(math.sqrt(n))
        rows = random.sample(self.df.index, sqrt_n)
        self.random = self.df.ix[rows]
        self.remaining = self.df.drop(rows)

    def hac(self):
        """Clusters sqrt(n) random samples into self.k clusters using heirarchical clustering.

        Algorithm
            1. Creates a similarity matrix of sqrt(n) random samples
                with Euclidean Distance
                    See: Buckshot.init_matrix()
            2. Merges two closest clusters
            3. Recalculates similarity matrix using averages of clusters
                    See: Buckshot.update_matrix()
            4. If the number of clusters is k then stop.

        Linkage (Centroid): Merges clusters with closest mean distance

        """
        self.get_random_samples()
        self.init_matrix()

        while(len(self.clusters) > self.k):
            #get index of the two clusters that are closest together
            minimum = np.where(self.matrix == self.matrix.min())[0]
            min_x = minimum[0]
            min_y = minimum[1]
            #remove cluster to be merged into clusters[min_x]
            to_merge = self.clusters.pop(min_y)
            self.clusters[min_x].merge_clusters(to_merge)
            self.update_matrix()

    def init_matrix(self):
        """Initializes size [sqrt(n),sqrt(n)] similarity matrix and creates sqrt(n) clusters.

        New Class Members
            self.clusters (array): array of all clusters
            self.matrix (np.array): [sqrt(n),sqrt(n)] distance matrix
        """
        #Start with sqrt(n) random clusters
        n = len(self.random.index)
        matrix = np.empty(shape=[n,n])

        self.clusters = []
        index = range(0,n)
        for n1 in index:
            row1 = self.random.iloc[n1]
            current_cluster = Cluster(values=row1)
            #create a distance matrix
            for n2 in index:
                row2 = self.random.iloc[n2]
                dist = distance(row1,row2)
                matrix[n1][n2] = dist

            self.clusters.append(current_cluster)
        #minimums won't return self -> 0,0, 1,1, 2,2...
        np.fill_diagonal(matrix, np.inf)
        self.matrix = matrix

    def update_matrix(self):
        """Updates the similarity matrix to account for clusters with > 1 tuple."""
        #create new distance matrix
        n = len(self.clusters)
        matrix = np.empty(shape=[n,n])
        print("Clusters: %s" % str(n))

        for cluster1 in self.clusters:
            r1 = cluster1.centroid
            i1 = self.clusters.index(cluster1)
            for cluster2 in self.clusters:
                r2 = cluster2.centroid
                i2 = self.clusters.index(cluster2)
                matrix[i1][i2] = distance(r1, r2)

        np.fill_diagonal(matrix, np.inf)
        del self.matrix
        self.matrix = matrix

    def k_means(self):
        """Clusters the remaining data into self.clusters based on the closest cluster."""
        n = len(self.remaining)
        means = pd.Series([clust.centroid for clust in self.clusters])
        print(means)
        #for clust in self.clusters:
           # means.append(clust.centroid)

        distances = []
        for i in range(0,n):
            row = self.remaining.iloc[i]
            for mean in means:
                distances.append(distance(row,mean))
            distances = np.array(distances)
            min_dist = distances.where(distances == distances.min())
            print(min_dist)
            self.clusters[min_dist].add(row)


def distance(x, y, squared=False):
        """Returns Euclidean distance of two matrices/vectors.

        Attributes:
            x (ndarry, pd.Series, pd.DataFrame): first row to compare distance
            y (np.ndarry, pd.Series, pd.DataFrame): second row to compare distance
            squared (Bool): if True calculates the squared euclidean distance
                mainly for intra-cluster distance

        Returns:
            dist (float64): distance between x and y

        """
        dist = 0
        for i,v in x.iteritems():
            j = v
            k = y[i]
            if type(j) == str:
                if j == k:
                    k = 0
                    j = 0
                else:
                    j = 0
                    k = 1
            dist += (j-k)**2
            #time.sleep(.01)
        if squared == False:
            dist = math.sqrt(dist)
        #print(dist)
        return dist


def load_arff_to_df(filepath):
    """Loads arff into csv then pandas reads that into a dataframe.

    Returns:
        df (pandas.DataFrame): Loaded DataFrame from filepath
    """
    temp_csv = filepath[:-5] + '.csv'
    try:
        if os.path.isfile(temp_csv):
           raise BuckshotFileError("CSV exists. Skipping load_arff_to_df()")
        jvm.start(max_heap_size='512m')
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(filepath)
        saver = Saver(classname="weka.core.converters.CSVSaver")
        saver.save_file(data, temp_csv)
        jvm.stop()
    except BuckshotFileError as e:
        #print(e)
        pass
    finally:
        df = pd.read_csv(temp_csv, na_values='?', float_precision=12)
        #convert all ints to floats
        #os.remove(temp_csv)
        return df


class BuckshotFileError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


if __name__ == '__main__':
    b = Buckshot(50)
    #ab.scipy_test()
    b.run()
