from __future__ import division
import pandas as pd
import numpy as np
import json
import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
import os
import random
import math
import matplotlib as plt
import time
import sys
import logging
from matplotlib import cm

np.set_printoptions(precision=10)
pd.set_option('display.precision',10)

#  create logger with 'spam_application'
logger = logging.getLogger('Buckshot')
logger.setLevel(logging.DEBUG)
#  create file handler which logs even debug messages
fh = logging.FileHandler('results_test.log')
fh.setLevel(logging.DEBUG)
#  create formatter and add it to the handlers
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
#  add the handlers to the logger
logger.addHandler(fh)


try:
    with open("buckshot.cfg") as f:
        config = json.load(f)
    DPATH = config.get("data_path")
    DATASET = config.get("dataset")
except Exception as e:
    print(e)
    print("Make sure the config file data_path is setup.")


class Cluster(object):
    """A Cluster of similar values(tuples from adult.arff).

    Attributes
        values (pd.Series, pd.DataFrame): similar data grouped together
        right_cluster (Cluster): right leaf Cluster
        left_cluster (Cluster):  left leaf Cluster
        verbose (Bool): whether to display anything
    """

    def __init__(self, values, centroid=None, nominal=None, numerical=None,
                right_cluster=None, left_cluster=None, verbose=False):
        # data frame for values
        self.values = pd.DataFrame([])
        # if the type of values is a series, array, or raw data
        if type(values) != type(self.values):
            data = np.array([values.values])
            self.values = pd.DataFrame(data, columns=values.index.values, index=[values.name])
            self.values = self.values.convert_objects(convert_numeric=True)
            self.centroid = values
        else:
            self.values = values.copy()

        #leaf clusters
        self.right = right_cluster
        self.left = left_cluster

        if (nominal == None) or (numerical == None):
            for col in self.values.columns:
                current = self.values[col]
                if current.dtype == 'float64':
                    self.numerical.append(col)
                else:
                    self.nominal.append(col)
        else:
            self.nominal = nominal
            self.numerical = numerical

        if verbose is True:
            print("Centroid:", self.centroid)

    def merge_clusters(self, new_cluster):
        """Merges two clusters together via Cluster.add()

        Attributes
            new_cluster (Cluster): cluster to add
        """
        old_left = self.left
        self.left = self.__class__(self.values,
                                    centroid=self.centroid,
                                    nominal=self.nominal,
                                    numerical=self.numerical,
                                    right_cluster=self.right,
                                    left_cluster=old_left)
        self.right = new_cluster
        self.add(new_cluster.get_values())

    def add(self, new_values):
        """Adds one cluster's values or a pd.Series to this.

        Attributes:
            new_values (pd.DataFrame, pd.Series): dataFrame to concat onto self.values
        """
        try:
            self.values = self.values.append(new_values)
            self.calculate_centroid()
        except Exception as e:
            logger.warn(e)
            logger.warn("\n")
            logger.warn(self.values)
            logger.warn("Centroid\n")
            logger.warn(self.centroid)
            raise(e)

    def calculate_centroid(self):
        """Determines the centroid by combining the mean and modes of self.values.

        Null values are replaced by the first column's index of those values
            to prevent a concat with pd.nan (null), which create a dataFrame
            with indexes that are column's names and many many more pd.nan values.
        """
        mean = self.values[self.numerical].mean()
        mode = self.values[self.nominal].mode().iloc[0]
        for i,_ in  mode.isnull()[mode.isnull() == True].iteritems():
            mode[i] = self.values.iloc[0][i]
        if np.any(mode.isnull()):
            logger.warn("ModeNanError\n")
            logger.warn(mode)
            raise Exception("Null in mode.")

        self.centroid = pd.concat([mean,mode], axis=0)

    def get_values(self):
        return self.values

    def length(self):
        return len(self.values.index)


class Buckshot(object):
    """Performs Buckshot clustering (HAC + K-Means)

    Linkage = Centroid

    Attributes
        k (int): Number of clusters.
    """

    data_name = 'adult-big'
    output = 'adult.out'
    class_label = 'class'

    def __init__(self, k=10):
        self.dataset = DATASET
        self.k = k
        self.time = time.time()

    def run(self):
        """Runs preprocessing, hac, and k_means."""
        self.preprocess()
        self.hac()
        self.k_means()

    def preprocess(self):
        """Runs all preprocessing functions."""
        self.load_df()
        self.drop_feature(labels='fnlwgt')
        self.replace_missing_values()
        self.normalize_data()

    def print_report(self):
        logger.warn("Samples: %d\n" % len(self.df.index))
        logger.warn("Class: %s" % self.class_label)
        logger.warn("\nClusters (k): %d\n" % self.k)
        sys.stdout.flush()
        self.cluster_sizes()
        self.ratios_and_inter_distance()
        self.print_time()

    def load_df(self):
        """Converts the dataset to pandas and numpy friendly file format."""
        if 'arff' in self.dataset:
            self.df = load_arff_to_df(self.dataset)
            self.df = self.df.rename(columns=lambda x: x.replace(':', ''))
            self.df = self.df.rename(columns=lambda x: x.replace('-', '_'))

    def drop_feature(self, labels=['fnlwgt']):
        self.df = self.df.drop(labels, axis=1)

    def replace_missing_values(self):
        """Replaces missing values with mean and mode."""
        self.numerical = []
        self.nominal = []
        for col in self.df.columns:
            current = self.df[col]
            if current.dtype == 'int64':
                current.fillna(current.mean(), inplace=True)
                self.numerical.append(col)
            else:
                current.fillna(self.df[col].mode()[0], inplace=True)
                self.nominal.append(col)
            self.df[col] = current

    def normalize_data(self):
        """Normalizes all numerical values in dataFrame."""
        # self.df['class'] = self.df['class'].apply(convert_class)
        int_df = self.df[self.numerical]
        int_df = (int_df - int_df.mean()) / (int_df.max() - int_df.min())
        self.df[self.numerical] = int_df

    def get_random_samples(self):
        """Creates random sampled dataframe equal to sqrt(n)."""
        n = len(self.df.index)
        self.sqrt_n = int(math.sqrt(n))
        rows = random.sample(self.df.index, self.sqrt_n)
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
        n = len(self.clusters)
        starting = len(self.clusters)
        total = starting - self.k
        # could be a for loop, for _ in range(0,n-self.k)
        while(n > self.k):
            # get index of the two clusters that are closest together
            num_merged = starting - n
            progress = num_merged/starting * 100
            sys.stdout.write("\rHAC  %0.2f%% Complete. Clusters: %d Clusters Merged: %d of %d. Time Elapsed: %.02f"
                                % (progress, n, num_merged, total, (time.time() - self.time)))
            sys.stdout.flush()
            minimum = np.where(self.matrix == self.matrix.min())
            min_x = minimum[0][0]
            min_y = minimum[1][0]
            #print "Minimum: (", min_x,",",min_y,") Distance: ", self.matrix[min_x][min_y]
            # remove cluster to be merged into clusters[min_x]
            to_merge = self.clusters.pop(min_y)
            n -= 1
            self.clusters[min_x].merge_clusters(to_merge)
            self.update_matrix()

    def init_matrix(self, ignore_class=True):
        """Initializes size [sqrt(n),sqrt(n)] similarity matrix and creates sqrt(n) clusters.

        New Class Members
            self.clusters (array): array of all clusters
            self.matrix (np.array): [sqrt(n),sqrt(n)] distance matrix
        """
        # Start with sqrt(n) random clusters
        n = len(self.random.index)
        matrix = np.empty(shape=[n,n])

        self.clusters = []
        index = range(0,n)
        # Create the initial distance matrix
        for n1 in index:
            row1 = self.random.iloc[n1]
            current_cluster = Cluster(values=row1,
                                      centroid=row1,
                                      numerical=self.numerical,
                                      nominal=self.nominal)
            for n2 in index:
                row2 = self.random.iloc[n2]
                dist = distance(row1,row2)
                matrix[n1][n2] = dist

            self.clusters.append(current_cluster)
        # minimums won't return self -> 0,0, 1,1, 2,2...
        np.fill_diagonal(matrix, np.inf)
        self.matrix = matrix

    def update_matrix(self, ignore_class=True):
        """Updates the similarity matrix to account for clusters with > 1 tuple."""
        # create new distance matrix
        n = len(self.clusters)
        progress =  (self.sqrt_n - n) / (self.sqrt_n - self.k)
        matrix = np.empty(shape=[n,n])

        for cluster1 in self.clusters:
            r1 = cluster1.centroid
            i1 = self.clusters.index(cluster1)
            for cluster2 in self.clusters:
                r2 = cluster2.centroid
                i2 = self.clusters.index(cluster2)
                matrix[i1][i2] = distance(r1, r2, ignore_class=ignore_class)

        np.fill_diagonal(matrix, np.inf)
        del self.matrix
        self.matrix = matrix

    def k_means(self):
        """Clusters the remaining data into self.clusters based on the closest cluster."""
        n = len(self.remaining)

        #  go in c
        for i in range(0,n):
            row = self.remaining.iloc[i]
            distances = np.array([distance(row, clust.centroid) for clust in self.clusters])
            min_dist = np.where(distances == distances.min())[0][0]
            self.clusters[min_dist].add(row)
            progress = i/n * 100
            sys.stdout.write("\rK-Means  %0.2f%% Complete. Iteration %s of %s. Time Elapsed: %0.2fs            "
                                % (progress, i, n, time.time() - self.time))
            sys.stdout.flush()


    def cluster_sizes(self):
        sizes = np.array([len(clust.values.index) for clust in self.clusters])
        logger.warn("Cluster Sizes\n")
        logger.warn("Largest: %d\n" % sizes.max(axis=0))
        logger.warn("Smallest: %d\n" % sizes.min(axis=0))
        logger.warn("Average: %0.0f\n" % np.mean(sizes))

    def ratios_and_inter_distance(self):
        inter_dist = 0
        self.avg_ratio = {'>50K':0, '<=50K':0}
        self.ratios = pd.DataFrame([], cols=['cluster', 'n', '>50K', '<=50K'])
        # Inter-Distance + Average Ratio
        for c in self.clusters:
            dist = 0
            clust_number = self.clusters.index(c)
            ratios['cluster'][clust_number] = 'cluster' + str(clust_number)
            ratios['n'][clust_number] = len(c.values.index)
            # Intra-Distance
            for i in range(0, len(c.values.index)):
                row = c.values.iloc[i]
                dist += distance(row, c.centroid)
            inter_dist += dist
            # Cluster Ratio
            logger.warn("Cluster %s  " % str(clust_number))
            for i,x in c.values[self.class_label].value_counts().iteritems():
                current_ratio = x/len(c.values.index)
                logger.warn("Label: %s Ratio: %f" % (i,current_ratio))
                self.avg_ratio[i][clust_number] += current_ratio
                self.ratios[i] = x

        logger.warn("Average Class Ratio")
        for i in self.avg_ratio:
            logger.warn("%s: %f\n" % (i,(self.avg_ratio[i]/len(self.clusters))))
        logger.warn("Inter-Cluster Distance: %f\n" % inter_dist)
        self.inter_dist = inter_dist
        self.avg_ratio = avg_ratio

    def print_time(self):
        stop = time.time()
        logger.warn("Time: %0.2fs\n\n" % (self.time - stop))

    def plot_clusters(self, x_axis='age', y_axis='hours_per_week'):
        x_axis = 'age'
        y_axis = 'hours_per_week'

        self.colors = ['#99b433', '#00a300', '#1e7145', '#ff0097', '#9f00a7',
                          '#7e3878', '#603cba', '#1d1d1d', '#eff4ff',
                          '#2d89ef', '#2b5797', '#ffc40d', '#e3a21a', '#da532c',
                          '#ee1111', '#b91d47','#99b433', '#00a300',
                          '#1e7145', '#ff0097', '#9f00a7',
                          '#7e3878', '#603cba', '#1d1d1d', '#eff4ff',
                          '#2d89ef', '#2b5797', '#ffc40d', '#e3a21a', '#da532c',
                          '#ee1111', '#b91d47','#99b433', '#00a300',
                          '#1e7145', '#ff0097', '#9f00a7',
                          '#7e3878', '#603cba', '#1d1d1d', '#eff4ff',
                          '#2d89ef', '#2b5797', '#ffc40d', '#e3a21a', '#da532c',
                          '#ee1111', '#b91d47','#99b433', '#00a300',
                          '#1e7145', '#ff0097', '#9f00a7',
                          '#7e3878', '#603cba', '#1d1d1d', '#eff4ff',
                          '#2d89ef', '#2b5797', '#ffc40d', '#e3a21a', '#da532c',
                          '#ee1111', '#b91d47']

        ax = self.clusters[0].values.plot(kind='scatter', color=self.colors[0],
                                          label="Cluster1", x=x_axis, y=y_axis)
        fig = ax.get_figure()
        fig.savefig("graphs/Cluster1-%d-buckshot_%s_%s.png" % (self.k, x_axis, y_axis))
        for i in range(1,len(self.clusters)):
            label = "Cluster" + str(i+1)
            # add to total cluster graph
            self.clusters[i].values.plot(kind='scatter', x=x_axis, y=y_axis,
                          color=self.colors[i], label=label, ax=ax)
            # create graph for just this cluster
            dx = self.clusters[i].values.plot(kind='scatter', x=x_axis, y=y_axis,
                                      label=label, color=self.colors[i])
            centroid = pd.DataFrame([self.clusters[i].centroid])
            centroid.plot(kind='scatter', x=x_axis, y=y_axis, label='Centroid', colors='k', sharex=True, sharey=True,ax=dx)
            centroid.plot(kind='scatter', x=x_axis, y=y_axis, label=None, colors='k', sharex=True, sharey=True,ax=ax)

            fig = dx.get_figure()
            fig.savefig("graphs/%s-%d-buckshot_%s_%s.png" % (label, self.k, x_axis, y_axis))
        fig = ax.get_figure()
        fig.set_size_inches(18.5,10.5)
        fig.savefig('graphs/%d-buckshot_%s_%s.png' % (self.k, x_axis, y_axis), dpi=100)

    def plot_ratios(self):
        ax=self.ratios(kind='line', x='cluster', y='n', label='Values ')
def distance(x, y, squared=False, ignore_class=True):
        """Returns Euclidean distance of two matrices/vectors.

        Attributes:
            x (ndarry, pd.Series, pd.DataFrame): first row to compare distance
            y (np.ndarry, pd.Series, pd.DataFrame): second row to compare distance
            squared (Bool): if True calculates the squared euclidean distance
                mainly for intra-cluster distance

        Returns:
            dist (float64): distance between x and y

        """
        try:
            dist = 0
            for i,v in x.iteritems():
                if ignore_class:
                    if i is 'class':
                        continue
                j = v
                # make sure to get correct index if series mooved around
                k = y[i]
                if type(j) == str:
                    if j == k:
                        k = 0
                        j = 0
                    else:
                        j = 0
                        k = 1
                dist += (j-k)**2
            if not squared:
                dist = math.sqrt(dist)
            return dist
        except KeyError as e:
            logger.debug(e)
            logger.debug(x)
            logger.debug(y)
            raise(e)
        except TypeError as t:
            raise(t)


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
        # print(e)
        pass
    finally:
        df = pd.read_csv(temp_csv, na_values='?', float_precision=12)
        # os.remove(temp_csv)
        return df


class BuckshotFileError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return repr(self.msg)


def test(runs=5):
    for i in range(2,runs+5):
        b = Buckshot(i*5)
        b.run()
        b.print_report()


if __name__ == '__main__':
    b = Buckshot(10)
    b.run()
    b.print_report()
    test()
