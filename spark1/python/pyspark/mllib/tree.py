#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import

import random

from pyspark import SparkContext, RDD, since
from pyspark.mllib.common import callMLlibFunc, inherit_doc, JavaModelWrapper
from pyspark.mllib.linalg import _convert_to_vector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import JavaLoader, JavaSaveable

__all__ = ['DecisionTreeModel', 'DecisionTree', 'RandomForestModel',
           'RandomForest','CustomEnsembleModel',
           'CustomEnsemble', 'GradientBoostedTreesModel', 'GradientBoostedTrees']


class TreeEnsembleModel(JavaModelWrapper, JavaSaveable):
    """TreeEnsembleModel

    .. versionadded:: 1.3.0
    """
    @since("1.3.0")
    def predict(self, x):
        """
        Predict values for a single data point or an RDD of points using
        the model trained.

        .. note:: In Python, predict cannot currently be used within an RDD
            transformation or action.
            Call predict directly on the RDD instead.
        """
        if isinstance(x, RDD):
            return self.call("predict", x.map(_convert_to_vector))

        else:
            return self.call("predict", _convert_to_vector(x))

    @since("1.3.0")
    def numTrees(self):
        """
        Get number of trees in ensemble.
        """
        return self.call("numTrees")

    @since("1.3.0")
    def totalNumNodes(self):
        """
        Get total number of nodes, summed over all trees in the ensemble.
        """
        return self.call("totalNumNodes")

    def __repr__(self):
        """ Summary of model """
        return self._java_model.toString()

    @since("1.3.0")
    def toDebugString(self):
        """ Full model """
        return self._java_model.toDebugString()


class DecisionTreeModel(JavaModelWrapper, JavaSaveable, JavaLoader):
    """
    A decision tree model for classification or regression.

    .. versionadded:: 1.1.0
    """
    @since("1.1.0")
    def predict(self, x):
        """
        Predict the label of one or more examples.

        .. note:: In Python, predict cannot currently be used within an RDD
            transformation or action.
            Call predict directly on the RDD instead.

        :param x:
          Data point (feature vector), or an RDD of data points (feature
          vectors).
        """
        if isinstance(x, RDD):
            return self.call("predict", x.map(_convert_to_vector))

        else:
            return self.call("predict", _convert_to_vector(x))

    @since("1.1.0")
    def numNodes(self):
        """Get number of nodes in tree, including leaf nodes."""
        return self._java_model.numNodes()

    @since("1.1.0")
    def depth(self):
        """
        Get depth of tree (e.g. depth 0 means 1 leaf node, depth 1
        means 1 internal node + 2 leaf nodes).
        """
        return self._java_model.depth()

    def __repr__(self):
        """ summary of model. """
        return self._java_model.toString()

    @since("1.2.0")
    def toDebugString(self):
        """ full model. """
        return self._java_model.toDebugString()

    @classmethod
    def _java_loader_class(cls):
        return "org.apache.spark.mllib.tree.model.DecisionTreeModel"


class DecisionTree(object):
    """
    Learning algorithm for a decision tree model for classification or
    regression.

    .. versionadded:: 1.1.0
    """

    @classmethod
    def _train(cls, data, type, numClasses, features, impurity="gini", maxDepth=5, maxBins=32,
               minInstancesPerNode=1, minInfoGain=0.0):
        first = data.first()
        assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
        model = callMLlibFunc("trainDecisionTreeModel", data, type, numClasses, features,
                              impurity, maxDepth, maxBins, minInstancesPerNode, minInfoGain)
        return DecisionTreeModel(model)

    @classmethod
    @since("1.1.0")
    def trainClassifier(cls, data, numClasses, categoricalFeaturesInfo,
                        impurity="gini", maxDepth=5, maxBins=32, minInstancesPerNode=1,
                        minInfoGain=0.0):
        """
        Train a decision tree model for classification.

        :param data:
          Training data: RDD of LabeledPoint. Labels should take values
          {0, 1, ..., numClasses-1}.
        :param numClasses:
          Number of classes for classification.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param impurity:
          Criterion used for information gain calculation.
          Supported values: "gini" or "entropy".
          (default: "gini")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 5)
        :param maxBins:
          Number of bins used for finding splits at each node.
          (default: 32)
        :param minInstancesPerNode:
          Minimum number of instances required at child nodes to create
          the parent split.
          (default: 1)
        :param minInfoGain:
          Minimum info gain required to create a split.
          (default: 0.0)
        :return:
          DecisionTreeModel.

        Example usage:

        >>> from numpy import array
        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import DecisionTree
        >>>
        >>> data = [
        ...     LabeledPoint(0.0, [0.0]),
        ...     LabeledPoint(1.0, [1.0]),
        ...     LabeledPoint(1.0, [2.0]),
        ...     LabeledPoint(1.0, [3.0])
        ... ]
        >>> model = DecisionTree.trainClassifier(sc.parallelize(data), 2, {})
        >>> print(model)
        DecisionTreeModel classifier of depth 1 with 3 nodes

        >>> print(model.toDebugString())
        DecisionTreeModel classifier of depth 1 with 3 nodes
          If (feature 0 <= 0.0)
           Predict: 0.0
          Else (feature 0 > 0.0)
           Predict: 1.0
        <BLANKLINE>
        >>> model.predict(array([1.0]))
        1.0
        >>> model.predict(array([0.0]))
        0.0
        >>> rdd = sc.parallelize([[1.0], [0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "classification", numClasses, categoricalFeaturesInfo,
                          impurity, maxDepth, maxBins, minInstancesPerNode, minInfoGain)

    @classmethod
    @since("1.1.0")
    def trainRegressor(cls, data, categoricalFeaturesInfo,
                       impurity="variance", maxDepth=5, maxBins=32, minInstancesPerNode=1,
                       minInfoGain=0.0):
        """
        Train a decision tree model for regression.

        :param data:
          Training data: RDD of LabeledPoint. Labels are real numbers.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param impurity:
          Criterion used for information gain calculation.
          The only supported value for regression is "variance".
          (default: "variance")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 5)
        :param maxBins:
          Number of bins used for finding splits at each node.
          (default: 32)
        :param minInstancesPerNode:
          Minimum number of instances required at child nodes to create
          the parent split.
          (default: 1)
        :param minInfoGain:
          Minimum info gain required to create a split.
          (default: 0.0)
        :return:
          DecisionTreeModel.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import DecisionTree
        >>> from pyspark.mllib.linalg import SparseVector
        >>>
        >>> sparse_data = [
        ...     LabeledPoint(0.0, SparseVector(2, {0: 0.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 1.0})),
        ...     LabeledPoint(0.0, SparseVector(2, {0: 0.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 2.0}))
        ... ]
        >>>
        >>> model = DecisionTree.trainRegressor(sc.parallelize(sparse_data), {})
        >>> model.predict(SparseVector(2, {1: 1.0}))
        1.0
        >>> model.predict(SparseVector(2, {1: 0.0}))
        0.0
        >>> rdd = sc.parallelize([[0.0, 1.0], [0.0, 0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "regression", 0, categoricalFeaturesInfo,
                          impurity, maxDepth, maxBins, minInstancesPerNode, minInfoGain)




@inherit_doc
class CustomEnsembleModel(TreeEnsembleModel, JavaLoader):
    """
    Represents a Custom Ensemble model.

    .. versionadded:: 2.1.0
    """

    @classmethod
    def _java_loader_class(cls):
        return "org.apache.spark.mllib.tree.model.CustomEnsembleModel"    

@inherit_doc
class CustomEnsembleCrossValidation(object):
    '''
    Represets a Custom Ensemble Cross Validation model.

    .. versionadded:: 2.1.0
    '''
    @classmethod
    def __init__(cls,datagroup,crossValidationModel):
        cls.splitData = datagroup
        cls.crossValidationModel = crossValidationModel

    def getModelSize(self):
      return len(self.splitData)

    def predict(self):
      result = 0.0
      
      accuracy = self.getPredictList()

      for i in range(self.getModelSize()):
        result += accuracy[i]
      
      result /= self.getModelSize()
      return result 

    def getPredictList(self):
      assert self.splitData != None , "method for cross validation"
      assert self.crossValidationModel != None, "method for cross validation"

      split_predict = [None]*self.getModelSize()
      split_labels_and_preds = [None]*self.getModelSize()
      split_test_accuracy = [None]*self.getModelSize()

      for i in range(self.getModelSize()):
        split_predict[i] = self.crossValidationModel[i].predict(self.splitData[i].map(lambda x: x.features))
        split_labels_and_preds[i] = self.splitData[i].map(lambda p : p.label).zip(split_predict[i])
        split_test_accuracy[i] = split_labels_and_preds[i].filter(lambda (v, p): v == p).count() / float(self.splitData[i].count())
     
      return split_test_accuracy  

class CustomEnsemble(object):
    """
    Learning algorithm for a Custom Ensemble model for classification or
    regression.

    .. versionadded:: 2.1.0

    """
    supportedFeatureSubsetStrategies = ("auto", "all", "sqrt", "log2", "onethird")
    supportedCrossval = ("auto","true","false")

    @classmethod
    def _train(cls, data, algo, numClasses, categoricalFeaturesInfo, numTrees,
               featureSubsetStrategy, impurity, maxDepth, maxBins, seed, crossval,splitSize):
        
        def crossValidation():
          
          datagroup = [None]*splitSize
          weights = [1.0/splitSize]*splitSize

          split_model = [None]*splitSize
          crossValidationModel = [None]*splitSize

          datagroup = data.randomSplit(weights,seed)

          def callRDD(index):
  
            union_RDD = None
  
            for k in range(splitSize):
              if k != index:
                if union_RDD == None:
                  union_RDD = datagroup[k]
                else:
                  union_RDD = union_RDD.union(datagroup[k])
            return union_RDD

          for i in range(splitSize):
            split_model[i] = callMLlibFunc("trainCustomEnsembleModel", callRDD(i), algo, numClasses,
                              categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
                              maxDepth, maxBins, seed)
            crossValidationModel[i] = CustomEnsembleModel(split_model[i])

          return CustomEnsembleCrossValidation(datagroup,crossValidationModel)

        first = data.first()
        assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
        if featureSubsetStrategy not in cls.supportedFeatureSubsetStrategies:
            raise ValueError("unsupported featureSubsetStrategy: %s" % featureSubsetStrategy)
        if seed is None:
            seed = random.randint(0, 1 << 30)

        #check crossValidation option
        if crossval not in cls.supportedCrossval:
          raise ValueError("unsupported Crossval: %s" % crossval)
        if splitSize <= 0:
          raise ValueError("incorrect Value: %d " % splitSize)  
        if (data.count() < 300 and crossval == 'auto') or crossval == 'true':
          return crossValidation()
              


        model = callMLlibFunc("trainCustomEnsembleModel", data, algo, numClasses,
                              categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
                              maxDepth, maxBins, seed)
        return CustomEnsembleModel(model)

    @classmethod
    @since("1.2.0")
    def trainClassifier(cls, data, numClasses, categoricalFeaturesInfo, numTrees,
                        featureSubsetStrategy="auto", impurity="gini", maxDepth=4, maxBins=32,
                        seed=None,crossval = "auto",splitSize = 5):
        """
        Train a random forest model for binary or multiclass
        classification.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels should take values
          {0, 1, ..., numClasses-1}.
        :param numClasses:
          Number of classes for classification.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param numTrees:
          Number of trees in the random forest.
        :param featureSubsetStrategy:
          Number of features to consider for splits at each node.
          Supported values: "auto", "all", "sqrt", "log2", "onethird".
          If "auto" is set, this parameter is set based on numTrees:
          if numTrees == 1, set to "all";
          if numTrees > 1 (forest) set to "sqrt".
          (default: "auto")
        :param impurity:
          Criterion used for information gain calculation.
          Supported values: "gini" or "entropy".
          (default: "gini")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 4)
        :param maxBins:
          Maximum number of bins used for splitting features.
          (default: 32)
        :param seed:
          Random seed for bootstrapping and choosing feature subsets.
          Set as None to generate seed based on system time.
          (default: None)
        :param crossval:
          Criterion used for CrossValidation
          Supported values: "auto", "true", "false"
          If "auto" is set, this parameter is set based on data(RDD) scale:
          if data's scale < 300, set to "true".
          if data's scale >= 300, set to "false".
          (default: "auto")
        :param splitSize:
          Number of data group for using crossValidation
          (default: 5)
        :return:
          If "crossValidation" not used: 
          RandomForestModel that can be used for prediction.
          If "crossValidation" used: 
          List that contain RandomForestModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import RandomForest
        >>>
        >>> data = [
        ...     LabeledPoint(0.0, [0.0]),
        ...     LabeledPoint(0.0, [1.0]),
        ...     LabeledPoint(1.0, [2.0]),
        ...     LabeledPoint(1.0, [3.0])
        ... ]
        >>> model = RandomForest.trainClassifier(sc.parallelize(data), 2, {}, 3, seed=42)
        >>> model.numTrees()
        3
        >>> model.totalNumNodes()
        7
        >>> print(model)
        TreeEnsembleModel classifier with 3 trees
        <BLANKLINE>
        >>> print(model.toDebugString())
        TreeEnsembleModel classifier with 3 trees
        <BLANKLINE>
          Tree 0:
            Predict: 1.0
          Tree 1:
            If (feature 0 <= 1.0)
             Predict: 0.0
            Else (feature 0 > 1.0)
             Predict: 1.0
          Tree 2:
            If (feature 0 <= 1.0)
             Predict: 0.0
            Else (feature 0 > 1.0)
             Predict: 1.0
        <BLANKLINE>
        >>> model.predict([2.0])
        1.0
        >>> model.predict([0.0])
        0.0
        >>> rdd = sc.parallelize([[3.0], [1.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "classification", numClasses,
                          categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
                          maxDepth, maxBins, seed,crossval,splitSize)

    @classmethod
    @since("1.2.0")
    def trainRegressor(cls, data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy="auto",
                       impurity="variance", maxDepth=4, maxBins=32, seed=None, crossval = "auto",splitSize = 5):
        """
        Train a random forest model for regression.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels are real numbers.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param numTrees:
          Number of trees in the random forest.
        :param featureSubsetStrategy:
          Number of features to consider for splits at each node.
          Supported values: "auto", "all", "sqrt", "log2", "onethird".
          If "auto" is set, this parameter is set based on numTrees:
          if numTrees == 1, set to "all";
          if numTrees > 1 (forest) set to "onethird" for regression.
          (default: "auto")
        :param impurity:
          Criterion used for information gain calculation.
          The only supported value for regression is "variance".
          (default: "variance")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 4)
        :param maxBins:
          Maximum number of bins used for splitting features.
          (default: 32)
        :param seed:
          Random seed for bootstrapping and choosing feature subsets.
          Set as None to generate seed based on system time.
          (default: None)
        :return:
          RandomForestModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import RandomForest
        >>> from pyspark.mllib.linalg import SparseVector
        >>>
        >>> sparse_data = [
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 1.0})),
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 2.0}))
        ... ]
        >>>
        >>> model = RandomForest.trainRegressor(sc.parallelize(sparse_data), {}, 2, seed=42)
        >>> model.numTrees()
        2
        >>> model.totalNumNodes()
        4
        >>> model.predict(SparseVector(2, {1: 1.0}))
        1.0
        >>> model.predict(SparseVector(2, {0: 1.0}))
        0.5
        >>> rdd = sc.parallelize([[0.0, 1.0], [1.0, 0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.5]
        """
        return cls._train(data, "regression", 0, categoricalFeaturesInfo, numTrees,
                          featureSubsetStrategy, impurity, maxDepth, maxBins, seed,crossval,splitSize)


@inherit_doc
class RandomForestModel(TreeEnsembleModel, JavaLoader):
    """
    Represents a random forest model.

    .. versionadded:: 1.2.0
    """

    @classmethod
    def _java_loader_class(cls):
        return "org.apache.spark.mllib.tree.model.RandomForestModel"

class RandomForest(object):
    """
    Learning algorithm for a random forest model for classification or
    regression.

    .. versionadded:: 1.2.0
    """

    supportedFeatureSubsetStrategies = ("auto", "all", "sqrt", "log2", "onethird")

    @classmethod
    def _train(cls, data, algo, numClasses, categoricalFeaturesInfo, numTrees,
               featureSubsetStrategy, impurity, maxDepth, maxBins, seed):
        first = data.first()
        assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
        if featureSubsetStrategy not in cls.supportedFeatureSubsetStrategies:
            raise ValueError("unsupported featureSubsetStrategy: %s" % featureSubsetStrategy)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        model = callMLlibFunc("trainRandomForestModel", data, algo, numClasses,
                              categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
                              maxDepth, maxBins, seed)
        return RandomForestModel(model)

    @classmethod
    @since("1.2.0")
    def trainClassifier(cls, data, numClasses, categoricalFeaturesInfo, numTrees,
                        featureSubsetStrategy="auto", impurity="gini", maxDepth=4, maxBins=32,
                        seed=None):
        """
        Train a random forest model for binary or multiclass
        classification.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels should take values
          {0, 1, ..., numClasses-1}.
        :param numClasses:
          Number of classes for classification.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param numTrees:
          Number of trees in the random forest.
        :param featureSubsetStrategy:
          Number of features to consider for splits at each node.
          Supported values: "auto", "all", "sqrt", "log2", "onethird".
          If "auto" is set, this parameter is set based on numTrees:
          if numTrees == 1, set to "all";
          if numTrees > 1 (forest) set to "sqrt".
          (default: "auto")
        :param impurity:
          Criterion used for information gain calculation.
          Supported values: "gini" or "entropy".
          (default: "gini")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 4)
        :param maxBins:
          Maximum number of bins used for splitting features.
          (default: 32)
        :param seed:
          Random seed for bootstrapping and choosing feature subsets.
          Set as None to generate seed based on system time.
          (default: None)
        :return:
          RandomForestModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import RandomForest
        >>>
        >>> data = [
        ...     LabeledPoint(0.0, [0.0]),
        ...     LabeledPoint(0.0, [1.0]),
        ...     LabeledPoint(1.0, [2.0]),
        ...     LabeledPoint(1.0, [3.0])
        ... ]
        >>> model = RandomForest.trainClassifier(sc.parallelize(data), 2, {}, 3, seed=42)
        >>> model.numTrees()
        3
        >>> model.totalNumNodes()
        7
        >>> print(model)
        TreeEnsembleModel classifier with 3 trees
        <BLANKLINE>
        >>> print(model.toDebugString())
        TreeEnsembleModel classifier with 3 trees
        <BLANKLINE>
          Tree 0:
            Predict: 1.0
          Tree 1:
            If (feature 0 <= 1.0)
             Predict: 0.0
            Else (feature 0 > 1.0)
             Predict: 1.0
          Tree 2:
            If (feature 0 <= 1.0)
             Predict: 0.0
            Else (feature 0 > 1.0)
             Predict: 1.0
        <BLANKLINE>
        >>> model.predict([2.0])
        1.0
        >>> model.predict([0.0])
        0.0
        >>> rdd = sc.parallelize([[3.0], [1.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "classification", numClasses,
                          categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
                          maxDepth, maxBins, seed)

    @classmethod
    @since("1.2.0")
    def trainRegressor(cls, data, categoricalFeaturesInfo, numTrees, featureSubsetStrategy="auto",
                       impurity="variance", maxDepth=4, maxBins=32, seed=None):
        """
        Train a random forest model for regression.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels are real numbers.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param numTrees:
          Number of trees in the random forest.
        :param featureSubsetStrategy:
          Number of features to consider for splits at each node.
          Supported values: "auto", "all", "sqrt", "log2", "onethird".
          If "auto" is set, this parameter is set based on numTrees:
          if numTrees == 1, set to "all";
          if numTrees > 1 (forest) set to "onethird" for regression.
          (default: "auto")
        :param impurity:
          Criterion used for information gain calculation.
          The only supported value for regression is "variance".
          (default: "variance")
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 4)
        :param maxBins:
          Maximum number of bins used for splitting features.
          (default: 32)
        :param seed:
          Random seed for bootstrapping and choosing feature subsets.
          Set as None to generate seed based on system time.
          (default: None)
        :return:
          RandomForestModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import RandomForest
        >>> from pyspark.mllib.linalg import SparseVector
        >>>
        >>> sparse_data = [
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 1.0})),
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 2.0}))
        ... ]
        >>>
        >>> model = RandomForest.trainRegressor(sc.parallelize(sparse_data), {}, 2, seed=42)
        >>> model.numTrees()
        2
        >>> model.totalNumNodes()
        4
        >>> model.predict(SparseVector(2, {1: 1.0}))
        1.0
        >>> model.predict(SparseVector(2, {0: 1.0}))
        0.5
        >>> rdd = sc.parallelize([[0.0, 1.0], [1.0, 0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.5]
        """
        return cls._train(data, "regression", 0, categoricalFeaturesInfo, numTrees,
                          featureSubsetStrategy, impurity, maxDepth, maxBins, seed)


@inherit_doc
class GradientBoostedTreesModel(TreeEnsembleModel, JavaLoader):
    """
    Represents a gradient-boosted tree model.

    .. versionadded:: 1.3.0
    """

    @classmethod
    def _java_loader_class(cls):
        return "org.apache.spark.mllib.tree.model.GradientBoostedTreesModel"


class GradientBoostedTrees(object):
    """
    Learning algorithm for a gradient boosted trees model for
    classification or regression.

    .. versionadded:: 1.3.0
    """

    @classmethod
    def _train(cls, data, algo, categoricalFeaturesInfo,
               loss, numIterations, learningRate, maxDepth, maxBins):
        first = data.first()
        assert isinstance(first, LabeledPoint), "the data should be RDD of LabeledPoint"
        model = callMLlibFunc("trainGradientBoostedTreesModel", data, algo, categoricalFeaturesInfo,
                              loss, numIterations, learningRate, maxDepth, maxBins)
        return GradientBoostedTreesModel(model)

    @classmethod
    @since("1.3.0")
    def trainClassifier(cls, data, categoricalFeaturesInfo,
                        loss="logLoss", numIterations=100, learningRate=0.1, maxDepth=3,
                        maxBins=32):
        """
        Train a gradient-boosted trees model for classification.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels should take values
          {0, 1}.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param loss:
          Loss function used for minimization during gradient boosting.
          Supported values: "logLoss", "leastSquaresError",
          "leastAbsoluteError".
          (default: "logLoss")
        :param numIterations:
          Number of iterations of boosting.
          (default: 100)
        :param learningRate:
          Learning rate for shrinking the contribution of each estimator.
          The learning rate should be between in the interval (0, 1].
          (default: 0.1)
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 3)
        :param maxBins:
          Maximum number of bins used for splitting features. DecisionTree
          requires maxBins >= max categories.
          (default: 32)
        :return:
          GradientBoostedTreesModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import GradientBoostedTrees
        >>>
        >>> data = [
        ...     LabeledPoint(0.0, [0.0]),
        ...     LabeledPoint(0.0, [1.0]),
        ...     LabeledPoint(1.0, [2.0]),
        ...     LabeledPoint(1.0, [3.0])
        ... ]
        >>>
        >>> model = GradientBoostedTrees.trainClassifier(sc.parallelize(data), {}, numIterations=10)
        >>> model.numTrees()
        10
        >>> model.totalNumNodes()
        30
        >>> print(model)  # it already has newline
        TreeEnsembleModel classifier with 10 trees
        <BLANKLINE>
        >>> model.predict([2.0])
        1.0
        >>> model.predict([0.0])
        0.0
        >>> rdd = sc.parallelize([[2.0], [0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "classification", categoricalFeaturesInfo,
                          loss, numIterations, learningRate, maxDepth, maxBins)

    @classmethod
    @since("1.3.0")
    def trainRegressor(cls, data, categoricalFeaturesInfo,
                       loss="leastSquaresError", numIterations=100, learningRate=0.1, maxDepth=3,
                       maxBins=32):
        """
        Train a gradient-boosted trees model for regression.

        :param data:
          Training dataset: RDD of LabeledPoint. Labels are real numbers.
        :param categoricalFeaturesInfo:
          Map storing arity of categorical features. An entry (n -> k)
          indicates that feature n is categorical with k categories
          indexed from 0: {0, 1, ..., k-1}.
        :param loss:
          Loss function used for minimization during gradient boosting.
          Supported values: "logLoss", "leastSquaresError",
          "leastAbsoluteError".
          (default: "leastSquaresError")
        :param numIterations:
          Number of iterations of boosting.
          (default: 100)
        :param learningRate:
          Learning rate for shrinking the contribution of each estimator.
          The learning rate should be between in the interval (0, 1].
          (default: 0.1)
        :param maxDepth:
          Maximum depth of tree (e.g. depth 0 means 1 leaf node, depth 1
          means 1 internal node + 2 leaf nodes).
          (default: 3)
        :param maxBins:
          Maximum number of bins used for splitting features. DecisionTree
          requires maxBins >= max categories.
          (default: 32)
        :return:
          GradientBoostedTreesModel that can be used for prediction.

        Example usage:

        >>> from pyspark.mllib.regression import LabeledPoint
        >>> from pyspark.mllib.tree import GradientBoostedTrees
        >>> from pyspark.mllib.linalg import SparseVector
        >>>
        >>> sparse_data = [
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 1.0})),
        ...     LabeledPoint(0.0, SparseVector(2, {0: 1.0})),
        ...     LabeledPoint(1.0, SparseVector(2, {1: 2.0}))
        ... ]
        >>>
        >>> data = sc.parallelize(sparse_data)
        >>> model = GradientBoostedTrees.trainRegressor(data, {}, numIterations=10)
        >>> model.numTrees()
        10
        >>> model.totalNumNodes()
        12
        >>> model.predict(SparseVector(2, {1: 1.0}))
        1.0
        >>> model.predict(SparseVector(2, {0: 1.0}))
        0.0
        >>> rdd = sc.parallelize([[0.0, 1.0], [1.0, 0.0]])
        >>> model.predict(rdd).collect()
        [1.0, 0.0]
        """
        return cls._train(data, "regression", categoricalFeaturesInfo,
                          loss, numIterations, learningRate, maxDepth, maxBins)


def _test():
    import doctest
    globs = globals().copy()
    from pyspark.sql import SparkSession
    spark = SparkSession.builder\
        .master("local[4]")\
        .appName("mllib.tree tests")\
        .getOrCreate()
    globs['sc'] = spark.sparkContext
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    spark.stop()
    if failure_count:
        exit(-1)

if __name__ == "__main__":
    _test()
