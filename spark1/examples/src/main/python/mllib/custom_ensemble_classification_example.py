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

"""
Random Forest Classification Example.
"""
from __future__ import print_function

from pyspark import SparkContext
# $example on$
from pyspark.mllib.tree import CustomEnsemble, CustomEnsembleModel
from pyspark.mllib.util import MLUtils
# $example off$

if __name__ == "__main__":
    sc = SparkContext(appName="PythonCustomEnsembleClassificationExample")
    # $example on$
    #read data
    train_data_rdd = sc.textFile("data/mllib/adult.data")

    train_data_rdd = train_data_rdd.filter(lambda x: len(x) > 80)

    #drop data
    dweight = [.01,.99]
    dseed = 21
    train_data_rdd,dropData = train_data_rdd.randomSplit(dweight,dseed)

    # Split the data into training and test sets (30% held out for testing)
    #(trainingData, testData) = data.randomSplit([0.7, 0.3])

    #define parsing Data to labeledPoint
    from pyspark.mllib.regression import LabeledPoint
    def parsePoint(line):
      values = [x.encode('utf-8').strip() for x in line.split(',')]
      label = 0.0
      feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
      if values[1]=='Private': 
        feature[0] = 1
      elif values[1]=='Self-emp-not-inc':
        feature[0] = 2
      elif values[1]=='Self-emp-inc':
        feature[0] = 3
      elif values[1]=='Federal-gov':
        feature[0] = 4
      elif values[1]=='Local-gov':
        feature[0] = 5
      elif values[1]=='State-gov':
        feature[0] = 6
      elif values[1]=='Without-pay':
        feature[0] = 7
      elif values[1]=='Never-worked':
        feature[0] = 8
      else:
        feature[0] = 0
        
      if values[3]=='Bachelors':
        feature[1] = 1
      elif values[3]=='Some-college':
        feature[1] = 2
      elif values[3]=='11th':
        feature[1] = 3
      elif values[3]=='HS-grad':
        feature[1] = 4
      elif values[3]=='Prof-school':
        feature[1] = 5
      elif values[3]=='Assoc-acdm':
        feature[1] = 6
      elif values[3]=='Assoc-voc':
        feature[1] = 7
      elif values[3]=='9th':
        feature[1] = 8
      elif values[3]=='7th-8th':
        feature[1] = 9
      elif values[3]=='12th':
        feature[1] = 10
      elif values[3]=='Masters':
        feature[1] = 11
      elif values[3]=='1st-4th':
        feature[1] = 12
      elif values[3]=='10th':
        feature[1] = 13
      elif values[3]=='Doctorate':
        feature[1] = 14
      elif values[3]=='5th-6th':
        feature[1] = 15
      elif values[3]=='Preschool':
        feature[1] = 16
      else:
        feature[1] = 0
        
      if values[5]=='Married-civ-spouse':
        feature[2] = 1
      elif values[5]=='Divorced':
        feature[2] = 2
      elif values[5]=='Never-married':
        feature[2] = 3
      elif values[5]=='Separated':
        feature[2] = 4
      elif values[5]=='Widowed':
        feature[2] = 5
      elif values[5]=='Married-spouse-absent':
        feature[2] = 6
      elif values[5]=='Married-AF-spouse':
        feature[2] = 7
      else:
        feature[2] = 0
        
      if values[6]=='Tech-support':
        feature[3] = 1
      elif values[6]=='Craft-repair':
        feature[3] = 2
      elif values[6]=='Other-service':
        feature[3] = 3
      elif values[6]=='Sales':
        feature[3] = 4
      elif values[6]=='Exec-managerial':
        feature[3] = 5
      elif values[6]=='Prof-specialty':
        feature[3] = 6
      elif values[6]=='Handlers-cleaners':
        feature[3] = 7
      elif values[6]=='Machine-op-inspct':
        feature[3] = 8
      elif values[6]=='Adm-clerical':
        feature[3] = 9
      elif values[6]=='Farming-fishing':
        feature[3] = 10
      elif values[6]=='Transport-moving':
        feature[3] = 11
      elif values[6]=='Priv-house-serv':
        feature[3] = 12
      elif values[6]=='Protective-serv':
        feature[3] = 13
      elif values[6]=='Armed-Forces':
        feature[3] = 14
      else:
        feature[3] = 0
        
      if values[7]=='Wife':
        feature[4] = 1
      elif values[7]=='Own-child':
        feature[4] = 2
      elif values[7]=='Husband':
        feature[4] = 3
      elif values[7]=='Not-in-family':
        feature[4] = 4
      elif values[7]=='Other-relative':
        feature[4] = 5
      elif values[7]=='Unmarried':
        feature[4] = 6
      else:
        feature[4] = 0
        
      if values[8]=='White':
        feature[5] = 1
      elif values[8]=='Asian-Pac-Islander':
        feature[5] = 2
      elif values[8]=='Amer-Indian-Eskimo':
        feature[5] = 3
      elif values[8]=='Other':
        feature[5] = 4
      elif values[8]=='Black':
        feature[5] = 5
      else:
        feature[5] = 0
        
      if values[9]=='Female':
        feature[6] = 1
      elif values[9]=='Male':
        feature[6] = 2
      else:
        feature[6] = 0
          
      if values[13]=='United-States':
        feature[7] = 1
      elif values[13]=='Cambodia':
        feature[7] = 2
      elif values[13]=='Puerto-Rico':
        feature[7] = 3
      elif values[13]=='Canada':
        feature[7] = 4
      elif values[13]=='Germany':
        feature[7] = 5
      elif values[13]=='Outlying-US(Guam-USVI-etc)':
        feature[7] = 6
      elif values[13]=='India':
        feature[7] = 7
      elif values[13]=='Japan':
        feature[7] = 8
      elif values[13]=='Greece':
        feature[7] = 9
      elif values[13]=='South':
        feature[7] = 10
      elif values[13]=='China':
        feature[7] = 11
      elif values[13]=='Cuba':
        feature[7] = 12
      elif values[13]=='Iran':
        feature[7] = 13
      elif values[13]=='Honduras':
        feature[7] = 14
      elif values[13]=='Philippines':
        feature[7] = 15
      elif values[13]=='Poland':
        feature[7] = 17
      elif values[13]=='Jamaica':
        feature[7] = 18
      elif values[13]=='Vietnam':
        feature[7] = 19
      elif values[13]=='Mexico':
        feature[7] = 20
      elif values[13]=='Portugal':
        feature[7] = 21
      elif values[13]=='Ireland':
        feature[7] = 22
      elif values[13]=='France':
        feature[7] = 23
      elif values[13]=='Dominican-Republic':
        feature[7] = 24
      elif values[13]=='Laos':
        feature[7] = 25
      elif values[13]=='Ecuador':
        feature[7] = 26
      elif values[13]=='Taiwan':
        feature[7] = 27
      elif values[13]=='Haiti':
        feature[7] = 28
      elif values[13]=='Columbia':
        feature[7] = 29
      elif values[13]=='Hungary':
        feature[7] = 30
      elif values[13]=='Guatemala':
        feature[7] = 31
      elif values[13]=='Nicaragua':
        feature[7] = 32
      elif values[13]=='Scotland':
        feature[7] = 33
      elif values[13]=='Thailand':
        feature[7] = 34
      elif values[13]=='Yugoslavia':
        feature[7] = 35
      elif values[13]=='El-Salvador':
        feature[7] = 36
      elif values[13]=='Trinadad&Tobago':
        feature[7] = 37
      elif values[13]=='Peru':
        feature[7] = 38
      elif values[13]=='Hong':
        feature[7] = 39
      elif values[13]=='Holand-Netherlands':
        feature[7] = 40
      elif values[13]=='Italy':
        feature[7] = 41
      else:
        feature[7] = 0
                
      feature[8] = values[0]
      feature[9] = values[2]
      feature[10] = values[4]
      feature[11] = values[10]
      feature[12] = values[11]
      feature[13] = values[12]
      
      if (values[14]=='>50K')or(values[14]=='>50K.'):
        label = 1.0
      elif (values[14]=='<=50K')or(values[14]=='<=50K.'):
        label = 2.0
           
      return LabeledPoint(label,feature)

    # parsing data
    parsedData = train_data_rdd.map(parsePoint)  

    # Train a RandomForest model.
    categoricalFeaturesInfo = {0:8,0:16,0:7,0:14,0:6,0:5,0:2,0:41}

    model = CustomEnsemble.trainClassifier(parsedData,14, categoricalFeaturesInfo, 10, seed=20, maxBins  = 64,crossval = "true")

    # Evaluate model on test instances and compute test error
    #predictions = model.predict(testData.map(lambda x: x.features))
    #labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    #testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    #print('Test Error = ' + str(testErr))
    
    print('Learned classification result:')
    print(model.getPredictList())
    print(model.predict())

    # Save and load model
    #model.save(sc, "target/tmp/myRandomForestClassificationModel")
    #sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
    # $example off$
