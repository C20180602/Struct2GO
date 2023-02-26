# DataSampling
## 1. 算法介绍
该模块是一种常用的数据预处理方法，通常可作为其他算法的前提。它提供了从原数据集里随机抽取特定的比例或者特定数量的小样本的方法。
其他常见的算法模块可以通过配置抽样率完成数据抽样的功能，无需单独使用该模块；该模块常用于抽取小样本用于数据的可视化。<br>
说明：最终抽样的比例是min(抽样率， 抽样量/总数据量)。因此如果抽样量参数为1000，最终的抽样量不一定是精确的1000 <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，任何数据
- output: 输出，抽样后的数据，格式与输入数据一致
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- featureCols：表示需要计算的特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数

#### 算法参数
- sampleRate：样本抽样率
- takeSample：抽样数目，选填
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "DataSampling angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.DataSamplingExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 \
  sampleRate:0.8 takeSample:25 \
  
```

# FillMissingValue
## 1. 算法介绍
该模块是对特征表中的空值进行填充，填充方式有4种：  <br>
1.missingValue，按照用户自定义值进行填充  <br>
2.mean，按照均值进行填充  <br>
3.median，按照中值进行填充  <br>
4.count，按照众数进行填充  <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，任何数据
- output: 输出，填充后的数据，格式与输入数据一致
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)

#### 算法参数
- user-files：用户配置文件，定义缺失值填充方式
- fillStatPath：输出最终的缺失值，填充方式 + " " + lable col + ":" + 填充值，例如 <br>
count 0:1 <br>
median 1:0.5 <br>
missingValue 2:888 <br>

user-files户配置文件，例如
```
# 样例数据的json配置文件
 
{
    "feature": [
      {
        "id": "0",
        "fillMethod": "count"
      },
      {
        "id": "1",
        "fillMethod": "median"
       },
       {
        "id": "2-5",
        "missingValue": "888"
      }
    ]
}
```

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output
fillStatPath=hdfs://my-hdfs/fillStatPath

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "FillMissingValueExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --files ./localPath/FillMissingValueConf.txt \
  --class com.tencent.angel.spark.examples.cluster.FillMissingValueExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output fillStatPath:$fillStatPath sep:tab partitionNum:4 \
  user-files:FillMissingValueConf.txt \
  
```

# Spliter
## 1. 算法介绍
该模块是根据fraction数值将数据集分割为两部分，并将两部分分别存储  <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，任何数据
- output1: 分割后的数据1，格式与输入数据一致
- output2: 分割后的数据2，格式与输入数据一致
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)

#### 算法参数
- fraction：数据分割比例，0.0-1.0之间的小数
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output1=hdfs://my-hdfs/output1
output2=hdfs://my-hdfs/output2

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "SpliterExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.SpliterExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output1:$output1 output2:$output2 sep:tab partitionNum:4 \
  fraction:0.8 \
  
```

# Dummy
## 1. 算法介绍
Dummy模块中包含两个阶段，** 特征交叉 ** 和 ** 特征One-Hot ** 。** 特征交叉 ** 根据json配置文件，对指定的特征字段做交叉，生成feature name组成的特征； ** 特征One-Hot ** 将feature name编码成全局统一、连续的index。 <br>
算法不涉及ps相关资源

## 2. 输入数据
Dummy模块输入的数据是Table数据类型。  <br>
```
// 样例数据
1 0.5 2 1
0 0.4 3 2
1 0.5 1 1
```
说明：  <br>
数据中的 0 和 -1是两个特殊值，0表示默认值，-1 表示非法值；读取数据过程中将会过滤掉这两个特殊值；因此特征的数值表示应该避开0和-1。  <br>
支持多值特征。顾名思义，多值特征是指该特征可以包含多个value，每个value以“|”分割。例如，某特征是“喜欢的游戏”，该特征对应的值应该是多个游戏名，即游戏列表。  <br>
必须包含target字段。训练数据的target字段是0，1的label值，预测数据的target字段是每条数据的标识id。  <br>
支持标准的libsvm数据格式，第一列是label，index和value以冒号分割。不支持多值特征。  <br>
```
// libsvm数据格式样例
1 3:0.4 5:0.6 6:10
0 1:0.1 2: 10 3:0.5
```

## 3. 特征交叉
特征交叉的配置文件有两个对象“fields”和“feature_cross”，"fields"对象保存着输入数据每个字段对应的name和index；“feature_cross”对象是生成特征的配置，其中“id_features”是指生成单个特征，“comb_features”是指交叉的特征，dependencies是指交叉的特征，用逗号分隔，可以指定多个。  <br>
说明：  <br>
必须包含target字段  <br>
若某个特征是多值特征，"id_features"会生成多个特征，"comb_features"也会多次与其他的dependencies做交叉。  <br>
以下就是样例数据的配置和特征交叉阶段生成的中间结果。  <br>
```
# 样例数据的json配置文件
{
  "fields": [
    {
      "name": "target",
      "index": "0"
    },
    {
      "name": "f1",
      "index": "1"
    },
    {
      "name": "f2",
      "index": "2"
    },
    {
      "name": "f3",
      "index": "3"
    }
  ],
  "feature_cross": {
    "id_features": [
      {
        "name": "f1"
      },
      {
        "name": "f3"
      }
    ],
    "comb_features": [
      {
        "name": "f1_f2",
        "dependencies": "f1,f2"
      },
      {
        "name": "f2_f3",
        "dependencies": "f2,f3"
      }
    ]
  }
}

```

```
// 样例数据特征交叉后的中间结果
1 f1_0.5 f3_1 f1_f2_0.5_2 f2_f3_2_1
0 f1_0.4 f3_2 f1_f2_0.4_2 f2_f3_3_2
1 f1_0.5 f3_1 f1_f2_0.5_1 f2_f3_1_1
```

## 4. 特征One-Hot
特征One-Hot是基于特征交叉后的中间结果，将feature name字符串替换成全局统一、连续的feature index。   <br>
生成dummy格式的样本数据，每个样本以逗号分隔，第一个元素是target字段（训练数据的label或者预测数据的样本ID），
其他字段是指非0的feature index。   <br>
```
// one hot之后的结果
1,0,2,4,7
0,1,3,5,8
1,0,2,6,9
```

## 5. 运行
#### 算法IO参数

- input：输入，任何数据
- output: 输出，特征Dummy的输出包含三个目录，featureCount、featureIndexs、instances。输出路径，当instanceDir、indexDir、countDir不存在时，自动生成路径
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- baseFeatIndexPath：指定了特征One-Hot时，使用的feature name到index之间的映射。属于选填参数。当需要基于上次特征Dummy的结果做增量更新或者预测数据的dummy，需要指定该参数，保证两次dummy的index是一致的。
- user-files: 配置文件
- instanceDir:保存了One-Hot后的dummmy格式的样本数据
- indexDir:保存了feature name到feature index的映射关系，中间用“:”分隔；
- countDir:保存了dummy后的特征空间维度，只保存了一个Int值，

#### 算法参数
- countThreshold：某个特征在整个数据集中出现的频次小于该阈值时，将会被过滤掉。一般设置成5左右
- negSampleRate:大部分应用场景中，负样本比例太大，可通过该参数对负样本采样
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "DummyExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.DummyExample \
  --files ./localPath/featConfPath \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 user-files:featConfPath \
  negSampleRate:1 countThreshold:5 \
  
```

# Correlation
## 1. 算法介绍
该模块通过计算特征两两之间的pearson或者spearman相关系数，获得特征之间的相关性。该模块的相关性计算模式主要分为两种： <br>
一种是对相关性未知的特征（下文简称新特征）进行两两之间的相关性计算，结果为一个对角矩阵，每个对角矩阵的元素为两两特征之间的相关性系数，该值越大，说明这两个特征相关性越强；  <br>
另一种是将新特征与相关性已知的特征（下文简称旧特征）分别进行相关性计算。注意：当某列特征的方差为0时，与该特征计算的相关性值为NaN。 <br>
算法不涉及ps相关资源  <br>

输出：  <br>
1.如果没有指定新特征列，则模块不进行任何计算，也不输出任何结果  <br>
2.如果只指定了新特征列，没有指定旧特征列，则仅计算新特征两两之间的相关性，输出格式如下文例子中的（1）  <br>
3.如果既指定了新特征列，又指定了旧特征列，则模块不光会计算新特征两两之间的相关性，同时还会将新特征分别与旧特征列计算相关性，输出格式如下文例子中的（2） <br>
下面举几个例子说明上述情况：  <br>
(1) 如果只指定了新特征列:输出为新特征两两之间的相关性。例如新特征列为1,2,3，则输出格式为：  <br>

```
# 相关性系数输出样例
X 1 2 3
1 1.0 0.15 0.25
2 0.15 1.0 0.38
3 0.25 0.38 1.0
```
上面的数据显示总共有三个特征进行两两之间的相关性计算。第一行与第一列分别显示新特征的Id(X可忽略)，除此之外的其他元素分别为新特征两两之间的相关性系数。各元素之间以空格隔开。   <br>
(2) 新特征与旧特征均被指定:输出为两种相关性矩阵的合并，这两种矩阵分别是新特征两两之间的相关性矩阵和新特征与旧特征的相关性矩阵。例如新特征为1,2,3，旧特征列为4,5,6,7，则输出格式为：   <br>
```
# 相关性系数输出样例
X 1 2 3 4 5 6 7
1 1.0 0.15 0.25 0.57 0.15 0.25 0.02
2 0.15 1.0 0.38 0.15 0.11 0.38 0.49
3 0.25 0.38 1.0 0.25 0.38 0.03 0.21
```
上面的数据中第一行为新特征与旧特征的Id(X可忽略)，第一列为新特征Id，除此之外的其他元素为新特征两两之间以及新特征与旧特征之间的相关性系数。

## 2. 运行
#### 算法IO参数

- input：输入，特征输入文件
- output: 输出，相关性系数输出
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- newColStr：新特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数
- oriColStr：旧特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数

#### 算法参数
- sampleRate：样本抽样率
- method：相关性计算方式，分为pearson和spearman两种,不填表示采用pearson方式
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "CorrelationExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.CorrelationExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 \
  sampleRate:0.8 newColStr:1-5 oriColStr:7-9 method:pearson \
  
```

# MultualInformation
## 1. 算法介绍
该模块采用互信息公式计算特征之间的相关性，当值越大说明特征相关性越强。其中互信息的原理与计算公式可参考,同PearsonOrSpearman模块，MultualInformation模块的相关性计算主要分为两种：一是对新的特征进行两两之间的相关性计算，结果为一个对角矩阵，每个对角矩阵的元素为两两特征之间的相关性系数，该值越大，说明这两个特征相关性越强；二是将新特征与旧特征分别进行相关性计算。 <br>
输出：输出格式同Correlation的输出，但新特征两两之间的相关系数矩阵中对角线元素为该特征的信息熵，其他元素为特征之间的互信息。 <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，特征输入文件
- output: 输出，互信息输出
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- newColStr：新特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数
- oriColStr：旧特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数

#### 算法参数
- sampleRate：样本抽样率
- takeSample：抽样数目，选填
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "MutualInformationExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.MutualInformationExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 \
  sampleRate:0.8 newColStr:1-5 oriColStr:7-9
  
```


# Discrete
## 1. 算法介绍
Discrete算法对特征数据进行离散化处理。离散方法包括等频和等值离散方式，等频离散法按照特征值从小到大的顺序将特征值划分到对应的桶中，每个桶中容纳的元素个数是相同的；等值离散法根据特征值中的最小值和最大值确定每个桶的划分边界，从而保证每个桶的划分宽度在数值上相等。 <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，任何数据
- output: 输出，抽样后的数据，格式与输入数据一致
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- disBoundsPath:离散特征边界的存放路径，格式为：特征id+边界组成的数组(边界之间用空格隔开)


#### 算法参数
- sampleRate：样本抽样率
- partitionNum：数据分区数，spark rdd数据的分区数量
- user-files：特征配置文件名

featureConfName：特征配置文件名，从tesla页面上传配置文件。下面是一个该模块的JSON格式的特征配置文件样例：
```
{
    "feature": [
      {
        "id": "2",
        "discreteType": "equFre",
        "numBin": "3"
      },
      {
	"id": "5",
        "discreteType": "equVal",
        "numBin": "3",
	"min": "0",
	"max": "100"
       },
       {
        "id": "0",
        "discreteType": "equVal",
        "numBin": "2",
	"min":"-1"
      }
    ]
}
```
以上是对3个特征进行配置，特征配置之间的顺序没有要求，如果某些特征不需要离散配置，则不写在配置中。该配置文件中的"feature"不可更改，用户只需对如下参数进行修改即可:  <br>
特征配置参数:  <br>
"id":表示特征Id,注意该Id是输入数据中特征所在的列号，从0开始计数   <br>
"discreteType":离散化的类型，"equFre"表示等频离散，"equVal"表示等值离散   <br>
"numBin":离散化的桶数，需注意，在等频离散方法中，如果桶数设置过大，每个桶中的元素个数过少，导致离散边界中有重复的点，则出错  <br>
"min":针对等值离散的配置，限定了特征值的最小值，如果特征值中有比这个值还小的则出错。如果没有需要可不填，同下面的"max"  <br>
"max":针对等值离散的配置，限定了特征值的最大值，如果特征值中有比这个值还大的则出错，没有需要可不填

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output
disBoundsPath=hdfs://my-hdfs/disBoundsPath


source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "DiscretizeExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --files ./localPath/DiscreteJson.txt \
  --class com.tencent.angel.spark.examples.cluster.DiscretizeExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output disBoundsPath:$disBoundsPath sep:tab partitionNum:4 \
  sampleRate:1 user-files:DiscreteJson.txt \
  
```

# Information Based
## 1. 算法介绍
基于信息的特征选择，该模块共包括4种算法：信息增益（Information Gain）、基尼系数（gini）、信息增益率（Information Gain Ratio）以及对称不确定性(Symmetry Uncertainly) <br>
算法不涉及ps相关资源  <br>

输入：Table数据   <br>
输出：特征重要度矩阵，如要计算重要度的特征为1,2,3，则输出格式为：  <br>
```
# 特征重要度矩阵
X IGR GI MI SU
1 0.03 0.04 0.2 0.07
2 0.15 0.018 0.38 0.009
3 0.25 0.33 0.025 0.17
```
第一行表示特征重要度计算指标(X可忽略)，IGR表示信息增益率，GI表示基尼系数，MI表示信息增益，SU表示对称不确定性。其余各行表示特征id(即输入数据中特征所在列号，从0开始) 以及各个指标对应的结果。

## 2. 运行
#### 算法IO参数

- input：输入，带有目标标签列的Table数据
- output: 输出，抽样后的数据，格式与输入数据一致
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- featureCols：表示需要计算的特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数
- labelCol:目标标签所在列，根据目标标签在表中的位置，从0开始计数


#### 算法参数
- sampleRate：样本抽样率
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "InfoComputeExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.InfoComputeExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 \
  sampleRate:1 labelCol:0 featureCols:1-10 \
  
```

# RandomizedSVD
## 1. 算法介绍
RandomizedSVD算法是根据论文《Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions》原理以及spark平台实现的矩阵svd分解算法。 <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，labeled数据，带有行标号的数据，每行数据表示矩阵的行标号以及矩阵中该行的值
- outputS: 所有特征值的行向量表示法的输出路径，即奇异值矩阵主对角线元素组成的行向量，并以降序排列的输出路径
- outputV:右奇异向量V的保存路径，labeled数据，每行以对应的行标号开始
- outputU:左奇异向量U的保存路径，labeled数据，每行以对应的行标号开始
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- featureCols：表示需要计算的特征所在列，例如“1-10,12,15”，其说明取特征在表中的第1到第10列，第12列以及第15列，从0开始计数
- labelCol:label所在列，从0开始计数
- 

#### 算法参数
- sampleRate：样本抽样率
- K：奇异值的个数
- rCond:条件数倒数
- iterationNormalizer:迭代方式，分为两种："QR"与"none","QR"方式表示每轮迭代都需要进行QR分解，"none"方式表示每轮迭代仅进行左乘矩阵A以及A的转置，中间无"QR"分解过程
- qOverSample:采样参数q
- numIteration:迭代轮数，如果是auto，最终迭代次数为 if (k < (min(matRow, matCol) * 0.1)) 7 else 4
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
outputS=hdfs://my-hdfs/outputS
outputV=hdfs://my-hdfs/outputV
outputU=hdfs://my-hdfs/outputU

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "RandomizedSVDExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.RandomizedSVDExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input outputS:$outputS outputU:$outputU outputV:$outputV sep:tab partitionNum:4 \
  sampleRate:1 iterationNormalizer:QR numIteration:3 qOverSample:1 K:2 abelCol:0 rCond:1e-9 \
  
```

# Scaler
## 1. 算法介绍
Scaler模块集成了最大最小值归一化（MinMaxScaler）和标准化（StandardScaler）两种方式，用户可通过特征配置文件来指定某个特征的归一化方法。下面分别介绍这两种方法：  <br>
MinMaxScaler算法对特征数据进行统一的归一化处理，默认归一化后的特征数值范围在[0,1]，用户也可以指定归一化后的取值范围为[min,max]。  <br>
该归一化的计算公式为： <br>
```
((x-EMin)/(EMax-EMin))*(max-min)+min
```
其中x表示需要归一化的特征值，EMin表示该特征下的最小值，EMax表示该特征下的最大值，min与max为用户设定的归一化后的数值范围。注意，当某列特征的最大最小值相等时，该列所有数值归一化为0.5*(max - min) + min。
StandardScaler算法主要对特征进行标准化处理，原始特征数据通过该算法的转化将成为方差为1，均值为0的新的特征。计算公式为：
```
((x-Mean)/Var)
```
其中，Mean代表该特征的平均值，Var表示该特征的样本标准差。以下的特殊情况应该注意:  <br>
(1)如果Var为0，则x标准化后的结果直接为0.0  <br>
(2)不需要均值化处理：此时算法只做方差处理，即：x/Var  <br>
(3)不需要方差处理:x直接取值(x-Mean)  <br>


算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，特征表
- output: 输出，在原始输入数据格式不变的基础上，对相应的特征进行归一化处理
- standardPath:保存特征的均值和方差，可选
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)
- user-files：特征配置文件名
scaleConfPath：特征配置文件名，从tesla页面上传配置文件。下面是一个该模块的JSON格式的特征配置文件样例  <br>
```
{
    "minmax":[
      {
        "colStr": "1-2",
        "min": "0",
	"max":"1"
      },
      {
	"colStr":"5",
        "min":"-1",
        "max": "1"
       }
       ],
	"standard":[
       {
        "colStr": "3,6-7",
        "std": "true",
	"mean":"false"
       },
       {
        "colStr": "8,9",
	"std":"true",
	"mean":"true"
       }
       ]
}

```

特征配置参数:  <br>
"minmax"和"standard":分别代表了相应的归一化模块MinMaxScaler和StandardScaler  <br>
"colStr":需要做相应处理的特征id,取值根据该特征在原始表的所在列，从0开始计数。多个特征可用","隔开，还可用"-"标识特征的起始到结束列，例如"1-20"表示第1列到第20列。   <br>
"min":归一化后的最小值   <br>
"max":归一化后的最大值   <br>
"std":是否需要做方差的标准化  <br>
"mean":是否需要做均值的标准化  <br>

#### 算法参数
- sampleRate：样本抽样率
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "ScalerExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --files ./localPath/scaleConf.txt \
  --class com.tencent.angel.spark.examples.cluster.ScalerExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output sep:tab partitionNum:4 \
  sampleRate:1 user-files:scaleConf.txt \
  
```
# Reindex（重索引）
## 1. 算法介绍
对图的节点id进行重索引，生成从0开始的递增新节点id，返回重索引后的边文件和重索引前后节点的映射关系 <br>
算法不涉及ps相关资源
## 2. 运行
#### 算法IO参数

- input：输入，网络结构输入，支持hdfs/tdw路径，每行表示一条边。当输入为hdfs时，必须是“ src分隔符dst”两列的格式，当输入为tdw时，需要指定src/dst所在的列数，默认分别为0/1，若是tdw输入，则输入表既可以是数值类型src(long) | dst(long) | weight(float)又可以是string类型 src(string) | dst(string) | weight(string)
- srcIndex:src节点所在列
- dstIndex:dst节点所在列
- weightIndex:表示权重所在列
- isWeighted:输入是否是带权图
- output: 输出，重索引后的边文件，2或3列，支持tdw和HDFS; 若输出表是tdw则需用户先创建表，格式也是既可以是数值类型或者string类型，src(long) | dst(long) | weight(float) 或者 src(string) | dst(string) | weight(string)
- maps:重索引的映射字典，2列，第一列是old id，第二列是new id；支持tdw和HDFS；若是tdw输出则需用户先创建表，格式也是既可以是数值类型或者string类型，oldid | newid
- sep: 数据分隔符，支持：空格(space)，逗号(comma)，tab(\t)


#### 算法参数
- partitionNum：数据分区数，spark rdd数据的分区数量

#### 任务提交示例

```
input=hdfs://my-hdfs/data
output=hdfs://my-hdfs/output
maps=hdfs://my-hdfs/maps

source ./spark-on-angel-env.sh
$SPARK_HOME/bin/spark-submit \
  --master yarn-cluster\
  --name "ReindexExample angel" \
  --jars $SONA_SPARK_JARS  \
  --driver-memory 5g \
  --num-executors 1 \
  --executor-cores 4 \
  --executor-memory 10g \
  --class com.tencent.angel.spark.examples.cluster.ReindexExample \
  ../lib/spark-on-angel-examples-3.2.0.jar \
  input:$input output:$output maps:$maps sep:tab partitionNum:4 \
  srcIndex:0 dstIndex:1 weightIndex:2 isWeighted:false \
  
```
