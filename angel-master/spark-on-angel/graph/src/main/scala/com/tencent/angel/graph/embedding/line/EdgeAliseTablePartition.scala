/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package com.tencent.angel.graph.embedding.line

import java.io.{DataInputStream, DataOutputStream}
import java.util.Random

import com.tencent.angel.PartitionKey
import com.tencent.angel.common.StreamSerdeUtils
import com.tencent.angel.ml.matrix.RowType
import com.tencent.angel.ps.server.data.request.UpdateOp
import com.tencent.angel.ps.storage.partition.UserDefinePartition
import com.tencent.angel.ps.storage.partition.storage.UserDefinePartitionStorage
import io.netty.buffer.ByteBuf
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap
import org.apache.commons.logging.LogFactory

/**
  * A user-define ps partition class for store alias table on PS
  *
  * @param partKey     partition key
  * @param rowType     row type
  * @param estSparsity estimate sparsity
  * @param storage     partition storage
  */
class EdgeAliasTablePartition(partKey: PartitionKey, rowType: RowType, estSparsity: Long,
                              storage: EdgeAliasTableStorage)
  extends UserDefinePartition(partKey, rowType, -1, storage) with IEdgeAliasTableOp {

  def this() = this(null, RowType.T_ANY_LONGKEY_SPARSE, 0, null)

  override def init(): Unit = {}

  override def reset(): Unit = getStorage.reset()

  override def getStorage: EdgeAliasTableStorage = super.getStorage.asInstanceOf[EdgeAliasTableStorage]

  override def batchSample(number: Int): (Array[Int], Array[Int]) = getStorage.batchSample(number)

  override def weightSum(): Double = getStorage.weightSum()
}


object EdgeAliasTablePartition {
  def main(args: Array[String]): Unit = {
    val edgeNum = 4
    val srcNodes = new Array[Int](edgeNum)
    val dstNodes = new Array[Int](edgeNum)
    var weights = new Array[Float](edgeNum)

    srcNodes(0) = 1
    dstNodes(0) = 2
    srcNodes(1) = 2
    dstNodes(1) = 3
    srcNodes(2) = 3
    dstNodes(2) = 4
    srcNodes(3) = 4
    dstNodes(3) = 1

    weights(0) = 1
    weights(1) = 2
    weights(2) = 3
    weights(3) = 4

    val storeage = new EdgeAliasTableStorage(0, srcNodes, dstNodes, weights)
    storeage.init()
    storeage.buildAliasTable

    val samples = storeage.batchSample(100000000)
    val srcs = samples._1
    val map = new Int2IntOpenHashMap(4)
    srcs.foreach(f => map.addTo(f, 1))

    val iter = map.int2IntEntrySet().fastIterator()
    while (iter.hasNext) {
      val entry = iter.next()
      println(s"key = ${entry.getIntKey}, value = ${entry.getIntValue}")
    }
  }
}

object EdgeAliasTableStorage {
  val LOG = LogFactory.getLog(classOf[EdgeAliasTableStorage])
}

/**
  * A user-define partition storage for store alias table
  *
  * @param rowOffset row offset
  * @param srcNodes  src nodes
  * @param dstNodes  dst nodes
  * @param weights   edge weights
  */
class EdgeAliasTableStorage(@transient var rowOffset: Int, @transient var srcNodes: Array[Int],
                            @transient var dstNodes: Array[Int], var weights: Array[Float])
  extends UserDefinePartitionStorage(rowOffset) with IEdgeAliasTableOp {

  def this() = this(-1, null, null, null)

  @volatile var prob: Array[Float] = null
  @volatile var alias: Array[Int] = null
  val rand = new Random(System.currentTimeMillis())
  @volatile var sum: Double = 0.0

  override def init(): Unit = {}

  def buildAliasTable: Unit = {
    weights.foreach(e => sum += e)
    val aliasTable = AliasTableUtils.buildAliasTable(weights)
    prob = aliasTable._1
    alias = aliasTable._2

    weights = null
  }

  override def reset(): Unit = {
    prob = null
    alias = null
  }

  override def update(buf: ByteBuf, op: UpdateOp): Unit = {
    throw new UnsupportedOperationException("")
  }

  override def getElemNum: Long = {
    if (prob != null) prob.length else 0
  }

  override def batchSample(number: Int): (Array[Int], Array[Int]) = {
    val srcSampleEdges = new Array[Int](number)
    val dstSampleEdges = new Array[Int](number)
    for (i <- (0 until number)) {
      val id = rand.nextInt(prob.length)
      val v = rand.nextDouble().toFloat
      if (v < prob(id)) {
        srcSampleEdges(i) = srcNodes(id)
        dstSampleEdges(i) = dstNodes(id)
      } else {
        srcSampleEdges(i) = srcNodes(alias(id))
        dstSampleEdges(i) = dstNodes(alias(id))
      }
    }

    (srcSampleEdges, dstSampleEdges)
  }

  override def weightSum(): Double = sum

  /**
    * Serialize object to the Output stream.
    *
    * @param output the Netty ByteBuf
    */
  override def serialize(output: DataOutputStream): Unit = {
    super.serialize(output)
    if (srcNodes == null || dstNodes == null || prob == null || alias == null) {
      StreamSerdeUtils.serializeBoolean(output, false)
      EdgeAliasTableStorage.LOG.warn("Alias table is not valid, write snapshot failed ")
    } else {
      StreamSerdeUtils.serializeBoolean(output, true)
      StreamSerdeUtils.serializeDouble(output, sum)

      StreamSerdeUtils.serializeInts(output, srcNodes)
      StreamSerdeUtils.serializeInts(output, dstNodes)
      StreamSerdeUtils.serializeFloats(output, prob)
      StreamSerdeUtils.serializeInts(output, alias)
    }
  }

  /**
    * Deserialize object from the input stream.
    *
    * @param input the input stream
    */
  override def deserialize(input: DataInputStream): Unit = {
    super.deserialize(input)
    if (!StreamSerdeUtils.deserializeBoolean(input)) {
      throw new RuntimeException("Recover alias table failed!!!")
    } else {
      sum = StreamSerdeUtils.deserializeDouble(input)

      srcNodes = StreamSerdeUtils.deserializeInts(input)
      dstNodes = StreamSerdeUtils.deserializeInts(input)
      prob = StreamSerdeUtils.deserializeFloats(input)
      alias = StreamSerdeUtils.deserializeInts(input)
    }
  }
}

/**
  * Alias table operator
  */
trait IEdgeAliasTableOp {
  /**
    * Get a batch of samples from alias table
    *
    * @param number sample number
    * @return samples
    */
  def batchSample(number: Int): (Array[Int], Array[Int])

  /**
    * Get the sum of all edges weights in this partition
    *
    * @return the sum of all edges weights in this partition
    */
  def weightSum(): Double
}