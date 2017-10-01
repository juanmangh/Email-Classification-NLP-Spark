// spamFilter.scala

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import java.lang.Math
import scala.collection.immutable.Map

object spamFilter {
	def probaWordDir (sc:SparkContext, filesDir:String)
	:(RDD[(String, Double)], Long) = {

	val files = sc.wholeTextFiles(filesDir)
	val nbFiles = sc.wholeTextFiles(filesDir).count
	val uwpf = files.flatMap(f=>f._2.split("\\s+").distinct)

	var wordDirOccurency = uwpf.map(f=>(f,1)).reduceByKey(_+_)
	val filt = Array(".",":",","," ","/","\\","-","'","(",")","@")
	val ma = filt.map(f=>(f,0))
	val nuevo = sc.parallelize(ma)

	wordDirOccurency = wordDirOccurency.subtractByKey(nuevo)

	val probaWord = wordDirOccurency.mapValues(a=>a/nbFiles.toDouble)

	return	(probaWord, nbFiles)
	}
	def computeMutualInformationFactor(
		probaWC:RDD[(String, Double)],
		probaW:RDD[(String, Double)],
		probaC: Double,
		probaDefault: Double // default value when a probability is missing
	):RDD[(String, Double)] = {
	
	val probaWordMerge = probaWC.fullOuterJoin(probaW)
	val probaWordMerged = probaWordMerge.mapValues(f=>f._1.getOrElse(probaDefault))

	val down = probaW.mapValues(f=>f*probaC)
	val down2 = down.collect.toArray.toMap
	
	val inside = probaWordMerged.map(f=>(f._1,(f._2/down2(f._1))))

	val outside = inside.mapValues(a=>(java.lang.Math.log(a))/(java.lang.Math.log(2)))
	val outside2 = outside.collect.toArray.toMap

	val fin = probaWordMerged.map(f=>(f._1,f._2*outside2(f._1)))

	return fin
	}

	def main(args: Array[String]) {

	val conf = new SparkConf().setAppName("Spam Filter")
	val sc = new SparkContext(conf)
	
	val (probaWordHam, nbFilesHam) = probaWordDir(sc,"/tmp/ling-spam/ham/*.txt")
	val (probaWordSpam, nbFilesSpam) = probaWordDir(sc,"/tmp/ling-spam/spam/*.txt")
	val totalFiles = nbFilesHam + nbFilesSpam

	val probaDefault = 0.2/totalFiles.toDouble

	val probaWordHamFalse = probaWordHam.mapValues(f=>1.0-f)
	val probaWordSpamFalse = probaWordSpam.mapValues(f=>1.0-f)

	val wordHam = probaWordHam.mapValues(f=>f*nbFilesHam.toDouble)
	val wordSpam = probaWordSpam.mapValues(f=>f*nbFilesSpam.toDouble)
	val words = wordHam.union(wordSpam)
	val words2 = words.reduceByKey(_+_)

	val probaW = words2.mapValues(a=>a/totalFiles.toDouble)
	val probaWFalse = probaW.mapValues(a=>1.0-a)

	val probaCHam = nbFilesHam/totalFiles.toDouble
	val probaCSpam = nbFilesSpam/totalFiles.toDouble

	val (trueHam) = computeMutualInformationFactor(probaWordHam,probaW,probaCHam,probaDefault)
	val (falseHam) = computeMutualInformationFactor(probaWordHamFalse,probaWFalse,probaCHam,probaDefault)
	val (trueSpam) = computeMutualInformationFactor(probaWordSpam,probaW,probaCSpam,probaDefault)
	val (falseSpam) = computeMutualInformationFactor(probaWordSpamFalse,probaWFalse,probaCSpam,probaDefault)

	val fin = trueHam.union(falseHam).union(trueSpam).union(falseSpam)
	val finfin = fin.reduceByKey(_+_)
	val fin2 = finfin.takeOrdered(10)(Ordering[Double].reverse.on(x=>x._2))
	fin2.take(10).foreach(println)
	val fin3 = sc.parallelize(fin2)
	fin3.coalesce(1,true).saveAsTextFile("/tmp/topWords.txt")
	}
}
