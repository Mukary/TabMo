import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, NaiveBayes}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import TabmoIndexer._
import TabmoEncoder._
import Process._

object TabmoLogisticRegression{

  val string2double: (String => Double) = (b: String) => if(b == "true") 1.0d else 0.0d
  val string2doubleFunc = udf(string2double)

  def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]").setAppName("Web Intelligence")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()

    sc.setLogLevel("ERROR")

    val datasTemp = Process.getProcessedData(args(0))
    val datasTemp2 = Process.getProcessedData(args(1))
    
    //Indexers
    val appOrSiteIndexer = getAppOrSiteIndexer
    val exchangeIndexer = getExchangeIndexer
    val labelIndexer = getLabelIndexer
    val mediaIndexer = getMediaIndexer
    val osIndexer = getOsIndexer
    val publisherIndexer = getPublisherIndexer
    val sizeIndexer = getSizeIndexer
    val periodIndexer = getPeriodIndexer
    val interestsIndexer = getInterestsIndexer

    //Encoders
    val appOrSiteEncoder = getAppOrSiteEncoder
    val exchangeEncoder = getExchangeEncoder
    val mediaEncoder = getMediaEncoder
    val osEncoder = getOsEncoder
    val publisherEncoder = getPublisherEncoder
    val sizeEncoder = getSizeEncoder
    val periodEncoder = getPeriodEncoder
    val interestsEncoder = getInterestsEncoder

    //Fuse features in one column only
    val assemblerEncoder = new VectorAssembler()
      .setInputCols(Array("bidfloor", "appOrSiteVec", "exchangeVec", "mediaVec", "osVec", "publisherVec", "sizeVec", "periodVec", "interestsVec"))
      .setOutputCol("features")

    val assemblerIndexer = new VectorAssembler()
        .setInputCols(Array("bidfloor", "appOrSiteIndex", "exchangeIndex", "mediaIndex", "osIndex", "publisherIndex", "sizeIndex", "periodIndex", "interestsIndex"))
        .setOutputCol("features")

    val datas = datasTemp.withColumn("bidfloor", datasTemp("bidfloor").cast(DoubleType)).withColumn("label", string2doubleFunc(col("label")))
    val datasToPredict = datasTemp2.withColumn("bidfloor", datasTemp2("bidfloor").cast(DoubleType))//.withColumn("label", lit(1.0d))

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setLabelCol("label").setFeaturesCol("features")
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features").setMaxBins(1024)
    val nb = new NaiveBayes().setFeaturesCol("features").setSmoothing(1)
    val pipeline = new Pipeline().setStages(Array(appOrSiteIndexer, exchangeIndexer, mediaIndexer, osIndexer, publisherIndexer,
      sizeIndexer, periodIndexer, interestsIndexer,/*appOrSiteEncoder, exchangeEncoder, mediaEncoder, osEncoder, publisherEncoder, sizeEncoder, periodEncoder,
      interestsEncoder, */assemblerIndexer, nb ))

    val splits = datas.randomSplit(Array(0.8, 0.2))
    val model = pipeline.fit(splits(0))
    /*println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    val predictions = model.transform(splits(1))
    val labelAndPredictions = predictions.select "labelIndex", "prediction")
    val truep = labelAndPredictions.filter($"prediction" === 0.0).filter($"labelIndex" === $"prediction").count
    val truen = labelAndPredictions.filter($"prediction" === 1.0).filter($"labelIndex" === $"prediction").count
    val falseN = labelAndPredictions.filter($"prediction" === 0.0).filter(not($"labelIndex" === $"prediction")).count
    val falseP = labelAndPredictions.filter($"prediction" === 1.0).filter(not($"labelIndex" === $"prediction")).count

    println(truep, truen, falseN, falseP)*/

    val testedDatas = model.transform(splits(1))
    val labelAndPredictions = testedDatas.select("label", "prediction")

    import spark.implicits._
    val truep = labelAndPredictions.filter($"prediction" === 0.0).filter($"label" === $"prediction").count
    val truen = labelAndPredictions.filter($"prediction" === 1.0).filter($"label" === $"prediction").count
    val falseN = labelAndPredictions.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count
    val falseP = labelAndPredictions.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count

    val recall = truep / (truep + falseN).toDouble
    val precision = truep / (truep + falseP).toDouble

    println(s"TP = $truep, TN = $truen, FN = $falseN, FP = $falseP")
    println(s"recall = $recall")
    println(s"precision = $precision")

    val predictedDatas = model.transform(datasToPredict)

    //val datasToWrite = datasTemp2.withColumn("prediction", predictedDatas.)

  }
}