import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, NaiveBayes}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{ChiSqSelector, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import TabmoIndexer._
import TabmoEncoder._
import Process._

object TabmoLogisticRegression{

  def balanceDataset(dataset: DataFrame): DataFrame = {

    // Re-balancing (weighting) of records to be used in the logistic loss objective function
    val numNegatives = dataset.filter(dataset("label") === 0.0d).count
    val datasetSize = dataset.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 0.0) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }

    dataset.withColumn("classWeightCol", calculateWeights(dataset("label")))
  }

  val string2double: (String => Double) = (b: String) => if(b == "true") 1.0d else 0.0d
  val string2doubleFunc = udf(string2double)

  def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]").setAppName("Web Intelligence")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()

    sc.setLogLevel("ERROR")

    val datasTemp = Process.getProcessedData(args(0)).na.fill("NA")
    val datasTemp2 = Process.getProcessedData(args(1)).na.fill("NA")
    
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

    val chisqSelector = new ChiSqSelector().setFpr(0.05).setFeaturesCol("features").setLabelCol("label").setOutputCol("selectedFeatures")

    val datas = balanceDataset(datasTemp.withColumn("bidfloor", datasTemp("bidfloor").cast(DoubleType)).withColumn("label", string2doubleFunc(col("label"))))
    val datasToPredict = datasTemp2.withColumn("bidfloor", datasTemp2("bidfloor").cast(DoubleType))//.withColumn("label", lit(1.0d))

    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("selectedFeatures").setWeightCol("classWeightCol")
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("selectedFeatures").setMaxBins(1024)
    val nb = new NaiveBayes().setFeaturesCol("selectedFeatures").setSmoothing(1)
    val pipeline = new Pipeline().setStages(Array(appOrSiteIndexer, exchangeIndexer, mediaIndexer, osIndexer, publisherIndexer,
      sizeIndexer, periodIndexer, interestsIndexer, appOrSiteEncoder, exchangeEncoder, mediaEncoder, osEncoder, publisherEncoder, sizeEncoder, periodEncoder,
      interestsEncoder, assemblerIndexer, chisqSelector, lr ))

    val splits = datas.randomSplit(Array(0.9, 0.1))
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
    println(s" negatives = ${labelAndPredictions.filter(labelAndPredictions("label") === 0.0d).count}")

    import spark.implicits._
    val truep = labelAndPredictions.filter($"prediction" === 1.0).filter($"label" === $"prediction").count
    val truen = labelAndPredictions.filter($"prediction" === 0.0).filter($"label" === $"prediction").count
    val falseN = labelAndPredictions.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count
    val falseP = labelAndPredictions.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count

    val recall = truep / (truep + falseN).toDouble
    val precision = truep / (truep + falseP).toDouble

    println(s"TP = $truep, TN = $truen, FN = $falseN, FP = $falseP")
    println(s"recall = $recall")
    println(s"precision = $precision")

    val predictedDatas = model.transform(datasToPredict)

    predictedDatas.select($"appOrSite", $"bidfloor", $"exchange", $"interests", $"media", $"os", $"publisher", $"size", $"period", $"prediction").coalesce(1).write.option("header","true").csv(args(2))

    //val datasToWrite = datasTemp2.withColumn("prediction", predictedDatas.)

  }
}