import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Row, RowFactory, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.util.SizeEstimator

object TabmoLogisticRegression{

  def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]").setAppName("Web Intelligence")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()

    sc.setLogLevel("ERROR")
    //val data = MLUtils.loadLibSVMFile(sc, args(0))
    
    val rawData = TabmoEncoder.getVectorizedData(args(0), true)
    val datasToPredict = TabmoEncoder.getVectorizedData(args(1), false)

    //Fuse features in one column only
    val assembler = new VectorAssembler()
      .setInputCols(Array("bidfloor", "appOrSiteVec", "exchangeVec", "mediaVec", "osVec", "publisherVec", "sizeVec", "periodVec", "interestsVec"))
      .setOutputCol("features")

    val output = assembler.transform(rawData.withColumn("bidfloor", rawData("bidfloor").cast(DoubleType)))
    val datasToPredictWithFusedFeatures = assembler.transform(datasToPredict.withColumn("bidfloor", datasToPredict("bidfloor").cast(DoubleType))
      .withColumn("labelIndex", lit(1d)))


    val mllibOutput = MLUtils.convertVectorColumnsFromML(output)
    val mllibDatasToPredictWithFusedFeatures = MLUtils.convertVectorColumnsFromML(datasToPredictWithFusedFeatures)

    mllibOutput.printSchema
    mllibDatasToPredictWithFusedFeatures.printSchema

    val labelIndex = mllibOutput.columns.indexOf("labelIndex")
    val labelIndex2 = mllibDatasToPredictWithFusedFeatures.columns.indexOf("labelIndex")

    val data = mllibOutput.rdd.map(row => {
       LabeledPoint(row.getDouble(labelIndex), row.getAs("features"))
     })

    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    //Train the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    //Now predict test datas
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    //Judge performance according to the predictions made
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    val confusionMatrix = metrics.confusionMatrix
    val tp = confusionMatrix(0 ,0)
    val fp = confusionMatrix(0 ,1)
    val tn = confusionMatrix(1, 1)
    val fn = confusionMatrix(1, 0)

    val recall = tp / (tp + fn)
    val precision = tp / (tp + fp)
    println(s"Accuracy = $accuracy")
    println(s"TP = $tp")
    println(s"TN = $tn")
    println(s"FN = $fn")
    println(s"fp = $fp")
    println(s"recall = $recall")
    println(s"precision = $precision")
    println("Confusion matrix = ")
    println(s"$confusionMatrix")

    //sourceCSV.write.option("header","true").csv(args(2))


    //Now predict real datas
    val datasToPredictLP = mllibDatasToPredictWithFusedFeatures.rdd.map(row => {
      LabeledPoint(row.getDouble(labelIndex2), row.getAs("features"))
    })

    println(datasToPredictLP.count)
    val realPredictions = datasToPredictLP.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val predictionsArray = realPredictions.collect()


    // Save and load model
    //model.save(sc, "/home/maxcabourg/Documents/Polytech/tmp/scalaLogisticRegressionWithLBFGSModel")
    //val sameModel = LogisticRegressionModel.load(sc, "/home/maxcabourg/Documents/Polytech/tmp/scalaLogisticRegressionWithLBFGSModel")
  }
}