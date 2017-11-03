import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import TabmoEncoder._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.DoubleType

object TabmoLogisticRegression{
  def bool2double(b: Boolean) = if (b) 1f else 0f

  def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]").setAppName("Web Intelligence")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()
    sc.setLogLevel("ERROR")
    //val data = MLUtils.loadLibSVMFile(sc, args(0))
    
    val rawData = TabmoEncoder.getVectorizedData(args(0))

    val assembler = new VectorAssembler()
      .setInputCols(Array("bidfloor", "appOrSiteVec", "exchangeVec", "mediaVec", "osVec", "publisherVec", "sizeVec", "periodVec", "typeVec", "interestsVec"))
      .setOutputCol("features")

    val output = assembler.transform(rawData.withColumn("bidfloor", rawData("bidfloor").cast(DoubleType)))
    output.printSchema
    val mllibOutput = MLUtils.convertVectorColumnsFromML(output)
    val labelIndex = rawData.columns.indexOf("labelIndex")


    //TODO: fix ClassCastException: java.lang.String cannot be cast to java.lang.Boolean
    val data = mllibOutput.rdd.map(row => {
       LabeledPoint(row.getDouble(labelIndex), row.getAs("features"))
     })

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
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


    // Save and load model
    /*model.save(sc, "/home/maxcabourg/Documents/Polytech/tmp/scalaLogisticRegressionWithLBFGSModel")
    val sameModel = LogisticRegressionModel.load(sc, "target/tmp/scalaLogisticRegressionWithLBFGSModel")*/
  } 
}