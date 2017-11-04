import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer

object TabmoIndexer{

  def getAppOrSiteIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("appOrSite")
      .setOutputCol("appOrSiteIndex")
  }

  def getExchangeIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("exchange")
      .setOutputCol("exchangeIndex")
  }

  def getLabelIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("label")
      .setOutputCol("labelIndex")
  }

  def getMediaIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("media")
      .setOutputCol("mediaIndex")
  }

  def getOsIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("os")
      .setOutputCol("osIndex")
  }

  def getSizeIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("size")
      .setOutputCol("sizeIndex")
  }

  def getPublisherIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("publisher")
      .setOutputCol("publisherIndex")
  }

  def getPeriodIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("period")
      .setOutputCol("periodIndex")
  }

  def getInterestsIndexer = {
    new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("interests")
      .setOutputCol("interestsIndex")
  }

  def main(args: Array[String]){
  // Load and parse the data file.
  // Cache the data since we will use it again to compute training error.
  val spark = SparkSession.builder.appName("Web Intelligence").getOrCreate()
  val data = spark.read.format("csv").option("header", "true").load(args(0))

  val appOrSiteIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("appOrSite")
      .setOutputCol("appOrSiteIndex")
  
  val appOrSiteIndexed = appOrSiteIndexer.fit(data).transform(data)

  val exchangeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("exchange")
      .setOutputCol("exchangeIndex")
  
  val exchangeIndexed = exchangeIndexer.fit(appOrSiteIndexed).transform(appOrSiteIndexed)


  val labelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("label")
      .setOutputCol("labelIndex")
  
  val labelIndexed = labelIndexer.fit(exchangeIndexed).transform(exchangeIndexed)

  val mediaIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("media")
      .setOutputCol("mediaIndex")
    
  val mediaIndexed = mediaIndexer.fit(labelIndexed).transform(labelIndexed)

  val osIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("os")
      .setOutputCol("osIndex")
  
  val osIndexed = osIndexer.fit(mediaIndexed).transform(mediaIndexed)

  val publisherIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("publisher")
      .setOutputCol("publisherIndex")
  
  val publisherIndexed = publisherIndexer.fit(osIndexed).transform(osIndexed)

  val sizeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("size")
      .setOutputCol("sizeIndex")
  
  val sizeIndexed = sizeIndexer.fit(publisherIndexed).transform(publisherIndexed)

  val periodIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("period")
      .setOutputCol("periodIndex")
  
  val periodIndexed = periodIndexer.fit(sizeIndexed).transform(sizeIndexed)

  val typeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("type")
      .setOutputCol("typeIndex")
  
  val typeIndexed = typeIndexer.fit(periodIndexed).transform(periodIndexed)

  val interestsIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("interests")
      .setOutputCol("interestsIndex")
  
  val interestsIndexed = interestsIndexer.fit(typeIndexed).transform(typeIndexed)

  val dataIndexed = interestsIndexed.drop("appOrSite")
      .drop("exchange")
      .drop("interests")
      .drop("label")
      .drop("media")
      .drop("os")
      .drop("publisher")
      .drop("size")
      .drop("period")
      .drop("type")
  // optional - dataIndexed.show
  dataIndexed.coalesce(1).write.option("header","true").csv(args(1))  
  }
}