import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.DataFrame

object TabmoEncoder{

  /**
    * String indexes and one-hot encodes datas in a file
    * @param sourceFile full path of the file containing datas
    * @return dataframe with encoded datas
    */
  def getVectorizedData(sourceFile: String): DataFrame = {
  // Load and parse the data file.
  // Cache the data since we will use it again to compute training error.
  val spark = SparkSession.builder.appName("Web Intelligence").getOrCreate()
  val data = spark.read.format("csv").option("header", "true").load(sourceFile)

  //appOrSite
  val appOrSiteIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("appOrSite")
      .setOutputCol("appOrSiteIndex")
  
  val appOrSiteIndexed = appOrSiteIndexer.fit(data).transform(data)

  val appOrSiteEncoder = new OneHotEncoder()
      .setInputCol("appOrSiteIndex")
      .setOutputCol("appOrSiteVec")

  val appOrSiteEncoded = appOrSiteEncoder.transform(appOrSiteIndexed)

  //Exchange
  val exchangeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("exchange")
      .setOutputCol("exchangeIndex")
  
  val exchangeIndexed = exchangeIndexer.fit(appOrSiteEncoded).transform(appOrSiteEncoded)

  val exchangeEncoder = new OneHotEncoder()
      .setInputCol("exchangeIndex")
      .setOutputCol("exchangeVec")

  val exchangeEncoded = exchangeEncoder.transform(exchangeIndexed)

  //Label
  val labelIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("label")
      .setOutputCol("labelIndex")
  
  val labelIndexed = labelIndexer.fit(exchangeEncoded).transform(exchangeEncoded)

  //Media
  val mediaIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("media")
      .setOutputCol("mediaIndex")
    
  val mediaIndexed = mediaIndexer.fit(labelIndexed).transform(labelIndexed)

  val mediaEncoder = new OneHotEncoder()
      .setInputCol("mediaIndex")
      .setOutputCol("mediaVec")

  val mediaEncoded = mediaEncoder.transform(mediaIndexed)
  //OS
  val osIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("os")
      .setOutputCol("osIndex")
  
  val osIndexed = osIndexer.fit(mediaEncoded).transform(mediaEncoded)

  val osEncoder = new OneHotEncoder()
      .setInputCol("osIndex")
      .setOutputCol("osVec")

  val osEncoded = osEncoder.transform(osIndexed)

  //Publisher
  val publisherIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("publisher")
      .setOutputCol("publisherIndex")
  
  val publisherIndexed = publisherIndexer.fit(osEncoded).transform(osEncoded)

  val publisherEncoder = new OneHotEncoder()
      .setInputCol("publisherIndex")
      .setOutputCol("publisherVec")

  val publisherEncoded = publisherEncoder.transform(publisherIndexed)

  //Size
  val sizeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("size")
      .setOutputCol("sizeIndex")
  
  val sizeIndexed = sizeIndexer.fit(publisherEncoded).transform(publisherEncoded)
  
  val sizeEncoder = new OneHotEncoder()
      .setInputCol("sizeIndex")
      .setOutputCol("sizeVec")

  val sizeEncoded = sizeEncoder.transform(sizeIndexed)

  //Period
  val periodIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("period")
      .setOutputCol("periodIndex")
  
  val periodIndexed = periodIndexer.fit(sizeEncoded).transform(sizeEncoded)

  val periodEncoder = new OneHotEncoder()
      .setInputCol("periodIndex")
      .setOutputCol("periodVec")

  val periodEncoded = periodEncoder.transform(periodIndexed)

  //Type
  val typeIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("type")
      .setOutputCol("typeIndex")
  
  val typeIndexed = typeIndexer.fit(periodEncoded).transform(periodEncoded)

  val typeEncoder = new OneHotEncoder()
      .setInputCol("typeIndex")
      .setOutputCol("typeVec")

  val typeEncoded = typeEncoder.transform(typeIndexed)

  //Interests
  val interestsIndexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("interests")
      .setOutputCol("interestsIndex")
  
  val interestsIndexed = interestsIndexer.fit(typeEncoded).transform(typeEncoded)

  val interestsEncoder = new OneHotEncoder()
      .setInputCol("interestsIndex")
      .setOutputCol("interestsVec")

  val interestsEncoded = interestsEncoder.transform(interestsIndexed)

  //Drop useless columns
  val dataIndexed = interestsEncoded.drop("appOrSite")
      .drop("exchange")
      .drop("interests")
      .drop("label")
      .drop("media")
      .drop("os")
      .drop("publisher")
      .drop("size")
      .drop("period")
      .drop("type")
      .drop("appOrSiteIndex")
      .drop("exchangeIndex")
      .drop("interestsIndex")
      //.drop("labelIndex")
      .drop("mediaIndex")
      .drop("osIndex")
      .drop("publisherIndex")
      .drop("sizeIndex")
      .drop("periodIndex")
      .drop("typeIndex")  

  //dataIndexed.show
  //dataIndexed.coalesce(1).write.option("header","true").csv(args(1))
  dataIndexed  
  }
}