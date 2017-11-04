import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.sql.DataFrame

/**
  * Methods to get one hot encoders
  */
object TabmoEncoder{

  def getAppOrSiteEncoder = {
    new OneHotEncoder()
      .setInputCol("appOrSiteIndex")
      .setOutputCol("appOrSiteVec")
  }

  def getExchangeEncoder = {
    new OneHotEncoder()
      .setInputCol("exchangeIndex")
      .setOutputCol("exchangeVec")
  }

  def getMediaEncoder = {
    new OneHotEncoder()
      .setInputCol("mediaIndex")
      .setOutputCol("mediaVec")
  }

  def getOsEncoder = {
    new OneHotEncoder()
      .setInputCol("osIndex")
      .setOutputCol("osVec")
  }

  def getPublisherEncoder = {
    new OneHotEncoder()
      .setInputCol("publisherIndex")
      .setOutputCol("publisherVec")
  }

  def getSizeEncoder = {
    new OneHotEncoder()
      .setInputCol("sizeIndex")
      .setOutputCol("sizeVec")
  }

  def getPeriodEncoder = {
    new OneHotEncoder()
      .setInputCol("periodIndex")
      .setOutputCol("periodVec")
  }

  def getInterestsEncoder = {
    new OneHotEncoder()
      .setInputCol("interestsIndex")
      .setOutputCol("interestsVec")
  }

}