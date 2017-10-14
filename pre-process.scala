import org.apache.spark.sql.SparkSession

val spark = SparkSession
  .builder()
  .appName("Spark SQL basic example")
  .config("spark.some.config.option", "some-value")
  .getOrCreate()

import spark.implicits._

val dataJSON = spark.read.json("sample-data")
dataJSON.show()