import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.sql.functions._

object Process{

    def findMostCommonBidfloor(df: DataFrame) = {
      df.groupBy("bidfloor").count().reduce((x, y) => if (x.getLong(1).toInt > y.getLong(1).toInt)  x else y).getDouble(0)
    }
   def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]")
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()

    import spark.implicits._
    
    val sourceJson = spark.read.json(args(0))
    val cleansedJson: DataFrame = sourceJson.drop("network")
      .drop("user")
      .drop("impid")
      .drop("city")
      .withColumn("timestamp", sourceJson("timestamp").cast(TimestampType).cast(DateType))
      .withColumn("os", lower($"os"))
      .withColumnRenamed("timestamp", "period")

     val filledEmptyBidfloor = cleansedJson.na.fill(findMostCommonBidfloor(cleansedJson), Seq("bidfloor"))

    filledEmptyBidfloor.withColumn("size", concat(lit("["), concat_ws(",",$"size"),lit("]"))).coalesce(1).write.option("header","true").csv(args(1))
   }
}

/*
val cleansedJson = sourceJson.drop("network").drop("user").drop("impid").drop("city").withColumn("timestamp", sourceJson("timestamp").cast(TimestampType).cast(DateType)).withColumn("os", lower($"os")).withColumnRenamed("timestamp", "period")
 */
