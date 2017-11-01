import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.sql.functions._

object Process{
   def main(args: Array[String]){
    val spark = SparkSession.builder.appName("Web Intelligence").getOrCreate()
    import spark.implicits._
    
    val sourceJson = spark.read.json("/home/data/data-students.json")
    val cleansedJson = sourceJson.drop("network").drop("impid").withColumn("timestamp", sourceJson("timestamp").cast(TimestampType).cast(DateType)).withColumn("os", lower($"os"))
    
    cleansedJson.withColumn("size", concat(lit("["), concat_ws(",",$"size"),lit("]"))).coalesce(1).write.option("header","true").csv("data-csv")
   }
}