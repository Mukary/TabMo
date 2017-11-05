import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.types.TimestampType
import org.apache.spark.sql.functions._

object Process{

  /**
    * Find the bidfloor with the most occurences in the dataframe in order to fill the lines with empty bidfloors
    * @param df the dataframe containing datas
    * @return the most common bidfloor
    */
    def findMostCommonBidfloor(df: DataFrame) = {
      df.groupBy("bidfloor").count().reduce((x, y) => if (x.getLong(1).toInt > y.getLong(1).toInt)  x else y).getDouble(0)
    }

  /**
    * Cleans the datas given in input
    * @param args
    */
   def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[2]")
    val spark = SparkSession.builder.appName("Web Intelligence").config(conf).getOrCreate()

    /**
    * Convert the value of the hour in input to one the strings : "Night" (between 6pm and 5am), "Morning" (between 6am and 11am) and "Afternoon" (between 12pm and 5pm) 
    * @param arg : one value from the column timestamp (one hour)
    * @return the correct period regarding the hour in input
    */
    val coder: (Int => String) = (arg: Int) => {
	if (((18 <= arg) && (arg <= 23)) || ((0 <= arg) && (arg <= 5))) "Night" 
	else if ((6 < arg) && (arg < 11)) "Morning" 
	else "Afternoon"
    }
    val timestampToConvert = udf(coder)

    /**
    * Delete all the interests which are not IAB writed
    * @param arg : one value from the column interests (one array of interests)
    * @return the fixed array of interests
    */
    val coder2: (String => String) = (arg: String) => { 
		try { 
			if (arg.length() > 0) { 
				val interests = arg.split(","); 
				var stringResult = ""; 
				var processOnStringResult = List[String](); 
				for (interest <- interests) { 
					val interestsSplitByDash = interest.split("-"); 
					for (interestSplitByDash <- interestsSplitByDash) { 
						if (interestSplitByDash.startsWith("IAB")) { 
							if (!processOnStringResult.exists(x => x == interestSplitByDash)) { 
								processOnStringResult = interestSplitByDash::processOnStringResult; 
								stringResult += interestSplitByDash; 
								stringResult += ","; 
							} 
						} 
						else stringResult.concat("KO") 
					}; 
				}; 
				stringResult.substring(0, stringResult.length()-1); 
			} else "Didn't work" 
		} 
		catch { 
			case _: Throwable => "No interest" 
		} 
    }
    val IABToProcess = udf(coder2)
    


    import spark.implicits._
    
    val sourceJson = spark.read.json(args(0))
    val cleansedJson: DataFrame = sourceJson.drop("network")
      .drop("user")
      .drop("impid")
      .drop("city")
      .drop("type")
      .withColumn("timestamp", $"timestamp".cast("timestamp"))
      .withColumn("timestamp", hour($"timestamp"))
      .withColumn("interests", IABToProcess(col("interests")))
      .withColumn("timestamp", timestampToConvert(col("timestamp")))
      .withColumn("os", lower($"os"))
      .withColumnRenamed("timestamp", "period")

     val filledEmptyBidfloor = cleansedJson.na.fill(findMostCommonBidfloor(cleansedJson), Seq("bidfloor"))

    filledEmptyBidfloor.withColumn("size", concat(lit("["), concat_ws(",",$"size"),lit("]"))).coalesce(1).write.option("header","true").csv(args(1))
   }
}

/*
val cleansedJson = sourceJson.drop("network").drop("user").drop("impid").drop("city").drop(type).withColumn("timestamp", sourceJson("timestamp").cast(TimestampType).cast(DateType)).withColumn("os", lower($"os")).withColumnRenamed("timestamp", "period")
 */
