//Converting JSON file to CSV with headers - by Julien Hebmann

 val data = spark.read.json("data-students.json")
 val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

 data.withColumn("size", stringify($"size")).coalesce(1).write.option("header","true").csv("./sample-data.csv") 