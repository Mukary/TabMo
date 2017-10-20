//Converting JSON file to CSV with headers - by Julien Hebmann

 val data = spark.read.json("data-students.json")
 data.withColumn("size", concat(lit("["), concat_ws(",",$"size"),lit("]"))).coalesce(1).write.option("header","true").csv("data-students")
