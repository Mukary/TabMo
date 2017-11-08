# TabMo - Web intelligence project

## Compile the sources

### Prerequisites

Have *Scala 2.11*, *SBT 1.0.2* and *Spark 2.2.0* installed

### Steps
* Download the project
* Go to the root of the project with a terminal
* Launch `sbt package`

You will get a generated jar in /target/scala-X.XX named simple-project_X.XX-X.X.jar

## Start the project

### Steps

1) Train data and predict raw data in one shot

Use the command spark-submit :
`spark-submit --class "TabmoLogisticRegression" --master local[2] path-to-jar/nameOfTheJar.jar arg1 arg2 arg3`
 with: 
 -  arg1 : path-to-training-data/train-data.json
 -  arg2 : path-to-data-to-predict/raw-data.csv
 -  arg3: directory name that will be created to generate predicted data file

* Files can be in JSON format or CSV format

Predicted data is a csv file located in directory-name/part-00000-642e21d2-1b48-4b22-99bb-7498b428b06c-c000.csv


