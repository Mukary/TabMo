# TabMo - Web intelligence project

## Compile the sources

### Prerequisites

Have *Scala*, *SBT* and *Spark* installed

### Steps
* Download the project
* Go to the root of the project with a terminal
* Launch `sbt package`

You will get a generated jar in /target/scala-X.XX named simple-project_X.XX-X.X.jar

## Start the project

### Steps

1°) Pre-process both training datas and real datas
Use the command spark-submit :
`spark-submit --class "Process" --master local[2] target/scala-2.XX/nameOfTheJar.jar arg1 arg2`
 with 
 -  arg1 : /ABSOLUTE-PATH/TO/TRAINING-JSON-DATAS
 -  arg2 : /ABSOLUTE-PATH/TO/DIRECTORY/STORING/CSV-DATAS
 and then a second time with the real json datas
 That will clean both the training and real datas to match our algorithm
 
2°) // TODO tell the command to launch the project (when project is done)


