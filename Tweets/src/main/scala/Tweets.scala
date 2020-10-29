import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.io._

object Tweets {


  def main(args: Array[String]): Unit = {
    //if running on windows machine - set homedir to where /bin/winutils.exe
//        System.setProperty("hadoop.home.dir", "C:\\hadoop\\")
    if (args.length != 2) {
      println("Usage: InputFile OutputDir")
    }
    // create Spark context with Spark configuration

    val spark = SparkSession.builder
      //change master to local[<no_of_cores>] for local testing
//            .master("local[4]")
      .master("yarn")
      .appName("Tweets Sentiment Analysis")
      .getOrCreate()
    import spark.implicits._

//    var iteration= args(1).toInt
//    val inputDF =spark.read.option("header","true").option("inferSchema","true").csv(args(0)).select($"ORIGIN",$"DEST")
    val inputDF =spark.read.option("header","true").option("multiLine","true").option("quote", "\"")
      .option("escape", "\"").option("inferSchema","true").csv(args(0)).select($"tweet_id",$"airline_sentiment",$"text")
    val inputDFClean=inputDF.na.drop(Array("text"))
    val Array(train, test) = inputDFClean.randomSplit(Array(0.8, 0.2),seed = 11L)
    val indexer = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")

    val tokenizer = new Tokenizer() // transformer
      .setInputCol("text")
      .setOutputCol("tokens_raw")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("tokens")

    val hashingTF = new HashingTF() //transformer
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features")

    val lr = new LogisticRegression() // algo definition
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(indexer,tokenizer,stopWordsRemover, hashingTF,lr)) //pipeline definition

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(500, 1000))
      .addGrid(lr.elasticNetParam,Array(0.0, 0.5, 1.0))
      .addGrid(lr.maxIter,Array(10, 20))
      .addGrid(lr.regParam,Array(0.01, 0.1, 1.0))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(2)
    val cvModel = cv.fit(train)

  // evaluating metrics on training data and writing out to text file passed in args
    val resultsTrain = cvModel.bestModel.transform(train).select("features", "label", "prediction")
    val predictionAndLabelsTrain = resultsTrain.select($"prediction",$"label").as[(Double, Double)].rdd
    val mMetricsTrain = new MulticlassMetrics(predictionAndLabelsTrain)
    val labelsTrain = mMetricsTrain.labels

    // evaluating metrics on test data and writing out to text file passed in args

    val resultsTest = cvModel.bestModel.transform(test).select("features", "label", "prediction")
    val predictionAndLabels = resultsTest.select($"prediction",$"label").as[(Double, Double)].rdd
    val mMetrics = new MulticlassMetrics(predictionAndLabels)

    val labels = mMetrics.labels
    val trainAccuracy="Train Accuracy:" +mMetricsTrain.accuracy
    val testAccuracy="Test Accuracy:" +mMetrics.accuracy
    val outRddOverall=spark.sparkContext.parallelize(Array(trainAccuracy,testAccuracy))
    val testMetrics=labels.map(l=>("Test: label-"+l,(("precision", mMetrics.precision(l)),("Recall", mMetrics.recall(l)),("FalsePositiveRate", mMetrics.falsePositiveRate(l)),
      ("F1-Score", mMetrics.fMeasure(l)))))
    val trainMetrics=labelsTrain.map(l=>("Train: label-"+l,(("precision", mMetricsTrain.precision(l)),("Recall", mMetricsTrain.recall(l)),("FalsePositiveRate", mMetricsTrain.falsePositiveRate(l)),
      ("F1-Score", mMetricsTrain.fMeasure(l)))))
//    val outRdd=spark.sparkContext.parallelize(Array(trainAccuracy,trainMetrics,testAccuracy,testMetrics))


    val outTrainMetrics=spark.sparkContext.parallelize(trainMetrics)
    outTrainMetrics.coalesce(1, shuffle = true).saveAsTextFile(args(1)+"trainMetrics")
    val outTestMetrics=spark.sparkContext.parallelize(testMetrics)
    outTestMetrics.coalesce(1, shuffle = true).saveAsTextFile(args(1)+"testMetrics")
    val overallMetricsRdd=spark.sparkContext.parallelize(Array(trainAccuracy,testAccuracy))
    overallMetricsRdd.coalesce(1, shuffle = true).saveAsTextFile(args(1)+"overallStats")

  }
}
