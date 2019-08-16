package org.ssketch;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import breeze.linalg.PCA;
import org.apache.hadoop.io.IntWritable;
import org.apache.log4j.Level;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.QRDecomposition;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.storage.StorageLevel;
import org.codehaus.janino.Java;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

public class SSketchPCA implements Serializable{
    private final static Logger log = LoggerFactory.getLogger(SSketchPCA.class);// getLogger(SparkPCA.class);


    static long startTime, endTime, totalTime;
    public static int nClusters = 4;

    public int lifeCount;
    public double lifeTime=0;
    public double lifeError=0;
    //public static Stat stat = new Stat();
    public static ArrayList<Stat> stat_list=new ArrayList<Stat>();


    public static void main(String[] args) {
        org.apache.log4j.Logger.getLogger("org").setLevel(Level.ERROR);
        org.apache.log4j.Logger.getLogger("akka").setLevel(Level.ERROR);



        final int nRows;
        final int nCols;
        final int nPCs;
        final int q;// default
        final double k_plus_one_singular_value;
        final double tolerance;
        final int subsample;
        final int subsampleNorm;
        int maxIterations=1;
        int calculateError=0;
        final int blockSize;
        final int stride;
        final String inputPath;

        final String outputPath;



        /*
        final int nRows=6237\\;
        final int nCols=617;
        final int nPCs=10;
        final int q=25;// default
        final double k_plus_one_singular_value=5;
        final double tolerance=0.05;
        final int subsample=8;
        final int subsampleNorm=8;
        int maxIterations=10;
        int calculateError=0;
        final int blockSize=1000;
        final int stride=1;
        final String inputPath="isolet_final_617.csv";

        String cwd = System.getProperty("user.dir");
        final String outputPath=cwd;
         */




        try {
            inputPath = System.getProperty("i");
            if (inputPath == null)
                throw new IllegalArgumentException();
        } catch (Exception e) {
            printLogMessage("i");
            return;
        }
        try {
            outputPath = System.getProperty("o");
            if (outputPath == null)
                throw new IllegalArgumentException();
        } catch (Exception e) {
            printLogMessage("o");
            return;
        }


        try {
            nRows = Integer.parseInt(System.getProperty("rows"));
        } catch (Exception e) {
            printLogMessage("rows");
            return;
        }

        try {
            nCols = Integer.parseInt(System.getProperty("cols"));
        } catch (Exception e) {
            printLogMessage("cols");
            return;
        }

        try {
            k_plus_one_singular_value = Double.parseDouble(System.getProperty("SingularValue"));
        } catch (Exception e) {
            printLogMessage("SingularValue");
            return;
        }

        try {
            tolerance = Double.parseDouble(System.getProperty("tolerance"));
        } catch (Exception e) {
            printLogMessage("tolerance");
            return;
        }

        try {
            subsample = Integer.parseInt(System.getProperty("subSample"));
            System.out.println("Subsample is set to" + subsample);
        } catch (Exception e) {
            printLogMessage("subsample");
            return;
        }

        try {
            subsampleNorm = Integer.parseInt(System.getProperty("subSampleNorm"));
            System.out.println("SubsampleNorm is set to" + subsampleNorm);
        } catch (Exception e) {
            printLogMessage("subsampleNorm");
            return;
        }

        try {
            q = Integer.parseInt(System.getProperty("q"));
            System.out.println("No of q is set to" + q);
        } catch (Exception e) {
            printLogMessage("q");
            return;
        }

        try {
            blockSize = Integer.parseInt(System.getProperty("blockSize"));
            System.out.println("BlockSize is set to" + blockSize);
        } catch (Exception e) {
            printLogMessage("BlockSize");
            return;
        }

        try {
            stride = Integer.parseInt(System.getProperty("stride"));
            System.out.println("Stride is set to" + stride);
        } catch (Exception e) {
            printLogMessage("Stride");
            return;
        }



        try {

            if (Integer.parseInt(System.getProperty("pcs")) == nCols) {
                nPCs = nCols - 1;
                System.out
                        .println("Number of princpal components cannot be equal to number of dimension, reducing by 1");
            } else
                nPCs = Integer.parseInt(System.getProperty("pcs"));
        } catch (Exception e) {
            printLogMessage("pcs");
            return;
        }

        try {
            nClusters = Integer.parseInt(System.getProperty("clusters"));
            System.out.println("No of partition is set to" + nClusters);
        } catch (Exception e) {
            log.warn("Cluster size is set to default: " + nClusters);
        }

        try {
            maxIterations = Integer.parseInt(System.getProperty("maxIter"));
        } catch (Exception e) {
            log.warn("maximum iterations is set to default: maximum Iterations=" + maxIterations);
        }

        try {
            calculateError = Integer.parseInt(System.getProperty("calculateError"));
        } catch (Exception e) {
            log.warn(
                    "Projected Matrix will not be computed, the output path will contain the principal components only");
        }


        SparkConf conf = new SparkConf().setAppName("SSKetchPCA");//.setMaster("local[*]");//
        // TODO
        // remove
        // this
        // part
        // for
        // building
        conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.kryoserializer.buffer.max", "128m");
        JavaSparkContext sc = new JavaSparkContext(conf);

        /*if(args.length>0)
        {
            inputPath=args[0];
        }*/
        //inputPath="Wholesale.csv";
        computePrincipalComponents(sc,inputPath,outputPath,blockSize, nRows, nCols, nPCs, subsample, tolerance,
                k_plus_one_singular_value, q, maxIterations, calculateError, subsampleNorm,stride);

    }

    /**1st new Added Functions
     * to create initial block of size 100.
     */
    public static JavaRDD<org.apache.spark.mllib.linalg.Vector> listToRdd(List<Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>> Rdd, JavaSparkContext sc){
        List<org.apache.spark.mllib.linalg.Vector> rdd=new ArrayList<org.apache.spark.mllib.linalg.Vector>();
        for(int i=0;i<Rdd.size();i++){
            org.apache.spark.mllib.linalg.Vector v=Rdd.get(i)._2;
            rdd.add(v);

        }
        return sc.parallelize(rdd);
    }

    /** 2nd new added Function for indexing the RDD
     *
     */
    public static  JavaPairRDD<Integer, org.apache.spark.mllib.linalg.Vector> indexedRDD(JavaRDD<org.apache.spark.mllib.linalg.Vector> rdd) {
        return rdd.zipWithIndex().mapToPair(new PairFunction<Tuple2<org.apache.spark.mllib.linalg.Vector, Long>, Integer, org.apache.spark.mllib.linalg.Vector>() {
            @Override
            public Tuple2<Integer, org.apache.spark.mllib.linalg.Vector> call(Tuple2<org.apache.spark.mllib.linalg.Vector, Long> elemIdx) {
                return new Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>(elemIdx._2().intValue(), elemIdx._1());
            }
        });
    }


    ////renew korar code check korte hobe mone nai kichu.
    public static JavaRDD<org.apache.spark.mllib.linalg.Vector> reNewRdd(JavaSparkContext sc, int stride,
                                                                         List<org.apache.spark.mllib.linalg.Vector> newRows,
                                                                         JavaRDD<org.apache.spark.mllib.linalg.Vector> oldRDD){

        int rdd_rows=(int)oldRDD.rdd().count();
        //List<org.apache.spark.mllib.linalg.Vector> newList=new ArrayList<org.apache.spark.mllib.linalg.Vector>();
        JavaPairRDD<Integer,org.apache.spark.mllib.linalg.Vector> pairedRdd=indexedRDD(oldRDD);
        List<Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>> list=pairedRdd.take(rdd_rows);
        // org.apache.spark.mllib.linalg.Vector newVector=newRow.get(0); ///ki hocche dekho to farzana

        List<org.apache.spark.mllib.linalg.Vector> rdd=new ArrayList<org.apache.spark.mllib.linalg.Vector>();



        for(int i=stride;i<list.size();i++){
            org.apache.spark.mllib.linalg.Vector v=list.get(i)._2;
            rdd.add(v);
        }

        for(int i=0;i<newRows.size();i++){

            org.apache.spark.mllib.linalg.Vector tempVector=newRows.get(i);
            rdd.add(tempVector);


        }

        JavaRDD<org.apache.spark.mllib.linalg.Vector>finalRDD=sc.parallelize(rdd);
        return finalRDD;


    }

    public static org.apache.mahout.math.Matrix RddToMatrix(List<Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>> Rdd, int row, int col){
        org.apache.mahout.math.Matrix output=new DenseMatrix(row,col);

        for(int i=0;i<Rdd.size();i++) {
            org.apache.spark.mllib.linalg.Vector v =Rdd.get(i)._2;
            for (int j = 0; j < v.size(); j++) {
                output.setQuick(i, j, v.apply(j));
            }
        }
        return output;

    }
    public static  JavaRDD<org.apache.spark.mllib.linalg.Vector> reConstructionError(JavaRDD<org.apache.spark.mllib.linalg.Vector> block,
                                                                                     org.apache.mahout.math.Matrix identity_matrix,
                                                                                     org.apache.mahout.math.Matrix eigen_vector,int input_col,JavaSparkContext sc)
    {
        int rdd_rows = (int) block.rdd().count();
        JavaPairRDD<Integer, org.apache.spark.mllib.linalg.Vector> pairedRdd = indexedRDD(block);
        List<Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>> list = pairedRdd.take(rdd_rows);

        org.apache.mahout.math.Matrix blockMatrix = RddToMatrix(list, rdd_rows, input_col);
        org.apache.mahout.math.Matrix reConsError = blockMatrix.times(identity_matrix.minus(eigen_vector.times(eigen_vector.transpose())));
        List<org.apache.spark.mllib.linalg.Vector> matToRdd = new ArrayList<org.apache.spark.mllib.linalg.Vector>();

        for (int i = 0; i < rdd_rows; i++) {
            double[] val = new double[input_col];
            for (int j = 0; j < input_col; j++) {
                val[j] = reConsError.getQuick(i, j);


            }
            matToRdd.add(Vectors.dense(val));

        }

        return sc.parallelize(matToRdd);
    }


    public static SvdOutput initialFunction(JavaSparkContext sc, JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors,
                                            final int nRows, final int nCols, final int nPCs, final int subsample,
                                            final double tolerance, final double k_plus_one_singular_value, final int q, final int maxIterations,
                                            final int calculateError, final int subsampleNorm){
        startTime = System.currentTimeMillis();
        // 1. Mean Job : This job calculates the mean and span of the columns of
        // the input RDD<org.apache.spark.mllib.linalg.Vector>


        final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
        final double[] internalSumY = new double[nCols];
        vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

            public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
                org.apache.spark.mllib.linalg.Vector yi;

                int i;
                while (arg0.hasNext()) {
                    yi = arg0.next();

                    for(i=0;i<yi.size();i++){
                        internalSumY[i]+=yi.apply(i);
                    }
                }
                /*System.out.println("Hello hello!");
                for(int k=0;k<internalSumY.length;k++){
                    System.out.println(internalSumY[k]);
                }*/
                matrixAccumY.add(internalSumY);
            }

        });// End Mean Job

        // Get the sum of column Vector from the accumulator and divide each
        // element by the number of rows to get the mean
        // not best of practice to use non-final variable

        long input_rows=vectors.rdd().count();

        //why should i use final value? amr to mean ber ber change hocchd? so meanVector.br_ym_mahout final rakhbo na ami
        Vector meanVector = new DenseVector(matrixAccumY.value()).divide(input_rows);
        Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);


        endTime = System.currentTimeMillis();
        totalTime = endTime - startTime;


        Stat stat=new Stat();
        stat.preprocessTime = (double) totalTime / 1000.0;

        stat.totalRunTime = stat.preprocessTime;

        stat.appName = "SSketchPCA";
        stat.dataSet = "inputPath";
        stat.nRows = nRows;
        stat.nCols = nCols;


        SvdOutput svdOut=sSketch(sc, stat, vectors, br_ym_mahout, meanVector, nRows, nCols, nPCs,
                subsample, tolerance, k_plus_one_singular_value, q, maxIterations, calculateError, subsampleNorm);








        return svdOut;
    }


    public static SvdOutput sSketch(JavaSparkContext sc, Stat stat,
                                    JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors, final Broadcast<Vector> br_ym_mahout,
                                    final Vector meanVector, final int nRows, final int nCols, final int nPCs,
                                    final int subsample, final double tolerance, final double k_plus_one_singular_value, final int q,
                                    final int maxIterations, final int calculateError, final int subsampleNorm) {

        startTime = System.currentTimeMillis();


        /************************** SSketchPCA PART *****************************/

        /**
         * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S; QR
         * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
         */

        // initialize & broadcast a random seed
        // org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix =
        // org.apache.spark.mllib.linalg.Matrices.randn(nCols,
        // nPCs + subsample, new SecureRandom());
        // //PCAUtils.printMatrixToFile(GaussianRandomMatrix,
        // OutputFormat.DENSE, outputPath+File.separator+"Seed");
        // final Matrix seedMahoutMatrix =
        // PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
        /**
         * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S; QR
         * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
         */

        // initialize & broadcast a random seed
        org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
                nPCs + subsample, new SecureRandom());
        //PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE, outputPath + File.separator + "Seed");
        Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
        // Matrix GaussianRandomMatrix = PCAUtils.randomValidationMatrix(nCols,
        // nPCs + subsample);
        // Matrix B = GaussianRandomMatrix;
        // PCAUtils.printMatrixToFile(PCAUtils.convertMahoutToSparkMatrix(GaussianRandomMatrix),
        // OutputFormat.DENSE, outputPath+File.separator+"Seed");

        final int s=nPCs+subsample;
        // Broadcast Y2X because it will be used in several jobs and several
        // iterations.
        Matrix V=null;
        double[] S=null;
        double spectral_error=0,error,prevError=tolerance+1;

        int rdd_rows = (int) vectors.rdd().count();

        for (int iter = 0; iter < maxIterations&&prevError>tolerance; iter++) {

            Matrix Seed=B;


            final Broadcast<Matrix> br_Seed = sc.broadcast(Seed);
            // Xm = Ym * Y2X
            Vector zm_mahout = new DenseVector(s);
            zm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), Seed, zm_mahout);

            // Broadcast Xm because it will be used in several iterations.
            final Broadcast<Vector> br_zm_mahout = sc.broadcast(zm_mahout);
            // We skip computing X as we generate it on demand using Y and Y2X

            // 3. X'X and Y'X Job: The job computes the two matrices X'X and Y'X
            /**
             * Xc = Yc * MEM (MEM is the in-memory broadcasted matrix Y2X)
             *
             * XtX = Xc' * Xc
             *
             * YtX = Yc' * Xc
             *
             * It also considers that Y is sparse and receives the mean vectors Ym
             * and Xm separately.
             *
             * Yc = Y - Ym
             *
             * Xc = X - Xm
             *
             * Xc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = X - Xm
             *
             * XtX = (X - Xm)' * (X - Xm)
             *
             * YtX = (Y - Ym)' * (X - Xm)
             *
             */
            final Accumulator<double[][]> matrixAccumZtZ = sc.accumulator(new double[s][s],
                    new MatrixAccumulatorParam());
            final Accumulator<double[][]> matrixAccumYtZ = sc.accumulator(new double[nCols][s],
                    new MatrixAccumulatorParam());
            final Accumulator<double[]> matrixAccumZ = sc.accumulator(new double[s], new VectorAccumulatorParam());

            /*
             * Initialize the output matrices and vectors once in order to avoid
             * generating massive intermediate data in the workers
             */
            final double[][] resArrayYtZ = new double[nCols][s];
            final double[][] resArrayZtZ = new double[s][s];
            final double[] resArrayZ = new double[s];

            /*
             * Used to sum the vectors in one partition.
             */
            final double[][] internalSumYtZ = new double[nCols][s];
            final double[][] internalSumZtZ = new double[s][s];
            final double[] internalSumZ = new double[s];

            vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

                public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
                    org.apache.spark.mllib.linalg.Vector yi;
                    while (arg0.hasNext()) {
                        yi = arg0.next();

                        /*
                         * Perform in-memory matrix multiplication xi = yi' * Y2X
                         */
                        PCAUtils.dense_mllib_VectorTimesMatrix(yi,br_Seed.value(),resArrayZ);

                        // get only the sparse indices
                        //int[] indices = ((SparseVector) yi).indices();

                        /*PCAUtils.outerProductWithIndices(yi, br_ym_mahout.value(), resArrayZ, br_zm_mahout.value(),
                                resArrayYtZ, indices);*/
                        PCAUtils.outerProductWithoutIndices(yi, br_ym_mahout.value(), resArrayZ, br_zm_mahout.value(),
                                resArrayYtZ);
                        PCAUtils.outerProductArrayInput(resArrayZ, br_zm_mahout.value(), resArrayZ, br_zm_mahout.value(),
                                resArrayZtZ);
                        int i, j, rowIndexYtZ;

                        // add the sparse indices only
                       /* for (i = 0; i < indices.length; i++) {
                            rowIndexYtZ = indices[i];
                            for (j = 0; j < s; j++) {
                                internalSumYtZ[rowIndexYtZ][j] += resArrayYtZ[rowIndexYtZ][j];
                                resArrayYtZ[rowIndexYtZ][j] = 0; // reset it
                            }

                        }*/

                        for(i=0;i<yi.size();i++){
                            for(j=0;j<s;j++){
                                internalSumYtZ[i][j]+=resArrayYtZ[i][j];
                                resArrayYtZ[i][j]=0;
                            }
                        }
                        for (i = 0; i < s; i++) {
                            internalSumZ[i] += resArrayZ[i];
                            for (j = 0; j < s; j++) {
                                internalSumZtZ[i][j] += resArrayZtZ[i][j];
                                resArrayZtZ[i][j] = 0; // reset it
                            }

                        }
                    }
                    matrixAccumZ.add(internalSumZ);
                    matrixAccumZtZ.add(internalSumZtZ);
                    matrixAccumYtZ.add(internalSumYtZ);
                }

            });// end X'X and Y'X Job

            /*
             * Get the values of the accumulators.
             */
            Matrix centralYtZ = new DenseMatrix(matrixAccumYtZ.value());
            Matrix centralZtZ = new DenseMatrix(matrixAccumZtZ.value());
            Vector centralSumZ = new DenseVector(matrixAccumZ.value());

            /*
             * Mi = (Yi-Ym)' x (Xi-Xm) = Yi' x (Xi-Xm) - Ym' x (Xi-Xm)
             *
             * M = Sum(Mi) = Sum(Yi' x (Xi-Xm)) - Ym' x (Sum(Xi)-N*Xm)
             *
             * The first part is done in the previous job and the second in the
             * following method
             */
            centralYtZ = PCAUtils.updateXtXAndYtx(centralYtZ, centralSumZ, br_ym_mahout.value(), zm_mahout, rdd_rows);//nRows chnge korlam
            centralZtZ = PCAUtils.updateXtXAndYtx(centralZtZ, centralSumZ, zm_mahout, zm_mahout, rdd_rows);//nRows change korsi


            Matrix R = new org.apache.mahout.math.CholeskyDecomposition(centralZtZ, false).getL().transpose();


            R = PCAUtils.inv(R);
            centralYtZ=centralYtZ.times(R);
            centralYtZ=centralYtZ.transpose();



            org.apache.mahout.math.SingularValueDecomposition SVD =
                    new org.apache.mahout.math.SingularValueDecomposition(centralYtZ);

            B=centralYtZ.transpose();

            V = SVD.getV().viewPart(0, nCols, 0, nPCs);

            S=SVD.getSingularValues();
            double k_plus_one_value=S[nPCs];
            stat.kSingularValue=k_plus_one_value;


            endTime = System.currentTimeMillis();
            totalTime = endTime - startTime;
            double time= (double) totalTime / 1000.0;
            stat.sketchTime.add(time);
            stat.totalRunTime += time;

            if (calculateError == 1) {
                // log.info("Computing the error at round " + round + " ...");
                System.out.println("Computing the error at round " + iter + " ...");

                stat.nIter++;

                // the following subsample is fixed
                spectral_error = norm(sc, vectors, nRows, nCols, 1, subsampleNorm, q, meanVector, V,br_ym_mahout);
                error = (spectral_error - k_plus_one_value) / k_plus_one_value;

                stat.errorList.add((Double) error);
                // log.info("... end of computing the error at round " + round +
                // " And error=" + error);
                System.out.println("... end of computing the error at round " + iter + " error=" + error);
                prevError = error;
            }
           // stat.spectral_error=spectral_error;
            /**
             * reinitialize
             */
            startTime = System.currentTimeMillis();



        }


        SvdOutput svdOut=new SvdOutput(nCols,nPCs);
        svdOut.S=S;
        svdOut.V=V;
        svdOut.stat=stat;

        return svdOut;
    }

    private static double norm(JavaSparkContext sc, final JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors,
                               final int nRows, final int nCols, final int nPCs, final int subsample, final int q, final Vector meanVector,
                               final Matrix centralC, final Broadcast<Vector> br_ym_mahout) {
        /************************** SSketchPCA PART *****************************/

        /**
         * Sketch dimension ,S=s Sketched matrix, B=A*S; QR decomposition,
         * Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
         */

        // initialize & broadcast a random seed
        org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
                nPCs + subsample, new SecureRandom());
        // PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE,
        // outputPath+File.separator+"Seed");
        Matrix B = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);

        System.out.println(PCAUtils.convertMahoutToSparkMatrix(centralC));
        Matrix V = new org.apache.mahout.math.SingularValueDecomposition(centralC).getU();

        org.apache.mahout.math.SingularValueDecomposition SVD = null;

        double S = 0;

        for (int iter = 0; iter < q; iter++) {
            // V'*omega
            final Matrix VtSeed = V.transpose().times(B);
            // V*V'*omega
            final Matrix VVtSeed = V.times(VtSeed);
            // omega-V*V'*omega
            Matrix Seed = B.minus(VVtSeed);

            // System.out.println(brSeedMu.value().getQuick(5));


            final int s=nPCs+subsample;
            // Broadcast Seed because it will be used in several jobs and several
            // iterations.
            final Broadcast<Matrix> br_Seed = sc.broadcast(Seed);

            // Zm = Ym * Seed
            Vector zm_mahout = new DenseVector(s);
            zm_mahout = PCAUtils.denseVectorTimesMatrix(br_ym_mahout.value(), Seed, zm_mahout);

            // Broadcast Zm because it will be used in several iterations.
            final Broadcast<Vector> br_zm_mahout = sc.broadcast(zm_mahout);
            // We skip computing Z as we generate it on demand using Y and Seed

            // 3. Z'Z and Y'Z Job: The job computes the two matrices Z'Z and Y'Z
            /**
             * Zc = Yc * MEM (MEM is the in-memory broadcasted matrix seed)
             *
             * ZtZ = Zc' * Zc
             *
             * YtZ = Yc' * Zc
             *
             * It also considers that Y is sparse and receives the mean vectors Ym
             * and Xm separately.
             *
             * Yc = Y - Ym
             *
             * Zc = Z - Zm
             *
             * Zc = (Y - Ym) * MEM = Y * MEM - Ym * MEM = Z - Zm
             *
             * ZtZ = (Z - Zm)' * (Z - Zm)
             *
             * YtZ = (Y - Ym)' * (Z - Zm)
             *
             */
            final Accumulator<double[][]> matrixAccumZtZ = sc.accumulator(new double[s][s],
                    new MatrixAccumulatorParam());
            final Accumulator<double[][]> matrixAccumYtZ = sc.accumulator(new double[nCols][s],
                    new MatrixAccumulatorParam());
            final Accumulator<double[]> matrixAccumZ = sc.accumulator(new double[s], new VectorAccumulatorParam());

            /*
             * Initialize the output matrices and vectors once in order to avoid
             * generating massive intermediate data in the workers
             */
            final double[][] resArrayYtZ = new double[nCols][s];
            final double[][] resArrayZtZ = new double[s][s];
            final double[] resArrayZ = new double[s];

            /*
             * Used to sum the vectors in one partition.
             */
            final double[][] internalSumYtZ = new double[nCols][s];
            final double[][] internalSumZtZ = new double[s][s];
            final double[] internalSumZ = new double[s];

            vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {
                public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
                    org.apache.spark.mllib.linalg.Vector yi;
                    while (arg0.hasNext()) {
                        yi = arg0.next();

                        /*
                         * Perform in-memory matrix multiplication zi = yi' * Seed
                         */
                        PCAUtils.dense_mllib_VectorTimesMatrix(yi, br_Seed.value(), resArrayZ);

                        // get only the sparse indices
                        //int[] indices = ((SparseVector) yi).indices();

                        PCAUtils.outerProductWithoutIndices(yi, br_ym_mahout.value(), resArrayZ, br_zm_mahout.value(),
                                resArrayYtZ);
                        PCAUtils.outerProductArrayInput(resArrayZ, br_zm_mahout.value(), resArrayZ, br_zm_mahout.value(),
                                resArrayZtZ);
                        int i, j, rowIndexYtZ;

                        // add the sparse indices only
                        /*for (i = 0; i < indices.length; i++) {
                            rowIndexYtZ = indices[i];
                            for (j = 0; j < s; j++) {
                                internalSumYtZ[rowIndexYtZ][j] += resArrayYtZ[rowIndexYtZ][j];
                                resArrayYtZ[rowIndexYtZ][j] = 0; // reset it
                            }

                        }*/
                        for( i=0;i<yi.size();i++){
                            for(j=0;j<s;j++){
                                internalSumYtZ[i][j]+=resArrayYtZ[i][j];
                                resArrayYtZ[i][j]=0;
                            }
                        }
                        for (i = 0; i < s; i++) {
                            internalSumZ[i] += resArrayZ[i];
                            for (j = 0; j < s; j++) {
                                internalSumZtZ[i][j] += resArrayZtZ[i][j];
                                resArrayZtZ[i][j] = 0; // reset it
                            }

                        }
                    }
                    matrixAccumZ.add(internalSumZ);
                    matrixAccumZtZ.add(internalSumZtZ);
                    matrixAccumYtZ.add(internalSumYtZ);
                }

            });// end Z'Z and Y'Z Job

            /*
             * Get the values of the accumulators.
             */
            Matrix centralYtZ = new DenseMatrix(matrixAccumYtZ.value());
            Matrix centralZtZ = new DenseMatrix(matrixAccumZtZ.value());
            Vector centralSumZ = new DenseVector(matrixAccumZ.value());

            int rdd_rows = (int) vectors.rdd().count();

            centralYtZ = PCAUtils.updateXtXAndYtx(centralYtZ, centralSumZ, br_ym_mahout.value(), zm_mahout, rdd_rows); //nRows change korsi
            centralZtZ = PCAUtils.updateXtXAndYtx(centralZtZ, centralSumZ, zm_mahout, zm_mahout, rdd_rows); //nRows chnge korsi
            Matrix R = new org.apache.mahout.math.CholeskyDecomposition(centralZtZ, false).getL().transpose();

            R = PCAUtils.inv(R);
            centralYtZ=centralYtZ.times(R);
            centralYtZ=centralYtZ.transpose();

            final Matrix QtAV = centralYtZ.times(V);
            final Matrix QtAVVt = QtAV.times(V.transpose());
            B = centralYtZ.minus(QtAVVt);

            SVD = new org.apache.mahout.math.SingularValueDecomposition(B);

            B = B.transpose();

            Double newS = SVD.getS().getQuick(nPCs - 1, nPCs - 1);
            newS = Math.round(newS * 10000.0) / 10000.0;
            if (newS == S)
                break;
            else
                S = newS;
            System.out.println(S);
        }

        return S;
    }












    public static org.apache.spark.mllib.linalg.Matrix computePrincipalComponents(JavaSparkContext sc, String inputPath, String outputPath, int blockSize, final int nRows, final int nCols, final int nPCs, final int subsample,
                                                                                  final double tolerance, final double k_plus_one_singular_value, final int q, final int maxIterations,
                                                                                  final int calculateError, final int subsampleNorm,int stride) {

        /**
         * preprocess the data
         *
         * @param nClusters
         *
         */



        JavaRDD<String> data = sc.textFile(inputPath ,nClusters);
        System.out.println("*******\nNumber of clusters: "+nClusters+"*******\n\n");



        JavaRDD<org.apache.spark.mllib.linalg.Vector> datamain = data.map(new Function<String, org.apache.spark.mllib.linalg.Vector>(){
                                                                              public org.apache.spark.mllib.linalg.Vector call(String s){
                                                                                  String[] sarray = s.trim().split(",");
                                                                                  double[] values = new double[sarray.length];
                                                                                  for (int i = 0; i < sarray.length; i++) {
                                                                                      values[i] = Double.parseDouble(sarray[i]);
                                                                                      //System.out.println(values[i]);
                                                                                  }
                                                                                  return Vectors.dense(values);
                                                                              }
                                                                          }
        ).persist(StorageLevel.MEMORY_ONLY_SER());


        int input_Columns=datamain.first().size();
        long input_rows=datamain.rdd().count();


        ///changed By Farzana to create initial Block.
        JavaPairRDD<Integer, org.apache.spark.mllib.linalg.Vector> indexKey=indexedRDD(datamain); //protita row of rdd ke ekta key index disse.

        List<Tuple2<Integer, org.apache.spark.mllib.linalg.Vector>> list=indexKey.take(blockSize);
        JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors=listToRdd(list,sc).persist(StorageLevel.MEMORY_ONLY_SER()); //do i need it here???



        org.apache.mahout.math.Matrix finalMatrix=new DenseMatrix(nRows,nPCs);
        org.apache.mahout.math.Matrix primaryMatrix;
        org.apache.mahout.math.Matrix outputMatrix=new DenseMatrix(blockSize,nPCs);

        primaryMatrix = RddToMatrix(list, blockSize, input_Columns);and u don't '


        org.apache.mahout.math.Matrix eigen_vector=new DenseMatrix(input_Columns,nPCs);
        org.apache.mahout.math.Matrix identity_matrix=new DenseMatrix(input_Columns,input_Columns);

        for(int i=0;i<input_Columns;i++){
            for(int j=0;j<input_Columns;j++)  if(i==j) identity_matrix.setQuick(i,j,1.0);
        }


        vectors=reConstructionError(vectors,identity_matrix,eigen_vector,input_Columns,sc).persist(StorageLevel.MEMORY_ONLY_SER());



        int cntEnd=0;
        double cntTime=0;
        double cntError=0;



        SvdOutput svdOut=initialFunction(sc,vectors,nRows,nCols,nPCs,subsample,tolerance,k_plus_one_singular_value,
                q,maxIterations,calculateError,subsampleNorm);




        for (int j = 0; j < svdOut.stat.sketchTime.size(); j++) {
            svdOut.stat.avgSketchTime += svdOut.stat.sketchTime.get(j);
        }
        svdOut.stat.avgSketchTime /= svdOut.stat.sketchTime.size();


        int blkNum=1;
        svdOut.stat.blockNum=blkNum;

        //PCAUtils.printStatToFile(stat,nClusters,nPCs,subsample, outputPath,blockSize,stride, blkNum);


        double[] S=new double[nPCs];///singularr Value


        for(int i=0;i<nPCs;i++){
            S[i]=svdOut.S[i];

        }


        eigen_vector=svdOut.V.viewPart(0,nCols,0,nPCs);  //right singular value  /// Ranak vaiar code e etar name silo eigen_vctor

        outputMatrix=primaryMatrix.times(eigen_vector);
        int itr;
        for( itr=0;itr<outputMatrix.numRows();itr++){
            for(int itrCol=0;itrCol<outputMatrix.numCols();itrCol++){
                finalMatrix.setQuick(itr,itrCol,outputMatrix.getQuick(itr,itrCol));
            }

        }
        itr=outputMatrix.numRows();




        JavaRDD<org.apache.spark.mllib.linalg.Vector> newVectors=vectors;




        ///calculate Spectral Error

        SvdOutput spectral_svd;

        JavaRDD<org.apache.spark.mllib.linalg.Vector> spectralRdd;
        spectralRdd=reConstructionError(newVectors,identity_matrix,eigen_vector,input_Columns,sc).persist(StorageLevel.MEMORY_ONLY_SER());

        spectral_svd=initialFunction(sc,spectralRdd,blockSize,nCols,1,1,0.05,1,10,10,0,1);;
        double[] singma=spectral_svd.S;
        svdOut.stat.spectral_error=singma[0];
        stat_list.add(svdOut.stat);

        System.out.println("cntEnd: ");
        System.out.print(cntEnd);

        cntTime+=svdOut.stat.totalRunTime;
        cntError+=svdOut.stat.spectral_error;




        for(int i=blockSize;i<input_rows;i+=stride){
            cntEnd++;



            int min_ind=-1;
            double S_min=100000000;

            for(int k=0;k<nPCs;k++){
                if(S[k]<S_min){
                    S_min=S[k];
                    min_ind=k;
                }
            }
            startTime = System.currentTimeMillis();
            //stat.avgSketchTime=0;

            List<org.apache.spark.mllib.linalg.Vector> newRows=
                    new ArrayList<org.apache.spark.mllib.linalg.Vector>();


            /****this part is added to control stride and create input matrix
             *
             */



            int count=0;


            if((i+stride)>=input_rows){
                stride=(int)input_rows-i;
            }

            org.apache.mahout.math.Matrix inputMatrix=new DenseMatrix(stride,nCols);
            org.apache.mahout.math.Matrix outMatrix;


            for (int j = i; j < (i + stride); j++) {

                List<org.apache.spark.mllib.linalg.Vector> tempRow = indexKey.lookup(j);
                org.apache.spark.mllib.linalg.Vector tempVector = tempRow.get(0);
                for (int k = 0; k < tempVector.size(); k++) {
                    inputMatrix.setQuick(count, k, tempVector.apply(k));
                }
                count++;

                newRows.add(tempVector);
            }


            newVectors=reNewRdd(sc,stride,newRows,newVectors).persist(StorageLevel.MEMORY_ONLY_SER());




            newVectors=reConstructionError(newVectors,identity_matrix,eigen_vector,input_Columns,sc).persist(StorageLevel.MEMORY_ONLY_SER());

            SvdOutput dummy_svdOut=initialFunction(sc,newVectors,nRows,nCols,nPCs,subsample,tolerance,k_plus_one_singular_value,
                    q,maxIterations,calculateError,subsampleNorm);



            for (int j = 0; j < dummy_svdOut.stat.sketchTime.size(); j++) {
                dummy_svdOut.stat.avgSketchTime += dummy_svdOut.stat.sketchTime.get(j);
            }
            dummy_svdOut.stat.avgSketchTime /= dummy_svdOut.stat.sketchTime.size();


            // save statistics



            double[] dummy_S=new double[nPCs];
            //System.out.println("Printing dummy_s****************");
            double dummy_max=-1000000000;
            int max_ind=-1;
            for(int j=0;j<nPCs;j++){
                dummy_S[j]=dummy_svdOut.S[j];
                //System.out.print(dummy_S[j]);
                //System.out.print(" ");

                if(dummy_S[j]>dummy_max){
                    max_ind=j;
                    dummy_max=dummy_S[j];
                }

            }



            org.apache.mahout.math.Matrix dummy_V=dummy_svdOut.V.viewPart(0,nCols,0,nPCs);

            if(dummy_max>S_min){
                S[min_ind]=dummy_max;
                dummy_svdOut.stat.changeDist.add(itr);
                for(int j=0;j<nCols;j++){
                   // System.out.println("Changes occurs in min_val");
                    eigen_vector.setQuick(j,min_ind,dummy_V.get(j,max_ind));

                }
            }

            blkNum++;
            //PCAUtils.printStatToFile(stat,nClusters,nPCs,subsample, outputPath,blockSize,stride, blkNum);
            dummy_svdOut.stat.blockNum=blkNum;


            outMatrix=inputMatrix.times(eigen_vector);

            int tmpItr=0;
            //System.out.println("stride: ");
            //System.out.println(stride);
            while(tmpItr<stride){
                //System.out.println(itr);


                for(int itrCol=0;itrCol<outMatrix.numCols();itrCol++){
                    finalMatrix.setQuick(itr,itrCol,outMatrix.getQuick(tmpItr,itrCol));
                }
                itr++;
                tmpItr++;
            }

            ///calculate Spectral Error

            //JavaRDD<org.apache.spark.mllib.linalg.Vector> spectralRdd;
            spectralRdd=reConstructionError(newVectors,identity_matrix,eigen_vector,input_Columns,sc).persist(StorageLevel.MEMORY_ONLY_SER());
            spectral_svd=initialFunction(sc,spectralRdd,blockSize,nCols,1,1,0.05,1,10,10,0,1);;
            singma=spectral_svd.S;

             dummy_svdOut.stat.spectral_error=singma[0];
             stat_list.add(dummy_svdOut.stat);

            /*System.out.println("i+ stride: ");
            System.out.println(i+stride);*/


            cntTime+=dummy_svdOut.stat.totalRunTime;
            cntError+=dummy_svdOut.stat.spectral_error;

            System.out.println("cntEnd: ");
            System.out.println(cntEnd);


            System.out.println("runTime: ");
            System.out.println(dummy_svdOut.stat.totalRunTime);

            System.out.println("Spectral Error: ");
            System.out.println(dummy_svdOut.stat.spectral_error);

            if(cntEnd==20) break;






        }


        System.out.println("Mean runtime: ");
        System.out.println(cntTime/stat_list.size());


        System.out.println("Mean spectral Error: ");
        System.out.println(cntError/stat_list.size());




        //PCAUtils.printMatrix(finalMatrix,nClusters,nPCs,subsample,outputPath,blockSize,stride);
        //PCAUtils.printPCAtoFile(S,nClusters,nPCs,subsample,eigen_vector,outputPath,blockSize,stride);
        //PCAUtils.printStatToFile_small(stat_list,nClusters,nPCs,subsample, outputPath,blockSize,stride);

        System.out.println("The End of a forLoop");


        return null;
    }

    private static void printLogMessage(String argName) {
        log.error("Missing arguments -D" + argName);
        log.info(
                "Usage: -Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DcalculateError=<0/1 (compute projected matrix or not)>] [-DBlockSize=<Block Size>] [-DStride=<Stride Size>]");
    }


}