package org.ssketch;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;


public class SvdOutput {
    public double[] S;
    Matrix V;
    public Stat stat;

    SvdOutput(int col,int nPCs){
        S=new double[nPCs];
        V=new DenseMatrix(col,nPCs);
        stat=new Stat();
    }
}
