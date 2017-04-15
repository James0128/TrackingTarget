package com.example.yifanyang.trackingtarget;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String TAG= " .MainActivity";

    private static final int VIEW_MODE_KLT_TRACKER = 0;
    private static final int VIEW_MODE_OPTICAL_FLOW =1;
    private int mViewMode;
    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mGray;
    private Mat mPrevGray;

    MatOfPoint2f prevFeatures,nextFeatures;
    MatOfPoint features;

    MatOfByte status;
    MatOfFloat err;
    private MenuItem mItemPreviewOpticalFlow,mItemPreviewKLT;
    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mloaderCallback= new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }

    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG,"called onCreate");
        super.onCreate(savedInstanceState);
        //防止运行应用时屏幕关闭
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG,"called onCreateOptionsMenu");
        mItemPreviewKLT = menu.add("KLT  Tracker");
        mItemPreviewOpticalFlow = menu.add("Optical Flow");
        return true;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null){
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,this,mloaderCallback);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null){
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
      mRgba = new Mat(height,width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height,width,CvType.CV_8UC4);
        mGray = new Mat(height , width,CvType.CV_8UC1);
        resetVars();
    }
    //重置所有Mat对象
    private void resetVars() {
        mPrevGray = new Mat(mGray.rows(),mGray.cols(),CvType.CV_8UC1);
        features = new MatOfPoint();
        prevFeatures = new MatOfPoint2f();
        nextFeatures = new MatOfPoint2f();
        status = new MatOfByte();
        err = new MatOfFloat();

    }

    @Override
    public void onCameraViewStopped() {
     mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG,"called onOptionsItemSelected");
        if (item ==mItemPreviewOpticalFlow){
            mViewMode = VIEW_MODE_OPTICAL_FLOW;
            resetVars();
        }else if (item == mItemPreviewKLT){
            mViewMode = VIEW_MODE_KLT_TRACKER;
            resetVars();
        }
        return true;
    }

    /*
        *处理每一帧
        * */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode){
            case VIEW_MODE_OPTICAL_FLOW:
                mGray = inputFrame.gray();//gray用于获取包含灰度格式被捕捉帧的MAt对象
                //如果是第一次运行，创建并填充一个Features数组，保存所有网格点的位置，网格用于计算光流
                if (features.toArray().length==0){
                    int rowStep = 50,colStep =100;
                    int nRows = mGray.rows()/rowStep,
                            ncols=mGray.cols()/colStep;
                    Log.i("aaa","\\nRows: \"+nRows+\"\nCols: \"+nCols+\"\\n");

                    Point points[] = new Point[nRows*ncols];
                    for (int i=0;i<nRows;i++){
                        for (int j=0;j<ncols;j++){
                            points[i*ncols+j]= new Point(j*colStep,i*rowStep);
                            Log.d("bbb","\\nRow: \"+i*rowStep+\"\\nCol: \"+j*colStep+\"\\n: ");
                        }
                    }
                    features.fromArray(points);

                    //点拷贝到preFeatures对象，用来计算光流场
                    prevFeatures.fromList(features.toList());
                    mPrevGray = mGray.clone();
                    break;
                }
                //下一帧保存在nextFeatures
                nextFeatures.fromArray(prevFeatures.toArray());
                Video.calcOpticalFlowPyrLK(mPrevGray,mGray,prevFeatures,nextFeatures,status,err);
                //绘制网格每点运动的线段
                List<Point> prevList= features.toList(),
                        nextList = nextFeatures.toList();
                Scalar color = new Scalar(255);

                for (int i=0;i<prevList.size();i++){
                    Imgproc.line(mGray,prevList.get(i),nextList.get(i),color);
                }
                mPrevGray=mGray.clone();
                break;
            case VIEW_MODE_KLT_TRACKER:
                mGray = inputFrame.gray();

                if (features.toArray().length == 0){
                    Imgproc.goodFeaturesToTrack(mGray,features,10,0.01,10);
                    Log.d("ccc",features.toList().size()+"");
                    mPrevGray=mGray.clone();
                    break;
                }
                //Video.calcOpticalFlowPyrLK(mPrevGray,mGray,prevFeatures,nextFeatures,status,err);
                List<Point> drawFeatures= nextFeatures.toList();
                for (int i = 0;i<drawFeatures.size();i++){
                    Point p = drawFeatures.get(i);
                    Imgproc.circle(mGray,p,5,new Scalar(255));
                }
                mPrevGray = mGray.clone();
                prevFeatures.fromList(nextFeatures.toList());
                break;
            default:
                mViewMode= VIEW_MODE_KLT_TRACKER;
        }

        return mGray;
    }
    
}
