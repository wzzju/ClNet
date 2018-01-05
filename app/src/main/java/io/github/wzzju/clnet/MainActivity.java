package io.github.wzzju.clnet;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.squareup.picasso.Picasso;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("clnet");
    }

    private Activity activity = this;//获取当前的context
    private ProgressBar progress;
    private ImageView imageView;
    private TextView textView;
    private Button run;
    private Button init;
    private Button clear;

    private String clPath;
    private final String TAG = "CLNET";
    private static final int SELECT_PICTURE = 9999;//选取图片的请求码
    private String selectedImagePath = null;//所选图片的路径

    StringBuilder content = new StringBuilder("***********************START***********************\n");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        progress = findViewById(R.id.progress);
        textView = findViewById(R.id.content);
        imageView = findViewById(R.id.iv_image);
        run = findViewById(R.id.run);
        init = findViewById(R.id.init);
        clear = findViewById(R.id.clear);
        imageView.setOnClickListener((v) -> {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE);
        });
        new AsyncCopyKernel().execute("clnet.cl");
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                if (CheckPermission.checkPermissionRead(activity))
                    selectedImagePath = Utility.getPathByData(activity, data);
//                Log.d(TAG, selectedImagePath);
                if (selectedImagePath != null)
                    imageView.setImageURI(selectedImageUri);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case CheckPermission.CLNET_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(activity, "GRANTED READ PERMISSION!",
                            Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(activity, "GRANTING READ PERMISSION IS DENIED!",
                            Toast.LENGTH_SHORT).show();
                }
                break;
            case CheckPermission.CLNET_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(activity, "GRANTED WRITE PERMISSION!",
                            Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(activity, "GRANTING WRITE PERMISSION IS DENIED!",
                            Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions,
                        grantResults);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    public void onRun(View v) {
        textView.setText("Run");
        textView.setText(content.toString() + runCL(clPath));
//        runNpy("/data/local/tmp/clnet/lenet_model/");
//        new AsyncProcessImage().execute();
    }

    public void onInit(View v) {
        textView.setText("Init");
        deviceQuery();
    }

    public void onClear(View v) {
        textView.setText("Empty");
        content.setLength(0);
        content.append("***********************START***********************\n");
    }


    /**
     * A native method that is implemented by the 'clnet' native library,
     * which is packaged with this application.
     */
    public native float[] inference(float[] data);

    public native String runCL(String path);

    public native void runNpy(String dir);

    public native void deviceQuery();

    private void setBtns(boolean isEnabled) {
        run.setEnabled(isEnabled);
        init.setEnabled(isEnabled);
        clear.setEnabled(isEnabled);
    }

    private class AsyncProcessImage extends AsyncTask<Void, Void, Void> {
        private Bitmap bm = null;

        @Override
        protected void onPreExecute() {
            setBtns(false);
            progress.setVisibility(View.VISIBLE);
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(Void v) {
            setBtns(true);
            progress.setVisibility(View.GONE);
            textView.setText(content.toString());
            super.onPostExecute(v);
        }

        @Override
        protected Void doInBackground(Void... voids) {
            /****************************单张图片的推断****************************/
            float[] data = getImageData();
            if (data != null) {
                float[] result = inference(data);
                float max = 0;
                int max_i = -1;
                for (int i = 0; i < 10; ++i) {
                    float value = result[i];
                    if (max < value) {
                        max = value;
                        max_i = i;
                    }
                }
                String inferenceResult = String.format(Locale.ENGLISH, "Max prob : %f, Class : %d\n", max, max_i);
                Log.d(TAG, inferenceResult);
                content.append(inferenceResult);
            }
            return null;
        }

        /**
         * 测试网络精度
         * CORRECT : 9916
         * TOTAL : 10000
         * ACCURACY : 0.9916
         * Cost time : 715797 ms
         */
        private void netAccuracy() {
            File file = new File("/data/local/tmp/mnist/test/test.txt");
            BufferedReader reader = null;
            int total = 0, correct = 0;
            long start = System.currentTimeMillis();
            try {
                reader = new BufferedReader(new FileReader(file));
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] keys = line.split(" ");
                    selectedImagePath = keys[0];
                    float[] data = getImageData();
                    float[] result = inference(data);
                    float max = 0;
                    int max_i = -1;
                    for (int i = 0; i < 10; ++i) {
                        float value = result[i];
                        if (max < value) {
                            max = value;
                            max_i = i;
                        }
                    }
                    if (max_i == Integer.parseInt(keys[1]))
                        correct++;
                    total++;
                }
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (reader != null) {
                    try {
                        reader.close();
                    } catch (IOException f) {
                    }
                }
            }
            long end = System.currentTimeMillis();
            Log.d(TAG, "CORRECT : " + correct);
            Log.d(TAG, "TOTAL : " + total);
            Log.d(TAG, "ACCURACY : " + ((double) correct / (double) total));
            Log.d(TAG, String.format(Locale.ENGLISH, "Cost time : %d ms", end - start));
        }

        private float[] getImageData() {
            if (selectedImagePath != null) {
                final int IMG_WIDTH = 28;
                final int IMG_HEIGHT = 28;
                final int IMG_C = 3;

                final float[] bitmapArray = new float[IMG_C * IMG_HEIGHT * IMG_WIDTH];

                try {
                    bm = Picasso.with(activity)
                            .load(new File(selectedImagePath))
                            .config(Bitmap.Config.ARGB_8888)
                            .resize(IMG_WIDTH, IMG_HEIGHT)
                            .get();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (bm != null) {
                    ExecutorService executor = Executors.newFixedThreadPool(8);
                    for (int c = 0; c < IMG_C; c++) {
                        for (int h = 0; h < IMG_HEIGHT; h++) {
                            final int final_C = c;
                            final int final_H = h;
                            executor.execute(() -> {
                                for (int w = 0; w < IMG_WIDTH; w++) {
                                    // The x coordinate (0...width-1) of the pixel to return
                                    // The yto coordinate (0...height-1) of the pixel to return
                                    int pixel = bm.getPixel(w, final_H);
                                    bitmapArray[final_C * IMG_HEIGHT * IMG_WIDTH + final_H * IMG_WIDTH + w] = Utility.getColorPixel(pixel, final_C);
                                }
                            });
                        }
                    }

                    executor.shutdown();
                    try {
                        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
//                Log.d(TAG, "Height = " + bm.getHeight() + ", Width = " + bm.getWidth());
                return bitmapArray;
            } else {
                return null;
            }
        }
    }

    private class AsyncCopyKernel extends AsyncTask<String, Void, String> {
        @Override
        protected void onPreExecute() {
            setBtns(false);
            progress.setVisibility(View.VISIBLE);
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(String result) {
            setBtns(true);
            progress.setVisibility(View.GONE);
            clPath = result;
            super.onPostExecute(result);
        }

        @Override
        protected String doInBackground(String... strings) {
            String path = null;
            try {
                File of = new File(
                        MainActivity.this.getDir("execdir", MainActivity.this.MODE_PRIVATE),
                        strings[0]);
                OutputStream out = new FileOutputStream(of);
                path = of.getAbsolutePath();
                InputStream in = MainActivity.this.getAssets().open(strings[0]);
                int length = in.available();
                byte[] buffer = new byte[length];
                in.read(buffer);
                out.write(buffer);
                in.close();
                out.close();
            } catch (
                    IOException e)

            {
                e.printStackTrace();
            }

            /******************************DEBUG*************************************
             try {
             FileInputStream fin = new FileInputStream(
             "/data/user/0/io.github.wzzju.clnet/app_execdir/clnet.cl"
             );
             int len = fin.available();
             byte[] buf = new byte[len];
             fin.read(buf);
             content.append(EncodingUtils.getString(buf, "UTF-8"));
             fin.close();
             } catch (IOException e) {
             e.printStackTrace();
             }
             ******************************DEBUG*************************************/

            return path;
        }
    }
}
