package io.github.wzzju.clnet;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.apache.http.util.EncodingUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("clnet");
    }

    private Button run;
    private Button clear;
    private TextView textView;
    private ProgressBar progress;
    private String clPath;
    private final String TAG = "CLNET";

    StringBuilder content = new StringBuilder("*************************START*************************\n");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        run = findViewById(R.id.run);
        clear = findViewById(R.id.clear);
        textView = findViewById(R.id.content);
        progress = findViewById(R.id.progress);
        new AsyncCopyKernel().execute("matvec.cl");
    }

    public void onRun(View v) {
        textView.setText(content.toString() + runCL(clPath));
        deviceQuery();
    }

    public void onClear(View v) {
        textView.setText("Empty");
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    public native String runCL(String path);

    public native void deviceQuery();

    private void setBtns(boolean isEnabled) {
        run.setEnabled(isEnabled);
        clear.setEnabled(isEnabled);
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
             "/data/user/0/io.github.wzzju.clnet/app_execdir/matvec.cl"
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
