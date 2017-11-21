package io.github.wzzju.clnet;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("clnet-lib");
    }

    private Button run;
    private Button clear;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        run = findViewById(R.id.run);
        clear = findViewById(R.id.clear);
        textView = findViewById(R.id.content);
    }

    public void onRun(View v) {
        textView.setText(stringFromJNI());
    }

    public void onClear(View v) {
        textView.setText("Empty");
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}
