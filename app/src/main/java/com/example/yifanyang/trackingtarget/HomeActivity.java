package com.example.yifanyang.trackingtarget;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class HomeActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        Button bPyramids,bOptFlowKLT;
        bPyramids = (Button) findViewById(R.id.bPyramids);
        bOptFlowKLT = (Button) findViewById(R.id.bOptFlowKLT);
        bOptFlowKLT.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(getApplicationContext(),MainActivity.class);
                startActivity(i);
            }
        });
        bPyramids.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(getApplicationContext(),PyramidActivity.class);
                startActivity(i);
            }
        });

    }
}
