package com.example.cameraxx

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class Main : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.main)
        val btnCause: Button = findViewById(R.id.btnCause)
        btnCause.setOnClickListener {
            val intent = Intent(this, CauseActivity::class.java)
            startActivity(intent)
        }

        val btnSymptoms: Button = findViewById(R.id.btnSymptoms)
        btnSymptoms.setOnClickListener {
            val intent = Intent(this, SymptonsActivity::class.java)
            startActivity(intent)
        }

        val btnPrevention: Button = findViewById(R.id.btnPrevention)
        btnPrevention.setOnClickListener {
            val intent = Intent(this, Prevention::class.java)
            startActivity(intent)
        }

        val btnProducts: Button = findViewById(R.id.btnProducts)
        btnProducts.setOnClickListener {
            val intent = Intent(this, ProductsActivity::class.java)
            startActivity(intent)
        }

        val btnAdd: Button = findViewById(R.id.btnAdd)
        btnAdd.setOnClickListener {
            val intent = Intent(this, AddActivity::class.java)
            startActivity(intent)
        }
    }
}