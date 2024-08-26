package com.example.cameraxx

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.cameraxx.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import com.google.gson.JsonObject

class resultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.result)

        fetchPredictionResult()

        val btnRetake: Button = findViewById(R.id.btnRetake)
        btnRetake.setOnClickListener {
            val intent = Intent(this, ExplainActivity::class.java)
            startActivity(intent)
        }
        val btnHome: Button = findViewById(R.id.btnHome)
        btnHome.setOnClickListener {
            val intent = Intent(this, Main::class.java)
            startActivity(intent)
        }
    }

    private fun fetchPredictionResult() {
        val tvStatus: TextView = findViewById(R.id.tvStatus)

        val call = RetrofitClient.apiService.getPrediction()
        call.enqueue(object : Callback<JsonObject> {
            override fun onResponse(call: Call<JsonObject>, response: Response<JsonObject>) {
                if (response.isSuccessful) {
                    val result = response.body()
                    tvStatus.text = result?.get("prediction")?.asString ?: "No prediction found"
                } else {
                    tvStatus.text = "Error: ${response.code()}"
                }
            }

            override fun onFailure(call: Call<JsonObject>, t: Throwable) {
                tvStatus.text = "Failed to get result: ${t.message}"
            }
        })
    }
}
