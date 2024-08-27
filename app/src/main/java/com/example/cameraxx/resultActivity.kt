package com.example.cameraxx

import ApiService
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.cameraxx.R
import com.example.cameraxx.network.PredictionResponse
import com.example.cameraxx.network.RetrofitClient
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class resultActivity : AppCompatActivity() {

    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.result)

        resultTextView = findViewById(R.id.tvStatus)
        progressBar = findViewById(R.id.ProgressBar)

        fetchResult()

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

    private fun fetchResult() {
        val apiService = RetrofitClient.apiService
        val maxRetries = 100
        var retryCount = 0

        val handler = Handler(Looper.getMainLooper())

        // 명명된 Runnable 클래스 정의
        val retryRunnable = RetryRunnable(apiService, handler, maxRetries)

        // 첫 번째 호출
        handler.post(retryRunnable)
    }

    private fun updateProgressBar(result: String?) {
        Log.d("ResultActivity", "Updating ProgressBar with result: $result")

        progressBar.isIndeterminate = false

        when (result) {
            "danger" -> {
                progressBar.progress = 90
                progressBar.progressTintList =
                    ContextCompat.getColorStateList(this, android.R.color.holo_red_dark)
            }
            "safe" -> {
                progressBar.progress = 30
                progressBar.progressTintList =
                    ContextCompat.getColorStateList(this, R.color.safe_green)
            }
            "error" -> {
                progressBar.progress = 100
                progressBar.progressTintList =
                    ContextCompat.getColorStateList(this, android.R.color.darker_gray)
            }
            "warning" -> {
                progressBar.progress = 60
                progressBar.progressTintList =
                    ContextCompat.getColorStateList(this, android.R.color.holo_orange_light)
            }
            else -> {
                progressBar.progress = 0
                progressBar.progressTintList =
                    ContextCompat.getColorStateList(this, android.R.color.darker_gray)
            }
        }
    }

    // 명명된 Runnable 클래스를 정의
    private inner class RetryRunnable(
        private val apiService: ApiService,
        private val handler: Handler,
        private val maxRetries: Int
    ) : Runnable {
        private var retryCount = 0

        override fun run() {
            if (retryCount < maxRetries) {
                val call = apiService.getPrediction()

                call.enqueue(object : Callback<PredictionResponse> {
                    override fun onResponse(
                        call: Call<PredictionResponse>,
                        response: Response<PredictionResponse>
                    ) {
                        if (response.isSuccessful) {
                            val prediction = response.body()?.prediction
                            Log.d("ResultActivity", "Received result: $prediction")
                            resultTextView.text = prediction
                            updateProgressBar(prediction)
                        } else {
                            retryCount++
                            Log.d("ResultActivity", "Retrying... ($retryCount/$maxRetries)")
                            /*  Toast.makeText(
                                  this@ResultActivity,
                                  "Retrying... ($retryCount/$maxRetries)",
                                  Toast.LENGTH_SHORT
                              ).show() */
                            handler.postDelayed(this@RetryRunnable, 5000) // 5초 후에 다시 시도
                        }
                    }

                    override fun onFailure(call: Call<PredictionResponse>, t: Throwable) {
                        retryCount++
                        Log.e("ResultActivity", "Retrying due to failure: ${t.message} ($retryCount/$maxRetries)")
                        handler.postDelayed(this@RetryRunnable, 5000) // 5초 후에 다시 시도
                    }
                })
            } else {
                Toast.makeText(this@resultActivity, "Failed to get result after $maxRetries retries.", Toast.LENGTH_LONG).show()
            }
        }
    }
}
