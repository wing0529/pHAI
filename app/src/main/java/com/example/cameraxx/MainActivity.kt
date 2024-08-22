package com.example.cameraxx

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.cameraxx.databinding.ActivityMainBinding
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

import androidx.camera.view.PreviewView

import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.TextView
import com.example.cameraxx.network.RetrofitClient

import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.ByteArrayOutputStream
import java.io.FileOutputStream


typealias LumaListener = (luma: Double) -> Unit


class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    private var imageCapture: ImageCapture? = null
    private var isContinuousCapturing = false

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    private lateinit var cameraExecutor: ExecutorService

    private val handler = Handler(Looper.getMainLooper())
    private lateinit var timerTextView: TextView
    private var countdownTime = 10
    private var photoCount = 0 // 연속 촬영에서 찍힌 사진의 개수를 저장하는 변수



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(R.layout.activity_main)

        timerTextView = findViewById(R.id.timerTextView)
        viewBinding.uploadStatusTextView.text = "Ready to upload" // TextView 초기화



        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listeners for take photo and video capture buttons
        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }
        viewBinding.continuousCaptureButton.setOnClickListener { toggleContinuousCapture() }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxyToBitmap(imageProxy)
                    imageProxy.close()
                    uploadImage(bitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                }
            }
        )
    }
    private fun uploadImage(bitmap: Bitmap) {
        // Bitmap을 File로 변환
        val imageFile = File(filesDir, "photo.jpg")
        val outputStream = FileOutputStream(imageFile)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
        outputStream.flush()
        outputStream.close()

        // 서버로 파일 업로드
        val requestBody = RequestBody.create("image/jpeg".toMediaTypeOrNull(), imageFile)
        val body = MultipartBody.Part.createFormData("file", imageFile.name, requestBody)

        val call = RetrofitClient.apiService.uploadImage(body)
        viewBinding.uploadStatusTextView.text = "Uploading..."
        call.enqueue(object : Callback<String> {
            override fun onResponse(call: Call<String>, response: Response<String>) {
                if (response.isSuccessful) {
                    val resultText = "Upload successful!"
                    Toast.makeText(this@MainActivity, resultText, Toast.LENGTH_LONG).show()
                    viewBinding.uploadStatusTextView.text = resultText
                } else {
                    val errorText = "Upload failed: ${response.errorBody()?.string()}"
                    Toast.makeText(this@MainActivity, errorText, Toast.LENGTH_LONG).show()
                    viewBinding.uploadStatusTextView.text = errorText
                }
            }

            override fun onFailure(call: Call<String>, t: Throwable) {
                val errorMessage = "Upload error: ${t.message}"
                Toast.makeText(this@MainActivity, errorMessage, Toast.LENGTH_LONG).show()
                viewBinding.uploadStatusTextView.text = errorMessage
            }
        })
    }



    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    private fun getRealPathFromURI(uri: Uri): String {
        var result = ""
        val cursor = contentResolver.query(uri, null, null, null, null)
        if (cursor != null && cursor.moveToFirst()) {
            val index = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA)
            result = cursor.getString(index)
            cursor.close()
        }
        return result
    }

    private fun toggleContinuousCapture() {
        isContinuousCapturing = !isContinuousCapturing
        if (isContinuousCapturing) {
            countdownTime = 10
            photoCount = 0 // 카운트 초기화
            updateTimer()
            Toast.makeText(this, "Continuous Capture Started", Toast.LENGTH_SHORT).show()
            handler.postDelayed({startContinuousCapture()}, 10000)
        } else {
            Toast.makeText(this, "Continuous Capture Stopped", Toast.LENGTH_SHORT).show()
            handler.removeCallbacksAndMessages(null)
        }
    }

    private fun updateTimer() {
        if (countdownTime > 0) {
            timerTextView.text = countdownTime.toString()
            countdownTime--
            handler.postDelayed({ updateTimer() }, 1000)
        } else {
            timerTextView.text = ""
        }
    }

    private fun captureVideo() {}

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    val viewFinder = null
                    it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder().build()


            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)

            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }
    private fun startContinuousCapture() {
        if (isContinuousCapturing && photoCount < 10) { // 10장 이하일 때만 촬영
            takePhoto()
            photoCount++
            handler.postDelayed({ startContinuousCapture() }, 500)
        } else {
            isContinuousCapturing = false
            Toast.makeText(this, "Continuous Capture Completed", Toast.LENGTH_SHORT).show()
        }
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraXApp"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}