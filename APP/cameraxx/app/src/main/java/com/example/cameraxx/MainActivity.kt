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
import com.google.gson.JsonObject

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

    private val capturedBitmaps = mutableListOf<Bitmap>() // 촬영된 비트맵을 저장할 리스트

    private lateinit var photoCountTextView: TextView



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        timerTextView = findViewById(R.id.timerTextView)

        photoCountTextView = findViewById(R.id.photoCountTextView) // 추가된 텍스트뷰 초기화


        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        // Set up the listeners for take photo and video capture buttons

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

    /*private fun takePhoto() {
        // Get a stable reference of the modifiable image capture use case
        val imageCapture = imageCapture ?: return

        // Create time stamped name and MediaStore entry.
       val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if(Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CameraX-Image")
            }
        }

        // Create output options object which contains file + metadata
        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues)
            .build()

        // Set up image capture listener, which is triggered after photo has
        // been taken
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                }

                override fun
                        onImageSaved(output: ImageCapture.OutputFileResults){
                    val msg = "Photo capture succeeded: ${output.savedUri}"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d(TAG, msg)
                }
            }
        )
    }*/
    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxyToBitmap(imageProxy)
                    imageProxy.close()
                    capturedBitmaps.add(bitmap) // 비트맵 리스트에 추가
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                }
            }
        )
    }

    private fun processCapturedBitmaps() {
        // 비트맵을 파일로 변환 및 파일 리스트에 추가
        val fileList = mutableListOf<File>()
        for (i in capturedBitmaps.indices) {
            val bitmap = capturedBitmaps[i]
            val fileName = java.lang.String.format("photo%02d.jpg", i + 1)
            val imageFile = File(filesDir, fileName)
            val outputStream = FileOutputStream(imageFile)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            outputStream.flush()
            outputStream.close()

            fileList.add(imageFile)
        }

        // 파일들을 서버로 업로드
        uploadImages(fileList)
        capturedBitmaps.clear() // 비트맵 리스트 초기화 (필요한 경우)
    }

    private fun uploadImages(files: MutableList<File>) {

        // 파일들을 MultipartBody.Part로 변환
        val multipartBodyList = files.map { file ->
            val requestBody = RequestBody.create("image/jpeg".toMediaTypeOrNull(), file)
            MultipartBody.Part.createFormData("files", file.name, requestBody)
        }.toMutableList()// map 결과를 MutableList로 변환


        val call = RetrofitClient.apiService.uploadMultipleImages(multipartBodyList)
        call.enqueue(object : Callback<JsonObject> {
            override fun onResponse(call: Call<JsonObject>, response: Response<JsonObject>) {
                if (response.isSuccessful) {
                    val jsonResponse = response.body()
                    val resultText = jsonResponse?.get("success")?.asString ?: "Upload successful!"

                    // 파일 초기화
                    multipartBodyList.clear()  // 이 부분은 필요 없을 수 있음 (로컬에서 초기화 목적이라면)
                    files.clear()  // 파일 리스트 초기화

                    // ResultActivity 시작
                    val intent = Intent(this@MainActivity, resultActivity::class.java)
                    intent.putExtra("RESULT_TEXT", resultText)
                    startActivity(intent)

                    Toast.makeText(this@MainActivity, resultText, Toast.LENGTH_LONG).show()


                } else {
                    val errorText = "Upload failed: ${response.errorBody()?.string()}"
                    Toast.makeText(this@MainActivity, errorText, Toast.LENGTH_LONG).show()

                }
            }

            override fun onFailure(call: Call<JsonObject>, t: Throwable) {
                val errorMessage = "Upload error: ${t.message}"
                Toast.makeText(this@MainActivity, errorMessage, Toast.LENGTH_LONG).show()

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
            Toast.makeText(this, "타이머 시작", Toast.LENGTH_SHORT).show()
            handler.postDelayed({startContinuousCapture()}, 10000)
        } else {
            Toast.makeText(this, "타이머 중지", Toast.LENGTH_SHORT).show()
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
            photoCountTextView.text = "현재 촬영된 사진: $photoCount 장" // 사진 개수 업데이트
            handler.postDelayed({ startContinuousCapture() }, 500)
        } else {
            isContinuousCapturing = false
            Toast.makeText(this, "사진촬영이 완료되었습니다. \n잠시만 기다려주세요.", Toast.LENGTH_SHORT).show()

            // 10장이 모두 촬영된 후 비트맵 변환 및 처리
            processCapturedBitmaps()

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