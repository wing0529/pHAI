package com.example.cameraxx.network

import okhttp3.MultipartBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {
    @Multipart
    @POST("/upload")  // Flask 서버에서 정의한 업로드 경로
    fun uploadImage(@Part file: MultipartBody.Part): Call<String>
}



