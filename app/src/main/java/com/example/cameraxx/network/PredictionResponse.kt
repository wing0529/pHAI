package com.example.cameraxx.network

import com.google.gson.annotations.SerializedName

data class PredictionResponse(
    @SerializedName("prediction") val prediction: String
)
