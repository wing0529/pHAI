

import com.example.cameraxx.network.PredictionResponse
import com.google.gson.JsonObject
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {
    @GET("/result")
    fun getPrediction(): Call<PredictionResponse>

    @Multipart
    @POST("/upload/multiple")
    fun uploadMultipleImages(
        @Part file: List<MultipartBody.Part>
    ): Call<JsonObject> // Return type should be ResponseBody or a specific type you expect

    @GET("/file/ready") // The endpoint to check if the file is ready
    fun isFileReady(): Call<Void> // or a specific response type if needed
}
