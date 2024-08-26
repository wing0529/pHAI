package com.example.cameraxx

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import androidx.appcompat.app.AppCompatActivity
import com.example.cameraxx.databinding.ActivityMainBinding
import android.view.LayoutInflater
import android.widget.TextView
import androidx.appcompat.app.AlertDialog

class ExplainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)

        setContentView(R.layout.method_explain)

        val backButton: ImageButton = findViewById(R.id.backButton)
        backButton.setOnClickListener {
            val intent = Intent(this, Main::class.java)
            startActivity(intent)
        }
        val startButton: Button = findViewById(R.id.startButton)
        startButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }
    }
    private fun showCustomPopup() {
        // 커스텀 레이아웃을 인플레이트
        val dialogView = LayoutInflater.from(this).inflate(R.layout.popup, null)

        // 다이얼로그 생성
        val dialog = AlertDialog.Builder(this)
            .setView(dialogView)
            .create()

        // 다이얼로그 뷰에서 컴포넌트 찾기
        val titleTextView = dialogView.findViewById<TextView>(R.id.dialogTitle)
        val messageTextView = dialogView.findViewById<TextView>(R.id.dialogMessage)
        val button = dialogView.findViewById<Button>(R.id.dialogButton)

        // 다이얼로그 제목과 메시지 설정
        titleTextView.text = "측정 방법 설명"
        messageTextView.text = "여기에서 측정 방법을 설명합니다."

        // 버튼 클릭 리스너 설정
        button.setOnClickListener {
            dialog.dismiss() // 다이얼로그 닫기
        }

        // 다이얼로그 표시
        dialog.show()
    }
}