package com.example.cameraxx

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.widget.TextView
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsControllerCompat
import io.reactivex.android.schedulers.AndroidSchedulers
import io.reactivex.disposables.CompositeDisposable
import io.reactivex.disposables.Disposable

class MainActivity2 : AppCompatActivity() {
    private lateinit var resultTextView: TextView
    private val compositeDisposable = CompositeDisposable()
    private val connectionHelper = ConnectionHelper()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultTextView = findViewById(R.id.resultTextView)

        // Edge-to-edge UI 설정
        val windowInsetsController = WindowInsetsControllerCompat(window, findViewById(R.id.main))
        windowInsetsController.isAppearanceLightStatusBars = true

        // 패딩 설정
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { view, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        fetchDatabaseData()
    }

    private fun fetchDatabaseData() {
        val disposable: Disposable = connectionHelper.connectToDatabase()
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe({ result ->
                resultTextView.text = result
            }, { throwable ->
                resultTextView.text = "Error: ${throwable.message}"
            })

        compositeDisposable.add(disposable)
    }

    override fun onDestroy() {
        super.onDestroy()
        compositeDisposable.clear()
    }
}
