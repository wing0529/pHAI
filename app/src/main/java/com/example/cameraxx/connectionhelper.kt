package com.example.cameraxx

import android.util.Log
import io.reactivex.Single
import io.reactivex.schedulers.Schedulers
import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet
import java.sql.SQLException
import java.sql.Statement

class ConnectionHelper {
    private val ip = "172.1.1.0"
    private val database = "AndroidSTDB"
    private val uname = "sa"
    private val pass = "test129"
    private val port = "1433"

    fun connectToDatabase(): Single<String> {
        return Single.fromCallable {
            val result = StringBuilder()
            var connection: Connection? = null
            val connectionURL: String

            try {
                Class.forName("net.sourceforge.jtds.jdbc.Driver")
                connectionURL = "jdbc:jtds:sqlserver://$ip:$port;databasename=$database;user=$uname;password=$pass;"
                connection = DriverManager.getConnection(connectionURL)

                if (connection != null) {
                    val query = "SELECT * FROM results" // SQL 쿼리 작성
                    val stmt: Statement = connection.createStatement()
                    val rs: ResultSet = stmt.executeQuery(query)

                    while (rs.next()) {
                        // 데이터를 가져와서 result 문자열에 추가
                        result.append(rs.getString("result_column")).append("\n")
                    }
                    rs.close()
                    stmt.close()
                }
            } catch (se: SQLException) {
                Log.e("SQL Error", se.message ?: "Unknown SQL error")
                return@fromCallable "SQL Error: ${se.message}"
            } catch (e: ClassNotFoundException) {
                Log.e("Class Error", e.message ?: "Unknown class error")
                return@fromCallable "Class Error: ${e.message}"
            } catch (e: Exception) {
                Log.e("Error", e.message ?: "Unknown error")
                return@fromCallable "Error: ${e.message}"
            } finally {
                connection?.close()
            }

            result.toString()
        }.subscribeOn(Schedulers.io())
    }
}
