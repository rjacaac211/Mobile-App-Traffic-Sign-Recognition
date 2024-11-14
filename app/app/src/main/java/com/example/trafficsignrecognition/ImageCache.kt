package com.example.trafficsignrecognition

import android.graphics.Bitmap
import android.util.LruCache

object ImageCache {
    private val cache: LruCache<String, Bitmap> by lazy {
        val maxMemory = (Runtime.getRuntime().maxMemory() / 1024).toInt()
        val cacheSize = maxMemory / 8
        LruCache<String, Bitmap>(cacheSize)
    }

    fun get(key: String): Bitmap? {
        return cache.get(key)
    }

    fun put(key: String, bitmap: Bitmap) {
        cache.put(key, bitmap)
    }
}
