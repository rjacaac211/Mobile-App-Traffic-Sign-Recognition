package com.example.trafficsignrecognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.view.View
import android.widget.ImageButton
import android.widget.ImageView
import android.os.Handler
import android.os.Looper
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AlertDialog
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.trafficsignrecognition.Constants.LABELS_PATH
import com.example.trafficsignrecognition.Constants.DETECTION_MODEL_PATH
import com.example.trafficsignrecognition.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

class MainActivity : AppCompatActivity(), Recognition.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private val isFrontCamera = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var recognition: Recognition

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var signImage1: ImageView
    private lateinit var signImage2: ImageView
    private lateinit var signImage3: ImageView

    private lateinit var tts: TextToSpeech
    private var isSpeaking = false
    private val recentSigns = mutableListOf<String>()
    private val handler = Handler(Looper.getMainLooper())
    private val signQueue = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val helpButton = findViewById<ImageButton>(R.id.helpButton)
        helpButton.setOnClickListener {
            showHelpDialog()
        }

        signImage1 = findViewById(R.id.signImage1)
        signImage2 = findViewById(R.id.signImage2)
        signImage3 = findViewById(R.id.signImage3)

        recognition = Recognition(baseContext, DETECTION_MODEL_PATH, LABELS_PATH, this)

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                if (result != TextToSpeech.LANG_MISSING_DATA && result != TextToSpeech.LANG_NOT_SUPPORTED) {
                    tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {}

                        override fun onDone(utteranceId: String?) {
                            isSpeaking = false
                            speakNextSign() // Speak the next sign after the previous one is done
                        }

                        override fun onError(utteranceId: String?) {
                            isSpeaking = false
                            speakNextSign() // Attempt to speak the next sign even if there's an error
                        }
                    })
                } else {
                    Log.e("TTS", "Language not supported")
                }
            } else {
                Log.e("TTS", "Initialization failed")
            }
        }
    }

    private fun speakNextSign() {
        if (signQueue.isNotEmpty() && !isSpeaking) {
            val nextSign = signQueue.removeAt(0)
            handler.postDelayed({
                speakOut(nextSign)
            }, 200) // 200 milliseconds
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            // Resize the rotated bitmap to 640x640 for the detection model
            val resizedBitmap = Bitmap.createScaledBitmap(
                rotatedBitmap,
                640, // Target width
                640, // Target height
                true
            )

            recognition.detect(resizedBitmap)
        }

        cameraProvider.unbindAll()
        camera = cameraProvider.bindToLifecycle(
            this,
            cameraSelector,
            preview,
            imageAnalyzer
        )
        preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        if (this::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
        recognition.clear()
        cameraExecutor.shutdown()
        super.onDestroy()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            clearSignImages()
            binding.overlay.invalidate()
        }
    }

    // Method to clear the ImageViews when no signs are detected
    private fun clearSignImages() {
        val signImageViews = listOf(binding.signImage1, binding.signImage2, binding.signImage3)
        signImageViews.forEach { imageView ->
            imageView?.setImageDrawable(null)
            imageView?.visibility = View.INVISIBLE
        }
    }

    private fun speakOut(text: String) {
        isSpeaking = true
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "utteranceId")
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        val topThreeBoxes = boundingBoxes.sortedByDescending { it.cnf }.take(3)

        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }
            updateSignImages(topThreeBoxes)

            topThreeBoxes.forEach { box ->
                if (!recentSigns.contains(box.clsName)) {
                    recentSigns.add(box.clsName)
                    signQueue.add(box.clsName)

                    // Remove this sign from the recent list after a delay
                    handler.postDelayed({
                        recentSigns.remove(box.clsName)
                    }, 5000) //  5 seconds
                }
            }

            // Start speaking the next sign if not already speaking
            if (!isSpeaking) {
                speakNextSign()
            }
        }
    }

    private fun updateSignImages(topThreeBoxes: List<BoundingBox>) {
        val signImageViews = listOf(signImage1, signImage2, signImage3)

        topThreeBoxes.forEachIndexed { index, boundingBox ->
            val imageName = "sign${boundingBox.cls + 1}"
            val cacheKey = imageName
            var cachedBitmap = ImageCache.get(cacheKey)

            if (cachedBitmap == null) {
                // If the image is not in cache, load it
                val imageResId = resources.getIdentifier(imageName, "drawable", packageName)
                if (imageResId != 0) {
                    cachedBitmap = BitmapFactory.decodeResource(resources, imageResId)
                    ImageCache.put(cacheKey, cachedBitmap)
                }
            }

            if (cachedBitmap != null) {
                signImageViews[index].apply {
                    setImageBitmap(cachedBitmap)
                    visibility = View.VISIBLE
                }
            } else {
                signImageViews[index].visibility = View.GONE
            }
        }
    }

    private fun showHelpDialog() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.help_dialog_title))
            .setMessage(getString(R.string.help_message))
            .setPositiveButton(getString(R.string.ok)) { dialog, _ -> dialog.dismiss() }
            .create()
            .show()
    }
}
