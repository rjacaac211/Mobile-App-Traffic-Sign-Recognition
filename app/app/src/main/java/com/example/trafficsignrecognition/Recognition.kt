package com.example.trafficsignrecognition

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

class Recognition(
    private val context: Context,
    private val detectionModel: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var detectionInterpreter: Interpreter? = null
    private var classificationInterpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        setupInterpreters()
        loadLabels()
    }

    private fun setupInterpreters() {
        // Create CompatibilityList to check if GPU is supported
        val compatList = CompatibilityList()

        // Set options for the detection interpreter with GPU if available
        val detectionOptions = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                // Enable GPU delegate for detection if supported
                val delegateOptions = compatList.bestOptionsForThisDevice
                addDelegate(GpuDelegate(delegateOptions))
            } else {
                numThreads = calculateOptimalThreads() + 2
            }
        }
        val classificationOptions = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                // Enable GPU delegate for classification if supported
                val delegateOptions = compatList.bestOptionsForThisDevice
                addDelegate(GpuDelegate(delegateOptions))
            } else {
                numThreads = calculateOptimalThreads()
            }
        }

        detectionInterpreter = Interpreter(loadModelFile(context, detectionModel), detectionOptions)
        classificationInterpreter = Interpreter(loadModelFile(context, Constants.CLASSIFICATION_MODEL_PATH), classificationOptions)

        // Get input/output shapes
        val inputShape = detectionInterpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape = detectionInterpreter?.getOutputTensor(0)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]
    }

    private fun calculateOptimalThreads(): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        return when {
            cores > 8 -> 4 // For high-end devices, use 4 threads
            cores > 4 -> 3 // For mid-range devices, use 3 threads
            else -> 2    // For lower-end devices, use 2 threads
        }
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun loadLabels() {
        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line.isNotEmpty()) {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        detectionInterpreter?.close()
        classificationInterpreter?.close()
    }

    fun detect(frame: Bitmap) {
        detectionInterpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1 , numChannel, numElements), OUTPUT_IMAGE_TYPE)
        detectionInterpreter?.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        // Sort the detected bounding boxes by their confidence scores in descending order
        val sortedBoxes = bestBoxes.sortedByDescending { it.cnf }

        // Loop through each bounding box to classify the signs
        sortedBoxes.forEach { boundingBox ->
            // Crop the detected region from the original frame
            val croppedBitmap = cropBitmap(frame, boundingBox)

            // Resize the cropped bitmap to 30x30 for the classification model
            val resizedCroppedBitmap = Bitmap.createScaledBitmap(
                croppedBitmap,
                30, // Target width
                30, // Target height
                true // Filter for better quality
            )

            // Classify the resized image
            val (classificationResult, maxConfidenceIndex) = classifySign(resizedCroppedBitmap)

            // Update the bounding box with the classification result
            boundingBox.clsName = classificationResult
            boundingBox.cls = maxConfidenceIndex
        }

        // Calculate the inference time after detection and classification are done
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        // Use the sorted list of bounding boxes to notify the DetectorListener
        detectorListener.onDetect(sortedBoxes, inferenceTime)
    }

    private fun classifySign(croppedBitmap: Bitmap): Pair<String, Int> {
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(Bitmap.createScaledBitmap(croppedBitmap, 30, 30, true))
        val processedImage = imageProcessor.process(tensorImage)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 30, 30, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        val outputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, numberOfClasses), DataType.FLOAT32)
        classificationInterpreter?.run(inputFeature0.buffer, outputFeature0.buffer)

        val confidenceScores = outputFeature0.floatArray
        val maxConfidenceIndex = confidenceScores.indices.maxByOrNull { confidenceScores[it] } ?: -1

        return labels[maxConfidenceIndex] to maxConfidenceIndex
    }

    private fun cropBitmap(bitmap: Bitmap, boundingBox: BoundingBox): Bitmap {
        val x = (boundingBox.x1 * bitmap.width).toInt()
        val y = (boundingBox.y1 * bitmap.height).toInt()
        val width = ((boundingBox.x2 - boundingBox.x1) * bitmap.width).toInt()
        val height = ((boundingBox.y2 - boundingBox.y1) * bitmap.height).toInt()

        return Bitmap.createBitmap(bitmap, x, y, width, height)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()
        for (c in 0 until numElements) {
            val confidences = (4 until numChannel).map { array[c + numElements * it] }
            val cnf = confidences.max()
            if (cnf > TSD_CONFIDENCE_THRESHOLD) {
                val cls = confidences.indexOf(cnf)
                val clsName = labels[cls]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = cnf, cls = cls, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val TSD_CONFIDENCE_THRESHOLD = 0.25F
        private const val IOU_THRESHOLD = 0.25F
        private const val numberOfClasses = 15
    }
}
