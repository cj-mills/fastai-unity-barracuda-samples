using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using UnityEngine.Rendering;

public class MultiLabelClassifier : MonoBehaviour
{
    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;

    [Header("Data Processing")]
    [Tooltip("The target minimum model input dimensions")]
    public int targetDim = 224;
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("The material with the fragment shader for GPU processing")]
    public Material processingMaterial;

    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The name for the custom sigmoid activation layer")]
    public string sigmoidLayer = "sigmoidLayer";
    [Tooltip("The name for the custom reshape output layer")]
    public string reshapeLayer = "reshapeLayer";
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;

    [Header("Output Processing")]
    [Tooltip("The minimum confidence score to keep a predicted label")]
    [Range(0f, 1.0f)]
    public float confidenceThreshold = 0.2f;
    [Tooltip("Asynchronously download model output from the GPU to the CPU.")]
    public bool useAsyncGPUReadback = true;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0,100)]
    public int fontScale = 50;
    [Tooltip("The number of seconds to wait between refreshing the fps value")]
    [Range(0.01f, 1.0f)]
    public float fpsRefreshRate = 0.1f;

    // The neural net model data structure
    private Model m_RunTimeModel;
    // The main interface to execute models
    private IWorker engine;
    // The name of the model output layer
    private string outputLayer;
    // Stores the input data for the model
    private Tensor input;

    // The source image texture
    private Texture imageTexture;
    // The model input texture
    private RenderTexture inputTexture;
    // The source image dimensions
    private Vector2Int imageDims;

    // Stores the raw model output on the GPU when using useAsyncGPUReadback
    private RenderTexture outputTextureGPU;
    // Stores the raw model output on the CPU when using useAsyncGPUReadback
    private Texture2D outputTextureCPU;

    // Stores the predicted label indices and confidence values
    private List<Tuple<int, float>> labelIndices;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;

    // The ordered list of class names
    private string[] classes = new string[] {
        "aeroplane", 
        "bicycle", 
        "bird", 
        "boat", 
        "bottle", 
        "bus", 
        "car", 
        "cat", 
        "chair", 
        "cow", 
        "diningtable", 
        "dog", 
        "horse", 
        "motorbike", 
        "person", 
        "pottedplant", 
        "sheep", 
        "sofa", 
        "train", 
        "tvmonitor"
    };


    // Start is called before the first frame update
    void Start()
    {
        // Get the source image texture
        imageTexture = Utils.GetScreenTexture(screen);
        // Get the source image dimensions as a Vector2Int
        imageDims = Utils.GetImageDims(imageTexture);
        // Resize and position the screen object using the source image dimensions
        Utils.InitializeScreen(screen, imageDims);
        // Resize and position the main camera using the source image dimensions
        Utils.InitializeCamera(imageDims);
        // Get an object oriented representation of the model
        m_RunTimeModel = ModelLoader.Load(modelAsset);
        // Get the name of the target output layer
        outputLayer = m_RunTimeModel.outputs[outputLayerIndex];

        // Create a model builder to modify the m_RunTimeModel
        ModelBuilder modelBuilder = new ModelBuilder(m_RunTimeModel);
        // Add a new sigmoid layer
        modelBuilder.Sigmoid(sigmoidLayer, outputLayer);

        // Add a new Argmax layer
        modelBuilder.Reshape(reshapeLayer, sigmoidLayer, new Int32[] { 1, 1, classes.Length, 1 }, rank: 4);

        // Initialize the interface for executing the model
        engine = Utils.InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Initialize the GPU output texture
        outputTextureGPU = RenderTexture.GetTemporary(classes.Length, 1, 24, RenderTextureFormat.ARGBHalf);
        // Initialize the CPU output texture
        outputTextureCPU = new Texture2D(classes.Length, 1, TextureFormat.RGBAHalf, false);
    }

    /// <summary>
    /// Called once AsyncGPUReadback has been completed
    /// </summary>
    /// <param name="request"></param>
    void OnCompleteReadback(AsyncGPUReadbackRequest request)
    {
        if (request.hasError)
        {
            Debug.Log("GPU readback error detected.");
            return;
        }

        // Make sure the Texture2D is not null
        if (outputTextureCPU)
        {
            // Fill Texture2D with raw data from the AsyncGPUReadbackRequest
            outputTextureCPU.LoadRawTextureData(request.GetData<uint>());
            // Apply changes to Textur2D
            outputTextureCPU.Apply();
        }
    }

    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    List<Tuple<int, float>> ProcessOutput(IWorker engine)
    {
        // Get raw model output
        Tensor output = engine.PeekOutput(reshapeLayer);
        if (printDebugMessages) Debug.Log(output.shape);
        List<Tuple<int, float>> labelIndices = new List<Tuple<int, float>>();


        if (useAsyncGPUReadback)
        {
            // Copy model output to a RenderTexture
            output.ToRenderTexture(outputTextureGPU);
            // Asynchronously download model output from the GPU to the CPU
            AsyncGPUReadback.Request(outputTextureGPU, 0, TextureFormat.RGBAHalf, OnCompleteReadback);

            for (int i = 0; i < output.length; i++)
            {
                // Process model output
                Color confidenceValue = outputTextureCPU.GetPixel(i, 0);
                
                if (confidenceValue.r >= confidenceThreshold)
                {
                    labelIndices.Add(new Tuple<int, float>(i, confidenceValue.r));
                }
            }            
        }
        else
        {
            // Process model output
            for (int i = 0; i < output.length; i++)
            {
                if (output[i] >= confidenceThreshold)
                {
                    labelIndices.Add(new Tuple<int, float>(i, output[i]));
                }
            }
        }

        // Dispose Tensor and associated memories.
        output.Dispose();

        return labelIndices;
    }

    
    // Update is called once per frame
    void Update()
    {
        // Scale the source image resolution
        Vector2Int inputDims = Utils.CalculateInputDims(imageDims, targetDim);
        if (printDebugMessages) Debug.Log($"Input Dims: {inputDims.x} x {inputDims.y}");

        // Initialize the input texture with the calculated input dimensions
        inputTexture = RenderTexture.GetTemporary(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);

        // Copy the source image texture into model input texture
        Graphics.Blit(imageTexture, inputTexture);

        if (SystemInfo.supportsComputeShaders)
        {
            // Normalize the input pixel data
            Utils.ProcessImageGPU(inputTexture, processingShader, "NormalizeImageNet");
            // Initialize a Tensor using the inputTexture
            input = new Tensor(inputTexture, channels: 3);
        }
        else
        {
            // Disable asynchronous GPU readback when not using Compute Shaders
            useAsyncGPUReadback = false;

            // Define a temporary HDR RenderTexture
            RenderTexture result = RenderTexture.GetTemporary(inputTexture.width,
                inputTexture.height, 24, RenderTextureFormat.ARGBHalf);
            RenderTexture.active = result;

            // Apply preprocessing steps
            Graphics.Blit(inputTexture, result, processingMaterial);

            // Initialize a Tensor using the inputTexture
            input = new Tensor(result, channels: 3);
            RenderTexture.ReleaseTemporary(result);
        }

        // Execute the model with the input Tensor
        engine.Execute(input);
        // Dispose Tensor and associated memories.
        input.Dispose();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTexture);
        // Get the predicted label indices
        labelIndices = ProcessOutput(engine);

        if (printDebugMessages)
        {
            string message = "Predicted Labels: ";
            foreach (Tuple<int, float> labelIndex in labelIndices)
            {
                string confidence = (labelIndex.Item2 * 100).ToString("0.00");
                message += $"{classes[labelIndex.Item1]} ({confidence})%   ";
            }
            Debug.Log(message);
        }
    }

    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        if (!printDebugMessages) return;

        Rect contentRect = new Rect(10, 10, 500, 500);

        GUIStyle style = new GUIStyle();
        style.fontSize = (int)(Screen.width * (1f / (100f - fontScale)));
        style.normal.textColor = textColor;

        string content = "Predicted Labels: ";
        foreach (Tuple<int, float> labelIndex in labelIndices)
        {
            string confidence = (labelIndex.Item2 * 100).ToString("0.00");
            content += $"{classes[labelIndex.Item1]} ({confidence})%   ";
        }
        GUI.Label(contentRect, new GUIContent(content), style);


        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        Rect fpsRect = new Rect(10, style.fontSize*1.5f, 500, 500);
        GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }

    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
