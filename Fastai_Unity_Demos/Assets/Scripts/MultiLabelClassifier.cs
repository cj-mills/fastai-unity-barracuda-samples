using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;

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

    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The name for the custom softmax output layer")]
    public string sigmoidLayer = "sigmoidLayer";
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

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;

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
        // Add a new Softmax layer
        modelBuilder.Sigmoid(sigmoidLayer, outputLayer);

        // Initialize the interface for executing the model
        engine = Utils.InitializeWorker(modelBuilder.model, workerType, useNCHW);
    }

    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    List<Tuple<int, float>> ProcessOutput(IWorker engine)
    {
        // Get raw model output
        Tensor output = engine.PeekOutput(sigmoidLayer);
        List<Tuple<int, float>> labelIndices = new List<Tuple<int, float>>();

        for (int i=0; i<output.length; i++)
        {
            if(output[i] >= confidenceThreshold)
            {
                labelIndices.Add(new Tuple<int, float>(i, output[i]));
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
        // Normalize the input pixel data
        Utils.ProcessImageGPU(inputTexture, processingShader, "NormalizeImageNet");

        // Initialize a Tensor using the inputTexture
        input = new Tensor(inputTexture, channels: 3);
        // Execute the model with the input Tensor
        engine.Execute(input);
        // Dispose Tensor and associated memories.
        input.Dispose();

        // Release the input texture
        RenderTexture.ReleaseTemporary(inputTexture);
        // Get the predicted class index
        List<Tuple<int, float>> labelIndices = ProcessOutput(engine);

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


    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
