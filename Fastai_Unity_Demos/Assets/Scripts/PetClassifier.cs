using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.Rendering;

public class PetClassifier : MonoBehaviour
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
    [Tooltip("The name for the custom softmax output layer")]
    public string softmaxLayer = "softmaxLayer";
    [Tooltip("The name for the custom softmax output layer")]
    public string argmaxLayer = "argmaxLayer";
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;

    [Header("Output Processing")]
    [Tooltip("Asynchronously download model output from the GPU to the CPU.")]
    public bool useAsyncGPUReadback = true;

    [Header("Debugging")]
    [Tooltip("Print debugging messages to the console")]
    public bool printDebugMessages = true;
    [Tooltip("The on-screen text color")]
    public Color textColor = Color.red;
    [Tooltip("The scale value for the on-screen font size")]
    [Range(0, 100)]
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
    
    // Stores the predicted class index
    private int classIndex;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;

    // The ordered list of class names
    private string[] classes = new string[] { 
        "Abyssinian", 
        "Bengal", 
        "Birman", 
        "Bombay", 
        "British_Shorthair", 
        "Egyptian_Mau", 
        "Maine_Coon", 
        "Persian", 
        "Ragdoll", 
        "Russian_Blue", 
        "Siamese", 
        "Sphynx", 
        "american_bulldog", 
        "american_pit_bull_terrier", 
        "basset_hound", 
        "beagle", 
        "boxer", 
        "chihuahua", 
        "english_cocker_spaniel", 
        "english_setter", 
        "german_shorthaired", 
        "great_pyrenees", 
        "havanese", 
        "japanese_chin", 
        "keeshond", 
        "leonberger", 
        "miniature_pinscher", 
        "newfoundland", 
        "pomeranian", 
        "pug", 
        "saint_bernard", 
        "samoyed", 
        "scottish_terrier", 
        "shiba_inu", 
        "staffordshire_bull_terrier", 
        "wheaten_terrier", 
        "yorkshire_terrier" 
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
        modelBuilder.Softmax(softmaxLayer, outputLayer);
        // Add a new Argmax layer
        modelBuilder.Reduce(Layer.Type.ArgMax, argmaxLayer, softmaxLayer);
        // Initialize the interface for executing the model
        engine = Utils.InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Initialize the GPU output texture
        outputTextureGPU = RenderTexture.GetTemporary(1, 1, 24, RenderTextureFormat.ARGBHalf);
        // Initialize the CPU output texture
        outputTextureCPU = new Texture2D(1, 1, TextureFormat.RGBAHalf, false);
        
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
    int ProcessOutput(IWorker engine)
    {
        int classIndex = -1;

        // Get raw model output
        Tensor output = engine.PeekOutput(argmaxLayer);

        if (useAsyncGPUReadback)
        {
            // Copy model output to a RenderTexture
            output.ToRenderTexture(outputTextureGPU);
            // Asynchronously download model output from the GPU to the CPU
            AsyncGPUReadback.Request(outputTextureGPU, 0, TextureFormat.RGBAHalf, OnCompleteReadback);
            // Get the predicted class index
            classIndex = (int)outputTextureCPU.GetPixel(0, 0).r;

            if (classIndex == -23)
            {
                Debug.Log("Output texture still needs to initialize, defaulting to classIndex=-1");
                classIndex = -1;
            }
        }
        else
        {
            // Get the predicted class index
            classIndex = (int)output[0];
        }

        if (printDebugMessages) Debug.Log($"Class Index: {classIndex}");

        // Dispose Tensor and associated memories.
        output.Dispose();

        return classIndex;
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
        // Get the predicted class index
        classIndex = ProcessOutput(engine);
        if (classIndex < 0 || classIndex >= classes.Length)
        {
            Debug.Log("Invalid class index");
            return;
        }
        if (printDebugMessages) Debug.Log($"Predicted Class: {classes[classIndex]}");

        // Unload assets when running in a web browser
        if (Application.platform == RuntimePlatform.WebGLPlayer) Resources.UnloadUnusedAssets();
    }


    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        if (!printDebugMessages) return;

        if (classIndex < 0 || classIndex >= classes.Length)
        {
            Debug.Log("Invalid class index");
            return;
        }

        Rect labelRect = new Rect(10, 10, 500, 500);

        GUIStyle style = new GUIStyle();
        style.fontSize = (int)(Screen.width * (1f / (100f - fontScale)));
        style.normal.textColor = textColor;

        string content = $"Predicted Class: {classes[classIndex]}";
        GUI.Label(labelRect, new GUIContent(content), style);

        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        Rect fpsRect = new Rect(10, style.fontSize * 1.5f, 500, 500);
        GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }

    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        RenderTexture.ReleaseTemporary(outputTextureGPU);

        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
