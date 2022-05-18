using Unity.Barracuda;
using UnityEngine;

public class HeadPoseEstimator : MonoBehaviour
{
    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;

    [Header("Data Processing")]
    [Tooltip("The target model input dimensions")]
    public Vector2Int inputDims = new Vector2Int(320, 240);
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;

    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;
    
    [Header("Pose Visualization")]
    [Tooltip("The radius for the pose coordinate dot")]
    public float dotRadius = 10f;
    [Tooltip("The color for the pose estimation dot")]
    public Color dotColor = Color.red;

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
    
    // The predicted head pose coordinates
    private Vector2 coords;
    // The base texture for the pose estimation dot
    private Texture2D dotTexture;


    // Start is called before the first frame update
    void Start()
    {
        // Initialize the texture which stores the model input
        inputTexture = new RenderTexture(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);

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
        // Initialize the interface for executing the model
        engine = Utils.InitializeWorker(m_RunTimeModel, workerType, useNCHW);

        // Initialize the texture for the head pose visualization
        dotTexture = Texture2D.whiteTexture;
    }

    /// <summary>
    /// Process the raw model output to get the predicted head position
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    Vector2 ProcessOutput(IWorker engine)
    {
        // Get raw model output
        Tensor output = engine.PeekOutput(outputLayer);
        // Initialize vector for coordinates
        Vector2 coords = new Vector2();

        // Process model output
        for (int i=0; i<output.length; i++)
        {
            coords[i] = ((output[i] + 1) / 2) * inputDims[i] * (imageDims[i] / inputDims[i]);
        }
        if (this.printDebugMessages) Debug.Log($"Predicted Coords: {coords}");

        // Dispose Tensor and associated memories.
        output.Dispose();

        return coords;
    }


    // Update is called once per frame
    void Update()
    {
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
        // Get the predicted head position
        coords = ProcessOutput(engine);
    }

    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        // Get the smallest dimension of the target display
        float minDimension = Mathf.Min(Screen.width, Screen.height);

        // The value used to scale the head pose coordinates up to the source resolution
        float scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

        // Get the smallest horizontal resolution between the target displaysource image
        float maxX = Mathf.Max(Screen.width, imageDims.x);
        // Get the largest horizontal resolution between the target displaysource image
        float minX = Mathf.Min(Screen.width, imageDims.x);

        // Scale the predicted head position to the target display
        float dot_x = (maxX - minX * scale) / 2 + coords.x * scale;
        float dot_y = coords.y * scale;

        // Specify the outer bounds for the dot
        Rect boxRect = new Rect(x: dot_x, y: dot_y, width: dotRadius, height: dotRadius);

        // Draw the dot at the predicted head position
        GUI.DrawTexture(
            position: boxRect, 
            image: dotTexture, 
            scaleMode: ScaleMode.StretchToFill, 
            alphaBlend: true, 
            imageAspect: 0, 
            color: dotColor, 
            borderWidth: 0, 
            borderRadius: dotRadius
            );
    }

    // OnDisable is called when the behaviour becomes disabled
    private void OnDisable()
    {
        Destroy(inputTexture);

        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
