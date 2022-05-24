using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class CamVidSegmentation : MonoBehaviour
{
    [Header("Scene Objects")]
    [Tooltip("The Screen object for the scene")]
    public Transform screen;

    [Header("Data Processing")]
    [Tooltip("The target model input dimensions")]
    public Vector2Int inputDims = new Vector2Int(320, 240);
    [Tooltip("The compute shader for GPU processing")]
    public ComputeShader processingShader;
    [Tooltip("The material with the fragment shader for GPU processing")]
    public Material processingMaterial;

    [Header("Barracuda")]
    [Tooltip("The Barracuda/ONNX asset file")]
    public NNModel modelAsset;
    [Tooltip("The name for the custom softmax output layer")]
    public string argmaxLayer = "argmaxLayer";
    [Tooltip("The model execution backend")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;
    [Tooltip("The target output layer index")]
    public int outputLayerIndex = 0;
    [Tooltip("EXPERIMENTAL: Indicate whether to order tensor data channels first")]
    public bool useNCHW = true;

    [Header("Output Processing")]
    [Tooltip("The compute shader for applying a segmentation mask on the GPU")]
    public ComputeShader maskShader;
    [Tooltip("Apply the predicted segmentation mask to the source image")]
    public bool maskImage = true;
    [Tooltip("The mask weight for blending the segmentation mask and source image")]
    [Range(0f, 1f)]
    public float maskWeight = 1f;

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
    // The predicted segmentation mask
    private RenderTexture maskTexture;
    // The masked output image
    private RenderTexture outputImage;

    // The current frame rate value
    private int fps = 0;
    // Controls when the frame rate value updates
    private float fpsTimer = 0f;

    // The ordered list of class names
    private string[] classes = new string[] {
        "Animal", 
        "Archway", 
        "Bicyclist", 
        "Bridge", 
        "Building", 
        "Car",
        "CartLuggagePram", 
        "Child", 
        "Column_Pole", 
        "Fence", 
        "LaneMkgsDriv",
        "LaneMkgsNonDriv", 
        "Misc_Text", 
        "MotorcycleScooter", 
        "OtherMoving",
        "ParkingBlock", 
        "Pedestrian", 
        "Road", 
        "RoadShoulder", 
        "Sidewalk",
        "SignSymbol", 
        "Sky", 
        "SUVPickupTruck", 
        "TrafficCone",
        "TrafficLight", 
        "Train", 
        "Tree", 
        "Truck_Bus", 
        "Tunnel",
        "VegetationMisc", 
        "Void", 
        "Wall"
    };

    // Start is called before the first frame update
    void Start()
    {
        // Initialize the texture which stores the model input
        inputTexture = new RenderTexture(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);
        // Initialize the texture which stores the predicted segmentation mask
        maskTexture = new RenderTexture(inputDims.x, inputDims.y, 24, RenderTextureFormat.ARGBHalf);

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
        // Add a new ArgMax layer
        modelBuilder.Reduce(Layer.Type.ArgMax, argmaxLayer, outputLayer);

        // Initialize the interface for executing the model
        engine = Utils.InitializeWorker(modelBuilder.model, workerType, useNCHW);

        // Initialize the texture which stores the masked output image
        outputImage = new RenderTexture(imageTexture.width, imageTexture.height, 24, RenderTextureFormat.ARGBHalf);
    }

    

    /// <summary>
    /// Process the raw model output to get the predicted class index
    /// </summary>
    /// <param name="engine">The interface for executing the model</param>
    /// <returns></returns>
    private void ProcessOutput(IWorker engine)
    {
        // Get raw model output
        Tensor output = engine.PeekOutput(argmaxLayer);
        if (printDebugMessages) Debug.Log(output.shape);

        // Copy model output to a RenderTexture
        output.ToRenderTexture(maskTexture);

        // Copy the mask texture into the higher resolution output texture
        Graphics.Blit(maskTexture, outputImage);
        // Copy the source image texture into a temporary RenderTexture
        RenderTexture sourceImage = RenderTexture.GetTemporary(imageTexture.width, imageTexture.height, 24, RenderTextureFormat.ARGBHalf);
        Graphics.Blit(imageTexture, sourceImage);
        // Apply the predicted mask color values to the image
        Utils.MaskImageGPU(outputImage, sourceImage, maskWeight, maskShader, "CamVidMaskImage");
        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(sourceImage);
        // Update the texture for the screen object
        screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture = maskImage ? outputImage : imageTexture;
        
        // Dispose Tensor and associated memories.
        output.Dispose();
    }

    
    // Update is called once per frame
    void Update()
    {
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

        // Get the predicted label colors
        ProcessOutput(engine);
    }


    // OnGUI is called for rendering and handling GUI events.
    public void OnGUI()
    {
        if (!printDebugMessages) return;

        GUIStyle style = new GUIStyle();
        style.fontSize = (int)(Screen.width * (1f / (100f - fontScale)));
        style.normal.textColor = textColor;

        if (Time.unscaledTime > fpsTimer)
        {
            fps = (int)(1f / Time.unscaledDeltaTime);
            fpsTimer = Time.unscaledTime + fpsRefreshRate;
        }

        Rect fpsRect = new Rect(10, 10, 500, 500);
        GUI.Label(fpsRect, new GUIContent($"FPS: {fps}"), style);
    }


    // OnDisable is called when the MonoBehavior becomes disabled
    private void OnDisable()
    {
        Destroy(inputTexture);
        Destroy(outputImage);

        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}
