using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;

public class Utils : MonoBehaviour
{
    /// <summary>
    /// Get the dimensions for an image Texture as a Vector2Int
    /// </summary>
    /// <param name="image">An image Texture</param>
    /// <returns></returns>
    public static Vector2Int GetImageDims(Texture image)
    {
        return new Vector2Int(image.width, image.height);
    }

    /// <summary>
    /// Get the texture for an in-game screen object
    /// </summary>
    /// <param name="screen">An in-scene screen object</param>
    /// <returns></returns>
    public static Texture GetScreenTexture(Transform screen)
    {
        return screen.gameObject.GetComponent<MeshRenderer>().material.mainTexture;
    }

    /// <summary>
    /// Resize and position an in-scene screen object
    /// </summary>
    /// <param name="screen">An in-scene screen object</param>
    /// <param name="newDims">The new dimensions for the screen object</param>
    public static void InitializeScreen(Transform screen, Vector2Int newDims)
    {
        // Adjust the VideoScreen dimensions for the new videoTexture
        screen.localScale = new Vector3(newDims.x, newDims.y, screen.localScale.z);
        // Adjust the VideoScreen position for the new videoTexture
        screen.position = new Vector3(newDims.x / 2, newDims.y / 2, 1);
    }

    /// <summary>
    /// Resize and position the main camera based on an in-scene screen object
    /// </summary>
    /// <param name="screenDims">The dimensions of an in-scene screen object</param>
    public static void InitializeCamera(Vector2Int screenDims)
    {
        // Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(screenDims.x / 2, screenDims.y / 2, -10f);
        // Render objects with no perspective (i.e. 2D)
        mainCamera.GetComponent<Camera>().orthographic = true;
        // Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = screenDims.y / 2;
    }


    /// <summary>
    /// Initialize an interface to execute the specified model using the specified backend
    /// </summary>
    /// <param name="model">The target model representation</param>
    /// <param name="workerType">The target compute backend</param>
    /// <param name="useNCHW">EXPERIMENTAL: The channel order for the compute backend</param>
    /// <returns></returns>
    public static IWorker InitializeWorker(Model model, WorkerFactory.Type workerType, bool useNCHW=true)
    {
        // Validate the selected worker type
        workerType = WorkerFactory.ValidateType(workerType);
                
        // Set the channel order of the compute backend to channel-first
        if (useNCHW) ComputeInfo.channelsOrder = ComputeInfo.ChannelsOrder.NCHW;

        // Create a worker to execute the model using the selected backend
        return WorkerFactory.CreateWorker(workerType, model);
    }


    /// <summary>
    /// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image">The target image RenderTexture</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    public static void ProcessImageGPU(RenderTexture image, ComputeShader computeShader, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = computeShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the InputImage variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "InputImage", image);

        // Execute the ComputeShader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, image);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }


    /// <summary>
    /// Apply a segmentation mask to an image on the GPU
    /// </summary>
    /// <param name="mask">The segmentation mask</param>
    /// <param name="source">The source image</param>
    /// <param name="maskWeight">The mask weight for blending the segmentation mask and source image</param>
    /// <param name="computeShader">The target ComputerShader</param>
    /// <param name="functionName">The target ComputeShader function</param>
    /// <returns></returns>
    public static void MaskImageGPU(RenderTexture mask, RenderTexture source, float maskWeight, ComputeShader computeShader, string functionName)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the specified function in the ComputeShader
        int kernelHandle = computeShader.FindKernel(functionName);
        // Define a temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(mask.width, mask.height, 24, mask.format);
        // Enable random write access
        result.enableRandomWrite = true;
        // Create the HDR RenderTexture
        result.Create();

        // Set the value for the Result variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "Result", result);
        // Set the value for the MaskTexture variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "MaskTexture", mask);
        // Set the value for the SourceImage variable in the ComputeShader
        computeShader.SetTexture(kernelHandle, "SourceImage", source);
        // Set the value for the maskWeight variable in the ComputeShader
        computeShader.SetFloat("maskWeight", maskWeight);

        // Execute the ComputeShader
        computeShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

        // Copy the result into the source RenderTexture
        Graphics.Blit(result, mask);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);
    }


    /// <summary>
    /// Scale the source image resolution to the target input dimensions
    /// while maintaing the source aspect ratio.
    /// </summary>
    /// <param name="imageDims"></param>
    /// <param name="targetDims"></param>
    /// <returns></returns>
    public static Vector2Int CalculateInputDims(Vector2Int imageDims, int targetDim)
    {
        // Clamp the minimum dimension value to 64px
        targetDim = Mathf.Max(targetDim, 64);

        Vector2Int inputDims = new Vector2Int();

        // Calculate the input dimensions using the target minimum dimension
        if (imageDims.x >= imageDims.y)
        {
            inputDims[0] = (int)(imageDims.x / ((float)imageDims.y / (float)targetDim));
            inputDims[1] = targetDim;
        }
        else
        {
            inputDims[0] = targetDim;
            inputDims[1] = (int)(imageDims.y / ((float)imageDims.x / (float)targetDim));
        }

        return inputDims;
    }
}
