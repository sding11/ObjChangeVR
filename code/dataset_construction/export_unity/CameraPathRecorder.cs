using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class CameraPathRecorder : MonoBehaviour
{
    public Transform cameraTransform;               
    public float recordInterval = 0.2f;             
    public string saveFileName = "camera_path.json"; 

    private float timer = 0f;
    private List<CameraPose> path = new List<CameraPose>(); 
    private string outputFilePath;                  
    private string lastPosePath;                    

    void OnEnable()
    {
        timer = 0f;
        path.Clear();

        string baseDir = Directory.GetCurrentDirectory();
        string folderPath = Path.Combine(baseDir, "record", "path");
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
            Debug.Log("[Recorder] Created directory: " + folderPath);
        }

        outputFilePath = Path.Combine(folderPath, saveFileName);
        lastPosePath = Path.Combine(baseDir, "last_camera_pose.json");

        if (File.Exists(lastPosePath))
        {
            string json = File.ReadAllText(lastPosePath);
            CameraPose lastPose = JsonUtility.FromJson<CameraPose>(json);
            cameraTransform.position = new Vector3(lastPose.px, lastPose.py, lastPose.pz);
            cameraTransform.rotation = new Quaternion(lastPose.rx, lastPose.ry, lastPose.rz, lastPose.rw);
            Debug.Log("[Recorder] Restored camera position from last_camera_pose.json");
        }

        Debug.Log("[Recorder] Recording started. Will save to: " + outputFilePath);
    }

    void Update()
    {
        timer += Time.deltaTime;
        if (timer >= recordInterval)
        {
            timer = 0f;

            // Record the current position and rotation of the camera
            CameraPose pose = new CameraPose
            {
                px = cameraTransform.position.x,
                py = cameraTransform.position.y,
                pz = cameraTransform.position.z,
                rx = cameraTransform.rotation.x,
                ry = cameraTransform.rotation.y,
                rz = cameraTransform.rotation.z,
                rw = cameraTransform.rotation.w
            };
            path.Add(pose);

            Debug.Log($"[Recorder] Recorded point #{path.Count - 1}");
            SavePathToFile();
        }
    }

    void OnDisable()
    {
        SavePathToFile();
        SaveLastPose();
        Debug.Log("[Recorder] Recording stopped and path saved to: " + outputFilePath);
        Debug.Log("[Recorder] Total recorded points: " + path.Count);
    }

    void SavePathToFile()
    {
        if (string.IsNullOrEmpty(outputFilePath)) return;
        string json = JsonUtility.ToJson(new CameraPathWrapper { path = path }, true);
        File.WriteAllText(outputFilePath, json);
    }

    void SaveLastPose()
    {
        if (path.Count == 0) return;
        CameraPose last = path[path.Count - 1];
        string json = JsonUtility.ToJson(last, true);
        File.WriteAllText(lastPosePath, json);
        Debug.Log("[Recorder] Last camera position saved to: " + lastPosePath);
    }

    // Serializable structure to store a single camera pose
    [System.Serializable]
    public class CameraPose
    {
        public float px, py, pz;      // Position (x, y, z)
        public float rx, ry, rz, rw;  // Rotation (Quaternion x, y, z, w)
    }

    [System.Serializable]
    public class CameraPathWrapper
    {
        public List<CameraPose> path;
    }
}
