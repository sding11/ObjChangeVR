using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Text;
using System.Collections.Concurrent; 

#if UNITY_EDITOR
using UnityEditor;
#endif

public class CameraPathPlayer : MonoBehaviour
{
    public Transform cameraTransform;
    public Camera playbackCamera;
    public int numToDisappear = 250;
    public float disappearDuration = 1000f;
    public float triggerAtFraction = 0.6f;
    public string loadFileName = "camera_path.json";
    public float delayBeforeExit = 2f;
    public int captureEveryNFrames = 2;

    private List<CameraPose> path = new List<CameraPose>();
    private int currentIndex = 0;
    private bool isPlaying = false;
    private bool hasFinished = false;
    private float totalPlayTime = 0f;
    private string baseDir;
    private string currentFolder;
    private string beforeFolder;
    private string afterFolder;
    private string csvFilePath;
    private int screenshotIndex = 1;
    private bool switchedToAfter = false;
    private bool hasDisappeared = false;
    private GameObject[] disappearObjects;
    private string recordPath;
    private HashSet<string> previousDisappearNames = new HashSet<string>();
    private int triggerIndex = -1;

    private static readonly object csvOpenLock = new object();
    private FileStream csvFs;
    private StreamWriter csvSw;
    private ConcurrentQueue<string> csvQueue = new ConcurrentQueue<string>();
    private Coroutine csvWriterCo;
    private bool stopCsvWriter = false;


    void Start()
    {
        if (cameraTransform == null || playbackCamera == null)
        {
            Debug.LogError("[Player] Missing camera references.");
            return;
        }

        LoadPathFromFile();
        SetupFolders();
        SetupDisappearObjects();

        if (path.Count > 0)
        {
            triggerIndex = Mathf.FloorToInt(path.Count * triggerAtFraction);
            isPlaying = true;
            currentIndex = 0;
            ApplyPose(path[0]);
            CaptureCurrentScreenshot();
            Debug.Log("[Player] Playback started. Total points: " + path.Count);
        }
        else
        {
            Debug.LogWarning("[Player] No path points loaded.");
            StartCoroutine(ExitAfterDelay());
        }
    }

    void LoadPathFromFile()
    {
        string filePath = Path.Combine(Directory.GetCurrentDirectory(), "record", "path", loadFileName);
        if (!File.Exists(filePath))
        {
            Debug.LogError("[Player] Path file not found: " + filePath);
            return;
        }

        string json = File.ReadAllText(filePath);
        CameraPathWrapper wrapper = JsonUtility.FromJson<CameraPathWrapper>(json);
        path = wrapper?.path ?? new List<CameraPose>();
    }

    void Update()
    {
        if (!isPlaying || currentIndex >= path.Count || hasFinished) return;

        currentIndex++;

        if (currentIndex < path.Count)
        {
            ApplyPose(path[currentIndex]);

            if (currentIndex % captureEveryNFrames == 0)
            {
                CaptureCurrentScreenshot();
            }

            if (!hasDisappeared && currentIndex >= triggerIndex && previousDisappearNames.Count == 0)
            {
                TriggerDisappearance();
                hasDisappeared = true;
            }
        }
        else
        {
            isPlaying = false;
            hasFinished = true;
            SavePlaybackTime();
            StartCoroutine(ExitAfterDelay());
        }
    }

    void ApplyPose(CameraPose pose)
    {
        cameraTransform.position = pose.GetPosition();
        cameraTransform.rotation = pose.GetRotation();
    }

    void SetupFolders()
    {
        baseDir = Directory.GetCurrentDirectory();
        string screenshotPath = Path.Combine(baseDir, "record", "screenshot");
        string groundtruthPath = Path.Combine(baseDir, "record", "groundtruth");
        recordPath = Path.Combine(baseDir, "record", "disappear_object");
        Directory.CreateDirectory(recordPath);

        bool screenshotExists = Directory.Exists(screenshotPath) &&
                                 (Directory.GetFiles(screenshotPath, "*.png").Length > 0 ||
                                  Directory.GetFiles(screenshotPath, "*.csv").Length > 0);

        currentFolder = screenshotExists ? groundtruthPath : screenshotPath;
        beforeFolder = Path.Combine(currentFolder, "before");
        afterFolder = Path.Combine(currentFolder, "after");
        Directory.CreateDirectory(beforeFolder);
        Directory.CreateDirectory(afterFolder);

        csvFilePath = Path.Combine(currentFolder, "data.csv");
        if (!File.Exists(csvFilePath))
        {
            File.WriteAllText(csvFilePath, "Index,Timestamp,ScreenshotFilename,Position,Rotation,IsDisappear\n");
        }

        lock (csvOpenLock)
        {
            if (csvSw == null)
            {
                csvFs = new FileStream(csvFilePath, FileMode.Append, FileAccess.Write, FileShare.Read);
                csvSw = new StreamWriter(csvFs, new UTF8Encoding(false)) { AutoFlush = false }; 
                                                                                                
                csvWriterCo = StartCoroutine(CsvWriterLoop());
            }
        }

    }

    private IEnumerator CsvWriterLoop()
    {
        var sb = new StringBuilder(4096);
        var wait = new WaitForSeconds(0.02f); 
        while (!stopCsvWriter)
        {
            try
            {
                sb.Clear();
                int drained = 0;

                while (csvQueue.TryDequeue(out var line))
                {
                    sb.Append(line);
                    sb.Append('\n');
                    drained++;
                    if (drained >= 256) break; 
                }

                if (drained > 0 && csvSw != null)
                {
                    csvSw.Write(sb.ToString());
                    csvSw.Flush();
                }
            }
            catch (IOException ex)
            {
                Debug.LogWarning("[Player] CSV writer IO exception: " + ex.Message);
            }

            yield return wait;
        }

        try
        {
            var tail = new StringBuilder();
            while (csvQueue.TryDequeue(out var line))
            {
                tail.Append(line);
                tail.Append('\n');
            }
            if (tail.Length > 0 && csvSw != null)
            {
                csvSw.Write(tail.ToString());
                csvSw.Flush();
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[Player] CSV writer final flush exception: " + ex.Message);
        }
    }


    void SetupDisappearObjects()
    {
        disappearObjects = GameObject.FindGameObjectsWithTag("disappear");

        var files = Directory.GetFiles(recordPath, "disappear_*.json");
        if (files.Length > 0)
        {
            string latest = files[0];
            foreach (var file in files)
            {
                if (File.GetLastWriteTime(file) > File.GetLastWriteTime(latest))
                    latest = file;
            }

            string json = File.ReadAllText(latest);
            DisappearRecord record = JsonUtility.FromJson<DisappearRecord>(json);

            foreach (var entry in record.entries)
            {
                GameObject obj = GameObject.Find(entry.objectName);
                if (obj != null)
                {
                    HighlightObjectRed(obj);
                    previousDisappearNames.Add(entry.objectName);
                }
            }

            Debug.Log("[Player] Highlighted previous disappearances in red.");
        }
    }

    void TriggerDisappearance()
    {
        if (disappearObjects == null || disappearObjects.Length == 0) return;

        Debug.Log("[Player] Triggering object disappearance by step count...");
        List<GameObject> all = new List<GameObject>(disappearObjects);
        if (numToDisappear > all.Count) numToDisappear = all.Count;

        DisappearRecord record = new DisappearRecord { entries = new List<DisappearEntry>() };

        for (int i = 0; i < numToDisappear; i++)
        {
            int index = UnityEngine.Random.Range(0, all.Count);
            GameObject obj = all[index];
            all.RemoveAt(index);

            HighlightObjectRed(obj);
            obj.SetActive(false);
            StartCoroutine(ReappearAfterTime(obj, disappearDuration));

            record.entries.Add(new DisappearEntry
            {
                objectName = obj.name,
                disappearTime = disappearDuration
            });
        }

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmm");
        string filePath = Path.Combine(recordPath, $"disappear_{timestamp}.json");
        File.WriteAllText(filePath, JsonUtility.ToJson(record, true));
        Debug.Log("[Player] Disappearance record saved.");
    }

    void HighlightObjectRed(GameObject obj)
    {
        //Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        //foreach (var rend in renderers)
        //{
        //    rend.material.color = Color.red;
        //}
    }

    IEnumerator ReappearAfterTime(GameObject obj, float delay)
    {
        yield return new WaitForSeconds(delay);
        obj.SetActive(true);
    }

    void CaptureCurrentScreenshot()
    {
        int width = Screen.width;
        int height = Screen.height;

        bool isAfterPhase = currentIndex >= triggerIndex;
        string targetFolder = isAfterPhase ? afterFolder : beforeFolder;
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
        string filename = "Screenshot_" + timestamp + ".png";
        string path = Path.Combine(targetFolder, filename);

        RenderTexture rt = new RenderTexture(width, height, 24);
        playbackCamera.targetTexture = rt;
        Texture2D screenShot = new Texture2D(width, height, TextureFormat.RGB24, false);
        playbackCamera.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenShot.Apply();

        playbackCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(path, bytes);

        Vector3 pos = cameraTransform.position;
        Quaternion rot = cameraTransform.rotation;

        string posStr = $"({pos.x:F2},{pos.y:F2},{pos.z:F2})";
        string rotStr = $"({rot.x:F2},{rot.y:F2},{rot.z:F2},{rot.w:F2})";
        int isDisappear = isAfterPhase ? 1 : 0;

        string line = string.Format("{0},{1},{2},\"{3}\",\"{4}\",{5}",
            screenshotIndex, timestamp, filename, posStr, rotStr, isDisappear);
        
        csvQueue.Enqueue(line);
        screenshotIndex++;

        Destroy(screenShot);


        Debug.Log("[Player] Screenshot saved: " + filename);
    }

    private void OnDisable()
    {
        StopAndCloseCsv();
    }

    private void OnApplicationQuit()
    {
        StopAndCloseCsv();
    }

    private void StopAndCloseCsv()
    {
        if (csvWriterCo != null)
        {
            stopCsvWriter = true;
            try { StopCoroutine(csvWriterCo); } catch { /* ignore */ }
            csvWriterCo = null;
        }

        lock (csvOpenLock)
        {
            try { csvSw?.Flush(); } catch { }
            try { csvSw?.Dispose(); } catch { }
            try { csvFs?.Dispose(); } catch { }
            csvSw = null;
            csvFs = null;
        }
    }


    void SavePlaybackTime()
    {
        string path = Path.Combine(baseDir, "record", "path", "playback_time.json");
        int minutes = (int)(totalPlayTime / 60);
        int seconds = (int)(totalPlayTime % 60);
        var data = new PlaybackTimeData { minutes = minutes, seconds = seconds, totalSeconds = (int)totalPlayTime };
        File.WriteAllText(path, JsonUtility.ToJson(data, true));
    }

    IEnumerator ExitAfterDelay()
    {
        Debug.Log("[Player] Will exit after delay...");
        yield return new WaitForSeconds(delayBeforeExit);
#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    [Serializable]
    public class CameraPose
    {
        public float px, py, pz;
        public float rx, ry, rz, rw;
        public Vector3 GetPosition() => new Vector3(px, py, pz);
        public Quaternion GetRotation() => new Quaternion(rx, ry, rz, rw);
    }

    [Serializable]
    public class CameraPathWrapper
    {
        public List<CameraPose> path;
    }

    [Serializable]
    public class PlaybackTimeData
    {
        public int minutes;
        public int seconds;
        public int totalSeconds;
    }

    [Serializable]
    public class DisappearEntry
    {
        public string objectName;
        public float disappearTime;
    }

    [Serializable]
    public class DisappearRecord
    {
        public List<DisappearEntry> entries;
    }
}
