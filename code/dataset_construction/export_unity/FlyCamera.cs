using UnityEngine;

public class FreeFlyCamera : MonoBehaviour
{
    public float moveSpeed = 2f;
    public float lookSpeed = 2f;

    private float yaw = 0f;
    private float pitch = 0f;

    void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
    }

    void Update()
    {
        // Mouse look
        yaw += lookSpeed * Input.GetAxis("Mouse X");
        pitch -= lookSpeed * Input.GetAxis("Mouse Y");
        pitch = Mathf.Clamp(pitch, -90f, 90f);
        transform.eulerAngles = new Vector3(pitch, yaw, 0f);

        // Movement
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        float y = 0f;
        if (Input.GetKey(KeyCode.E)) y += 1f;
        if (Input.GetKey(KeyCode.Q)) y -= 1f;

        Vector3 forward = transform.forward;
        forward.y = 0f; 
        forward.Normalize();

        Vector3 move = transform.right * x + forward * z + transform.up * y;
        transform.Translate(move * moveSpeed * Time.deltaTime, Space.World);
    }
}
