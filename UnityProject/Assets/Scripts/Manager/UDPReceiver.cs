using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using System.Net;
using static ViTPose;
using System.Runtime.CompilerServices;
using System.Diagnostics;

public class UDPReceiver : MonoBehaviour
{
    Thread keypoints_Thread;
    UdpClient keypoints_client;
    public int keypoints_port = 5252;
    public bool keypoints_startReceiving = true;
    public string keypoints_data;
    public Keypoints keypoints;
    private object keypointsLock = new object();
    public bool keypoints_dataReceived = false;

    Thread image_Thread;
    UdpClient image_client;
    public int image_port = 5253;
    public bool image_startReceiving = true;
    public bool image_dataReceived = false;
    public byte[] image_data;

    // Start is called before the first frame update
    public void Start()
    {

        keypoints_Thread = new Thread(new ThreadStart(ReceiveKeypoints));
        keypoints_Thread.IsBackground= true;
        keypoints_Thread.Start();

        image_Thread = new Thread(new ThreadStart(ReceiveImage));
        image_Thread.IsBackground = true;
        image_Thread.Start();
    }

    private void ReceiveKeypoints()
    {
        try
        {
            keypoints_client = new UdpClient(keypoints_port);
            while (keypoints_startReceiving)
            {
                try
                {
                    IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                    byte[] dataByte = keypoints_client.Receive(ref anyIP);
                    keypoints_data = Encoding.UTF8.GetString(dataByte);
                    Keypoints receivedKeypoints = JsonUtility.FromJson<Keypoints>(keypoints_data);
                    if (keypoints != null && keypoints.objects != null)
                    {
                        lock (keypointsLock)
                        {
                            keypoints = receivedKeypoints;
                        }
                        keypoints_dataReceived = true;
                    }
                }
                catch (SocketException ex)
                {
                    // 소켓이 닫혔을 때 발생하는 정상적인 예외
                    if (!keypoints_startReceiving)
                    {
                        UnityEngine.Debug.Log("[UDPReceiver] Keypoints 수신 스레드가 정상적으로 종료되었습니다.");
                        break;
                    }
                    else
                    {
                        UnityEngine.Debug.LogError($"[UDPReceiver] Keypoints 소켓 오류: {ex.Message}");
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            UnityEngine.Debug.LogError($"[UDPReceiver] Keypoints 스레드 오류: {ex.Message}");
        }
        finally
        {
            if (keypoints_client != null)
            {
                keypoints_client.Close();
                keypoints_client = null;
            }
        }
    }
    private void ReceiveImage()
    {
        try
        {
            image_client = new UdpClient(image_port);
            while (image_startReceiving)
            {
                try
                {
                    IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                    byte[] dataByte = image_client.Receive(ref anyIP);
                    image_data = dataByte;
                    image_dataReceived = true;
                }
                catch (SocketException ex)
                {
                    // 소켓이 닫혔을 때 발생하는 정상적인 예외
                    if (!image_startReceiving)
                    {
                        UnityEngine.Debug.Log("[UDPReceiver] Image 수신 스레드가 정상적으로 종료되었습니다.");
                        break;
                    }
                    else
                    {
                        UnityEngine.Debug.LogError($"[UDPReceiver] Image 소켓 오류: {ex.Message}");
                    }
                }
            }
        }
        catch (System.Exception ex)
        {
            UnityEngine.Debug.LogError($"[UDPReceiver] Image 스레드 오류: {ex.Message}");
        }
        finally
        {
            if (image_client != null)
            {
                image_client.Close();
                image_client = null;
            }
        }
    }
    private void Scale(Keypoints keypoints)
    {
        int scale_factor = 5;
        foreach (var obj in keypoints.objects)
        {
            foreach (var position in obj.GetAllPositions())
            {
                for (int i = 0; i < position.Length; i++)
                {
                    position[i] *= scale_factor;
                }
            }
        }
    }

    private void OnDestroy()
    {
        StopUDPReceiver();
    }

    private void OnApplicationQuit()
    {
        StopUDPReceiver();
    }

    private void OnDisable()
    {
        StopUDPReceiver();
    }

    private void StopUDPReceiver()
    {
        // 수신 중지
        keypoints_startReceiving = false;
        image_startReceiving = false;

        // 클라이언트 정리
        if (keypoints_client != null)
        {
            keypoints_client.Close();
            keypoints_client.Dispose();
            keypoints_client = null;
        }

        if (image_client != null)
        {
            image_client.Close();
            image_client.Dispose();
            image_client = null;
        }

        // 스레드 정리
        if (keypoints_Thread != null && keypoints_Thread.IsAlive)
        {
            keypoints_Thread.Join(1000); // 최대 1초 대기
            if (keypoints_Thread.IsAlive)
            {
                keypoints_Thread.Abort(); // 강제 종료 (최후의 수단)
            }
            keypoints_Thread = null;
        }

        if (image_Thread != null && image_Thread.IsAlive)
        {
            image_Thread.Join(1000); // 최대 1초 대기
            if (image_Thread.IsAlive)
            {
                image_Thread.Abort(); // 강제 종료 (최후의 수단)
            }
            image_Thread = null;
        }

        UnityEngine.Debug.Log("[UDPReceiver] UDP 연결 및 스레드가 정리되었습니다.");
    }

}
