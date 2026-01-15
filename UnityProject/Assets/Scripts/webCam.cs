using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class webCam : MonoBehaviour
{
    private WebCamTexture webCamTexture;  // 웹캠 텍스처

    void Awake()
    {
        // 웹캠 텍스처 초기화
        webCamTexture = new WebCamTexture(1920, 1080);

        Renderer renderer = GetComponent<Renderer>();
        renderer.material.mainTexture = webCamTexture;

        // 웹캠 텍스처 재생
        webCamTexture.Play();
    }

    // 종료 시 웹캠 텍스처 중지
    void OnDisable()
    {
        if (webCamTexture != null && webCamTexture.isPlaying)
        {
            webCamTexture.Stop();
        }
    }

}
