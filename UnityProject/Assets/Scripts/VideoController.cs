using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using TMPro;

public class VideoController : MonoBehaviour
{
    public VideoPlayer video;
    private int videoState; //영상 state 지정 변수

    //public GameObject webCam;
    public RawImage raw;
    private WebCamTexture webCamTexture;

    public TMP_Text countdownText;

    void Awake()
    {
        // 웹캠 텍스처 초기화
        webCamTexture = new WebCamTexture(1920, 1080);

        //Renderer renderer = webCam.GetComponent<Renderer>();

        raw.texture = webCamTexture;
        raw.material.mainTexture = webCamTexture;
        //renderer.material.mainTexture = webCamTexture;

        // 웹캠 텍스처 재생
        webCamTexture.Play();
    }

    //
    void OnEnable()
    {
        //비디오 끝날 때 호출되는 이벤트 연결
        video.loopPointReached += OnVideoEnd;
    }

    void OnDisable()
    {
        //이벤트 연결 해제
        video.loopPointReached -= OnVideoEnd;
    }

    public void click_start()
    {
        //start 버튼 클릭 시 호출
        //video.Play(); //비디오 재생
        //videoState = 0; //재생 후 state 0으로 전달
        //SendState(videoState); //전달하는 함수

        StartCoroutine(StartVideoWithDelay());
    }

    IEnumerator StartVideoWithDelay()
    {
        int countdown = 3; // 카운트다운 시작값

        // 3초 동안 대기
        while (countdown > 0)
        {
            countdownText.text = countdown.ToString();
            yield return new WaitForSeconds(1f); // 1초 대기
            countdown--; // 카운트다운 감소
        }

        video.Play(); // 비디오 재생

        // 상태 업데이트 및 전달
        videoState = 0;
        SendState(videoState); // 전달하는 함수 호출

        // 비디오가 시작되면 텍스트를 지우거나 필요에 맞게 업데이트
        //yield return new WaitForSeconds(0.5f); // 잠시 대기 후 텍스트 지우기
        countdownText.text = ""; // 텍스트 지움
    }


    //pause 버튼 클릭 시 호출
    public void click_pause()
    {
        video.Pause();
        videoState = 1;
        SendState(videoState);
    }

    //video 종료되면 호출
    private void OnVideoEnd(VideoPlayer vp) //여러 videoPlayer를 사용한다면, vp 인자로 구분. 지금은 단일 videoPlayer라 사용 안 함
    {
        videoState = 2;  // 종료 상태 전송
        SendState(videoState);  // 상태 전송 함수 호출
    }

    private void SendState(int state)
    {
        //state 전송하는 코드 작성

        //videoState == 0 비디오 재생 중
        //videoState == 1 비디오 일시 정지
        //videoState == 2 비디오 종료

        Debug.Log("Video State: " + state); //log에서 체크용
    }

}
