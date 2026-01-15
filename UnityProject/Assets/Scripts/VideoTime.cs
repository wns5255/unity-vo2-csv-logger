using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using TMPro;

public class VideoTime : MonoBehaviour
{

    public VideoPlayer videoPlayer; // VideoPlayer 컴포넌트
    public TMP_Text timeText;

    void Start()
    {
        // VideoPlayer가 재생을 시작할 때 남은 시간 업데이트 시작
        videoPlayer.started += OnVideoStarted;
    }

    void Update()
    {
        // 동영상이 재생 중일 때만 남은 시간을 계산하여 업데이트
        if (videoPlayer.isPlaying)
        {
            UpdateRemainingTime();
        }
    }

    void OnVideoStarted(VideoPlayer vp)
    {
        UpdateRemainingTime();
    }

    void UpdateRemainingTime()
    {
        // 전체 시간에서 현재 재생 시간을 빼서 남은 시간을 계산
        double remainingTime = videoPlayer.length - videoPlayer.time;

        // 시간을 분:초 형식으로 변환해서 표시
        int seconds = Mathf.FloorToInt((float)remainingTime);

        timeText.text = "남은 시간: " + seconds.ToString();
    }
}
