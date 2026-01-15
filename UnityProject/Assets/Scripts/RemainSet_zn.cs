using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using TMPro;

public class RemainSet_zn : MonoBehaviour
{
    public VideoPlayer videoPlayer; // VideoPlayer 컴포넌트
    public TMP_Text setText;            // 현재 세트 진행 상황을 표시할 Text UI
    private int setCount;           // 선택한 세트 수
    private int currentSet = 1;     // 현재 몇 번째 세트인지 추적

    public ChangeScene sceneChange;

    public VideoController videoController;

    void Start()
    {
        // PlayerPrefs에서 저장된 세트 수 불러오기
        setCount = PlayerPrefs.GetInt("SetCount", 1);  // 기본값은 1

        // 세트 수에 따라 비디오 반복
        videoPlayer.loopPointReached += OnVideoEnd;  // 비디오 재생이 끝날 때 이벤트 등록
        StartNextSet();  // 첫 번째 세트 시작
    }

    private void StartNextSet()
    {
        if (currentSet <= setCount)
        {
            // 현재 몇 번째 세트인지와 총 세트 수 표시
            setText.text = currentSet.ToString() + " / " + setCount.ToString() + " 세트";

            // 비디오 재생 시작
            videoPlayer.Play();
            //videoController.click_start();
        }
        else
        {
            // 모든 세트가 끝났을 때
            Debug.Log("All sets complete");
        }
    }

    private void OnVideoEnd(VideoPlayer vp)
    {
        // 비디오가 끝나면 다음 세트로 넘어감
        currentSet++;

        // 세트가 남아 있으면 다음 세트를 시작
        if (currentSet <= setCount)
        {
            StartNextSet();
        }
        else
        {
            // 모든 세트가 완료되었을 때
            setText.text = "Completed!";

            sceneChange.toResult(); //결과 창으로 전환

            Debug.Log("All sets finished");
        }
    }

    public void RetryCurrentSet()
    {
        // 비디오를 처음부터 다시 재생
        //videoPlayer.Stop();
        videoPlayer.frame = 0;
        //videoPlayer.Pause();
        videoController.click_pause();
        //videoPlayer.Play();
        videoController.click_start();

        // 현재 세트 정보를 다시 표시 (현재 세트는 그대로 유지)
        setText.text = currentSet.ToString() + "/" + setCount.ToString() + " 세트";

        Debug.Log("Retrying set " + currentSet);
    }
}