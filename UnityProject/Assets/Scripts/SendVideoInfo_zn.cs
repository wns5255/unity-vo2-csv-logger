using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public class SendVideoInfo_zn : MonoBehaviour
{
    public VideoPlayer videoPlayer;

    public VideoController videoController;

    void Start()
    {
        // 이전 씬에서 저장한 videoPath 가져오기 (비디오 파일 이름)
        string videoFileName = PlayerPrefs.GetString("exerciseName", "");

        if (!string.IsNullOrEmpty(videoFileName))
        {
            // Resources 폴더에 있는 비디오 파일 로드
            string videoPath = "Video/" + videoFileName; // Video는 Resources/Video 폴더를 가리킴

            // 비디오 파일 로드 및 재생
            VideoClip videoClip = Resources.Load<VideoClip>(videoPath);
            if (videoClip != null)
            {
                videoPlayer.clip = videoClip;
                videoPlayer.Prepare();
                videoPlayer.prepareCompleted += DisplayFirstFrame;
            }
            else
            {
                Debug.LogError("Video not found in Resources/Video: " + videoFileName);
            }
        }
        else
        {
            Debug.LogError("Video file name is not found!");
        }
    }

    private void DisplayFirstFrame(VideoPlayer vp)
    {
        // 첫 프레임으로 이동
        videoPlayer.frame = 0;

        // 비디오 일시정지
        //videoPlayer.Pause();
        videoController.click_pause();
    }
}
