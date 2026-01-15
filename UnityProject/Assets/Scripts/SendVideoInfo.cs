using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Video;

public class SendVideoInfo : MonoBehaviour
{
    public VideoPlayer videoPlayer;

    void Start()
    {
        // 이제 PlayerPrefs에서 exerciseId와 Type을 가져옵니다.
        string exerciseId = PlayerPrefs.GetString("exerciseId", "");
        string type = PlayerPrefs.GetString("Type", "");

        if (!string.IsNullOrEmpty(exerciseId) && !string.IsNullOrEmpty(type))
        {
            // 1. Intro 영상을 먼저 찾아봅니다.
            string introPath = $"Video/{type}/{exerciseId}_intro";
            VideoClip videoClip = Resources.Load<VideoClip>(introPath);

            // 2. Intro 영상이 없으면 Left 영상을 찾아봅니다.
            if (videoClip == null)
            {
                string leftPath = $"Video/{type}/{exerciseId}_left";
                videoClip = Resources.Load<VideoClip>(leftPath);

                if (videoClip == null)
                {
                    UnityEngine.Debug.LogError($"[SendVideoInfo] Intro와 Left 영상을 모두 찾을 수 없음: Resources/Video/{type}/{exerciseId}_left");
                    return;
                }
            }
            
            // 비디오 클립 설정
            videoPlayer.clip = videoClip;
            videoPlayer.Prepare();
            videoPlayer.prepareCompleted += DisplayFirstFrame;
        }
        else
        {
            UnityEngine.Debug.LogError("[SendVideoInfo] PlayerPrefs에서 exerciseId 또는 Type을 찾을 수 없습니다!");
        }
    }

    private void DisplayFirstFrame(VideoPlayer vp)
    {
        // 첫 번째 프레임으로 이동 후 일시정지
        videoPlayer.frame = 0;
        videoPlayer.Pause();
    }
}
