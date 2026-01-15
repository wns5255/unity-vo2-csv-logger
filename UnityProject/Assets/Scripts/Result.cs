using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Timeline;

public class Result : MonoBehaviour
{
    [Header("UI 연결")]
    public TMP_Text exerciseName;
    public TMP_Text totalNum;

    [Header("달성 결과")]
    public TMP_Text achivedText;
    public CircleFill circleFill;

    [Header("미달성 결과")]
    public TMP_Text unachivedText;
    public CircleFill unachivedCircleFill;

    [Header("피드백 결과")]
    public TMP_Text perfectText;
    public TMP_Text goodText;
    public TMP_Text badText;

    void Start()
    {
        // 1. 운동 이름 표시
        string videoFileName = PlayerPrefs.GetString("exerciseName", "");
        exerciseName.text = videoFileName;

        // 2. 데이터 불러오기
        int perfectCount = PlayerPrefs.GetInt("PerfectCount", 0);
        int notgoodCount = PlayerPrefs.GetInt("NotgoodCount", 0);
        int goodCount = PlayerPrefs.GetInt("GoodCount", 0);
        int achivedCount = PlayerPrefs.GetInt("AchivedCount", 0);
        int num = PlayerPrefs.GetInt("Num", 0);
        int totalSet = PlayerPrefs.GetInt("TotalSet", 0);
        Debug.Log("totalset"+totalSet);
        
        // 3. 총 횟수 계산 및 표시 (실제 수행 횟수 기준)
        int totalReps = (achivedCount>num * totalSet)?achivedCount:num * totalSet;
        totalNum.text = "총 횟수 " + totalReps + "회";

        achivedText.text = achivedCount.ToString();
        if (circleFill != null)
        {
            circleFill.value = achivedCount;
            circleFill.fillValue = (totalReps > 0) ? (float)achivedCount / totalReps : 0;
        }

        // 5. Good 횟수 및 그래프 업데이트
        int unachivedCount = totalReps - achivedCount;
        unachivedCount = (unachivedCount<0)?0:unachivedCount;
        unachivedText.text = unachivedCount.ToString();
        if (unachivedCircleFill != null)
        {
            unachivedCircleFill.value = unachivedCount;
            unachivedCircleFill.fillValue = (totalReps > 0) ? (float)unachivedCount / totalReps : 0;
        }

        perfectText.text = perfectCount.ToString();
        goodText.text = goodCount.ToString();
        badText.text = notgoodCount.ToString();
    }
}
