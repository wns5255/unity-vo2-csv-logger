using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using TMPro;


public class RecommendNumber: MonoBehaviour 
{
    [Header("UI")]
    public TMP_Text targetText;                 // 출력할 TMP_Text

    [Header("Prefs Keys / Limits")]
    public int minCount = 1;    // 키가 없거나 0 이하일 때 최소 보정
    public int maxCount = 200;  // 상한선(필요시 조정)

    void OnEnable()
    {
        UpdateRecommendedText();
    }

    void Start()
    {
        UpdateRecommendedText();
    }

    /// <summary>
    /// PlayerPrefs("RecommendedRepeatNum")만 사용해서 텍스트 갱신
    /// </summary>
    public void UpdateRecommendedText()
    {
        if (targetText == null)
        {
            Debug.LogError("[RecommendNumber] targetText 가 비어 있습니다.");
            return;
        }

        int rec = PlayerPrefs.GetInt("RecommendedRepeatNum", minCount);
        rec = Mathf.Clamp(rec, minCount, maxCount);

        targetText.text = "추천! " + rec.ToString() + "세트";
    }

}
