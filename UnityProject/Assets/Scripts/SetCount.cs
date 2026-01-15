using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SetCount : MonoBehaviour
{
    //public TMP_Text setCountText;
    //private int setCount = 1;

    //public void IncreaseSetCount()
    //{
    //    if (setCount < 10)
    //    {
    //        setCount++;
    //        UpdateSetCountText();
    //    }
    //}

    //public void DecreaseSetCount()
    //{
    //    if (setCount > 1)
    //    {
    //        setCount--;
    //        UpdateSetCountText();
    //    }
    //}

    //void UpdateSetCountText()
    //{
    //    setCountText.text = setCount.ToString();
    //}

    //public void SetCountNumber()
    //{
    //    PlayerPrefs.SetInt("SetCount", setCount);
    //}
    [Header("UI")]
    public TMP_Text setCountText;
        // ✅ 이름 입력 필드 추가
    public TMP_InputField inputName;

    [Header("����")]
    public int minCount = 1;                // �ּҰ�
    public int maxCount = 30;               // �ִ밪 (���� �ڵ� �⺻ 10 ����)

    private int setCount;

    void Start()
    {
        InitializeFromPrefsOnly();
        UpdateSetCountText();
    }

    /// <summary>
    /// ���� PlayerPrefs("RecommendedRepeatNum")�� ����� �ʱ�ȭ
    /// ���� ������ minCount ���
    /// </summary>
    private void InitializeFromPrefsOnly()
    {
        int rec = PlayerPrefs.GetInt("RecommendedRepeatNum", minCount);
        setCount = Mathf.Clamp(rec, minCount, maxCount);
    }

    public void IncreaseSetCount()
    {
        if (setCount < maxCount)
        {
            setCount++;
            UpdateSetCountText();
        }
    }

    public void DecreaseSetCount()
    {
        if (setCount > minCount)
        {
            setCount--;
            UpdateSetCountText();
        }
    }

    private void UpdateSetCountText()
    {
        if (setCountText != null)
            setCountText.text = setCount.ToString();
    }

    /// <summary>
    /// ���� ���� PlayerPrefs("SetCount")�� ����
    /// </summary>
    public void SaveSetCount()
    {
        // 1) 세트 수 저장
        PlayerPrefs.SetInt("SetCount", setCount);

        // 2) 이름 저장
        if (inputName != null)
        {
            string userName = inputName.text.Trim();
            if (!string.IsNullOrEmpty(userName))
            {
                PlayerPrefs.SetString("UserName", userName);
            }
        }

        PlayerPrefs.Save();

        Debug.Log($"[SetCount] 저장: SetCount={setCount}, UserName={PlayerPrefs.GetString("UserName", "")}");
    }
}


