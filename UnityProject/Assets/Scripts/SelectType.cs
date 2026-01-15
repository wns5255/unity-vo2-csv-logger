using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SelectType : MonoBehaviour
{
    private string exerciseType;

    public void selectShoulder()
    {
        PlayerPrefs.DeleteKey("Type");
        PlayerPrefs.SetString("Type", "운동 프로토콜");
    }

    public void selectStretching()
    {
        PlayerPrefs.DeleteKey("Type");
        PlayerPrefs.SetString("Type", "회전근개");
    }

    public void selectCore()
    {
        PlayerPrefs.DeleteKey("Type");
        PlayerPrefs.SetString("Type", "Core");
    }
}
