using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class CircleFill : MonoBehaviour
{
    public int? value = null;
    public int? maxValue = null;
    [Range(0, 1)] public float fillValue = 0;
    public Image circleFillImage;
    public RectTransform handlerEdgeImage;
    public RectTransform fillHandler;
    public Image handlercircleFill;

    public TMP_Text valueText;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (float.IsNaN(fillValue))
            fillValue = 0;
        fillCircleValue(fillValue);
        if (value != null)
            valueText.text = value.ToString();
    }

    void fillCircleValue(float fillAmount)
    {
        // 값을 0과 1 사이로 고정하여 안정성 확보
        fillAmount = Mathf.Clamp01(fillAmount);

        // 메인 이미지의 채우기 양 설정
        circleFillImage.fillAmount = fillAmount;

        // 그래프가 비어있거나(0%) 꽉 찼을 때(100%)는 핸들러를 숨겨서 깔끔하게 표시
        // 0.001과 0.999를 사용하는 것은 부동 소수점 오차를 방지하기 위함입니다.
        bool showHandlers = fillAmount > 0.001f && fillAmount < 0.999f;

        if (fillHandler != null)
        {
            fillHandler.gameObject.SetActive(showHandlers);
        }
        if (handlerEdgeImage != null)
        {
            handlerEdgeImage.gameObject.SetActive(showHandlers);
        }

        // --- 핸들러 회전 로직 수정 ---
        // 시작 핸들러(fillHandler)는 항상 12시 방향(0도)에 고정합니다.
        if (fillHandler != null)
        {
            fillHandler.localEulerAngles = Vector3.zero;
        }

        // 끝 핸들러(handlerEdgeImage)만 채우기 양에 따라 회전시킵니다.
        if (handlerEdgeImage != null)
        {
            float angle = -fillAmount * 360;
            handlerEdgeImage.localEulerAngles = new Vector3(0, 0, angle);
            if (circleFillImage != null)
            {
                handlercircleFill.rectTransform.localEulerAngles = new Vector3(0, 0, -angle);
            }
        }
    }
}
