using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
public class MessageManager : MonoBehaviour
{
    
    public static MessageManager Instance;

    [SerializeField]
    private GameObject message_box;
    [SerializeField]
    private TMP_Text message_text;
    [SerializeField]
    private Animator animator;

    private readonly WaitForSeconds delay2 = new WaitForSeconds(0.3f);
    private const float MESSAGE_DURATION = 2.0f;

    private Coroutine _hideCoroutine;
    private float _messageEndTime;
    private Vector2 _originalPosition;

    private void Awake()
    {
        message_box.SetActive(false);
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
        if (message_box != null)
        {
            _originalPosition = (message_box.transform as RectTransform).anchoredPosition;
        }

    }

    public void DisplayMessage(string message,float? yPosition=null)
    {
        //Debug.Log($"[MessageManager] DisplayMessage: {message}, Time: {Time.time}");
        // UI 컴포넌트가 없으면 아무것도 하지 않음
        if (message_box == null || message_text == null || animator == null)
            return;

        RectTransform rectTransform = message_box.transform as RectTransform;
        if (yPosition.HasValue)
        {
            rectTransform.anchoredPosition = new Vector2(_originalPosition.x, yPosition.Value);
        }
        else
        {
            rectTransform.anchoredPosition = _originalPosition;
        }

        // 메시지 내용을 업데이트하고, 메시지 박스를 활성화
        message_text.text = message;
        message_box.SetActive(true);
        animator.SetBool("isOn", true);
        
        // 메시지가 사라질 시간을 현재 시간으로부터 2초 뒤로 설정
        _messageEndTime = Time.time + MESSAGE_DURATION;

        // 숨기기 코루틴이 실행 중이지 않다면 새로 시작
        if (_hideCoroutine == null)
        {
            _hideCoroutine = StartCoroutine(HideMessageCoroutine());
        }
    }

    private IEnumerator HideMessageCoroutine()
    {
        // 메시지가 사라져야 할 시간까지 계속 대기
        // 루프 중간에 새로운 메시지 요청이 들어오면 _messageEndTime이 갱신되어 대기 시간이 연장됨
        while (Time.time < _messageEndTime)
        {
            yield return null;
        }

        // 시간이 다 되면 메시지 숨기기 애니메이션 실행
        if (animator != null)
            animator.SetBool("isOn", false);

        yield return delay2;

        // 최종적으로 UI 비활성화
        if (message_box != null)
            message_box.SetActive(false);
        if (message_text != null)
            message_text.text = "";

        // 코루틴 핸들 초기화
        _hideCoroutine = null;
    }
}