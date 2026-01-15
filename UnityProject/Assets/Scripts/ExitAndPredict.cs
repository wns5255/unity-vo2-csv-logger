using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

public class ExitAndPredict : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private VO2Bridge vo2;         // 씬의 VO2Bridge를 Drag&Drop
    [SerializeField] private Button exitButton;     // 종료 버튼(선택)
    [SerializeField] private Text statusText;       // 진행 상태 표시용(선택)

    [Header("Behavior")]
    [SerializeField] private float timeoutSeconds = 30f; // 파이썬이 오래 걸릴 대비
    [SerializeField] private bool quitEvenIfFailed = true; // 실패해도 종료할지

    public async void OnExitClicked()
    {
        if (exitButton) exitButton.interactable = false;
        SetStatus("예측 실행 중…");

        bool ok = false;
        try
        {
            if (vo2 == null)
            {
                SetStatus("VO2Bridge가 없습니다.");
            }
            else
            {
                // 타임아웃과 병렬 대기
                var runTask = vo2.RunPredictSafe();
                var done = await Task.WhenAny(runTask, Task.Delay(TimeSpan.FromSeconds(timeoutSeconds)));
                if (done == runTask)
                {
                    ok = runTask.Result; // 성공/실패
                    SetStatus(ok ? "예측 완료" : "예측 실패");
                }
                else
                {
                    SetStatus("예측 타임아웃");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            SetStatus("예측 중 예외 발생");
        }

        // 종료 정책
        if (ok || quitEvenIfFailed)
        {
            SetStatus("종료 중…");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
        else
        {
            SetStatus("종료 취소(실패)");
            if (exitButton) exitButton.interactable = true;
        }
    }

    private void SetStatus(string msg)
    {
        if (statusText) statusText.text = msg;
        Debug.Log($"[ExitAndPredict] {msg}");
    }
}
