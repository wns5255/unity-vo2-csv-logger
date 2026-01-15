using UnityEngine;
using UnityEngine.SceneManagement;
using System.Threading.Tasks;

public class ChangeScene : MonoBehaviour
{
    // 씬 전환 전 대기 타임아웃(ms). 0이면 무제한 대기
    [SerializeField] private int predictTimeoutMs = 25000;

    // 공통 진입점
    private async Task LoadAfterPredict(string sceneName)
    {
        var bridge = FindObjectOfType<VO2Bridge>();
        if (bridge != null)
        {
            // 성공/실패와 무관하게 전환은 진행(리턴값으로 분기 가능)
            bool ok = await bridge.RunPredictSafe(predictTimeoutMs);
            Debug.Log($"[ChangeScene] 예측 완료 ok={ok}, 이제 씬 전환: {sceneName}");
        }
        else
        {
            Debug.LogWarning("[ChangeScene] VO2Bridge가 씬에 없습니다. 바로 전환합니다.");
        }

        SceneManager.LoadScene(sceneName);
    }

    // ▼▼ 버튼에 이 메서드들을 연결하세요 ▼▼
    public async void toRegion()   { await LoadAfterPredict("pc_04_type"); }
    public async void toContents() { await LoadAfterPredict("pc_05_contents"); }
    public async void toRepeat()   { await LoadAfterPredict("pc_06_repeat"); }
    public async void toExercise() { await LoadAfterPredict("pc_07_exercise_znn"); }
    public async void toResult()   { await LoadAfterPredict("pc_08_result"); }
    public async void toMenu()     { await LoadAfterPredict("pc_10_menu"); }

    public void toQuit()
    {
        Application.Quit();
    }
}
