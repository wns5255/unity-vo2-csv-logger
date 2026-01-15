using UnityEngine;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Globalization; // ★추가: 파싱 문화권 고정

public class VO2Bridge : MonoBehaviour
{
    [Header("Paths")]
    public string pythonPath   = @"C:\Users\wns5255\AppData\Local\Programs\Python\Python310\python.exe";
    // public string scriptPath   = @"C:\VO2\vo2_trainer_resid.py";
    public string scriptPath   = @"C:\VO2\vo2_trainer_resid_pred.py";
    public string polarCsvPath = @"C:\VO2\unity_vo2\result.csv";
    
    // [추가] Reference Table 경로 변수 추가
    public string refTablePath = @"C:\VO2\ref_table.csv";

    // public string artifactPkl  = @"C:\VO2\out_vo2\model_artifact.pkl";
    public string artifactPkl  = @"C:\VO2\out_user_motion_calibrated\model_artifact.pkl";
    // public string outDir       = @"C:\VO2\out_unity";
    public string outDir       = @"C:\Users\wns5255\Desktop\251216_test";


    [Tooltip("파이썬이 출력하는 최종 CSV 파일명 (예: pred_from_polar.csv)")]
    public string resultFileName = "pred_from_polar.csv"; // ★A: 실제로 사용하도록 변경

    [Header("Run Options")]
    public bool runOnStart = false;   // 필요 시 자동 실행
    public bool showWindow = false;   // Tkinter 창 필요 시 true

    void Start()
    {
        if (runOnStart)
            _ = RunPredictSafe(); // fire-and-forget
    }

    // ★B: 에디터에서 우클릭 → 실행
    [ContextMenu("VO2 ▶ Run Predict Now")]
    public void RunPredictNow()
    {
        _ = RunPredictSafe(0);
    }

    /// <summary>
    /// 파이썬 예측 실행. 성공시 true 반환.
    /// timeoutMs>0 이면 해당 시간 내 완료 못하면 false 반환(프로세스 강제종료는 안함).
    /// </summary>
    public async Task<bool> RunPredictSafe(int timeoutMs = 0)
    {
        // ✅ 0-1. SceneIngestServer가 저장해 둔 최신 CSV 경로로 교체
        var lastPath = PlayerPrefs.GetString("LastPolarCsvPath", "");
        if (!string.IsNullOrEmpty(lastPath))
        {
            polarCsvPath = lastPath;
            Debug.Log($"[VO2] PlayerPrefs에서 최신 CSV 경로 사용: {polarCsvPath}");
        }
        // 0) 경로 검증 + out 디렉토리 준비
        if (!File.Exists(pythonPath)) { Debug.LogError($"[VO2] python 없음: {pythonPath}"); return false; }
        if (!File.Exists(scriptPath)) { Debug.LogError($"[VO2] 스크립트 없음: {scriptPath}"); return false; }
        if (!File.Exists(polarCsvPath)) { Debug.LogError($"[VO2] POLAR CSV 없음: {polarCsvPath}"); return false; }
        if (!File.Exists(artifactPkl)) { Debug.LogError($"[VO2] Artifact 없음: {artifactPkl}"); return false; }

        // [추가] Ref Table 존재 확인
        if (!File.Exists(refTablePath)) { Debug.LogError($"[VO2] Ref Table 없음: {refTablePath}"); return false; }

        if (!Directory.Exists(outDir)) Directory.CreateDirectory(outDir);

        // // 1) 인자 구성
        // var args =
        //     $"--predict-from-polar \"{polarCsvPath}\" " +
        //     $"--load-artifact \"{artifactPkl}\" " +
        //     $"--out \"{outDir}\"";

    // =================================================================================
        // [NEW] 1. 원본 파일명 추출 및 최종 저장 경로 설정
        // =================================================================================
        // 예: "C:\rehab\polar_bsj_a_1.csv" -> "polar_bsj_a_1.csv"
        string originFileName = Path.GetFileName(polarCsvPath); 
        
        // 파이썬이 생성할 기본 파일 경로 (고정)
        string pythonDefaultOutput = Path.Combine(outDir, "pred_from_polar.csv");
        
        // 우리가 최종적으로 만들 파일 경로 (outDir + 원본파일명)
        string finalOutputPath = Path.Combine(outDir, originFileName);

// 1) 인자 구성 (CMD에서 성공했던 명령어 옵션 그대로 적용)
        var args =
            $"--predict-from-polar \"{polarCsvPath}\" " +
            $"--load-artifact \"{artifactPkl}\" " +
            $"--out \"{outDir}\" " +
            $"--ref-table \"{refTablePath}\" " +      // [추가] Ref Table 경로
            $"--calibrate-active-only 1 " +           // [추가] 운동 구간만 보정
            $"--use-bias-head 0 " +                     // [추가] 2차 보정 끄기 (Ref값 유지)
            $"--polar-time-col \"isoTime\"";

        var workDir = Path.GetDirectoryName(scriptPath);

        try
        {
            Debug.Log("[VO2] Python 실행 시작…");
            var runTask = PythonRunner.RunAsync(
                pythonPath, scriptPath, args, workDir, noWindow: !showWindow
            );

            if (timeoutMs > 0)
            {
                var finished = await Task.WhenAny(runTask, Task.Delay(timeoutMs)) == runTask;
                if (!finished)
                {
                    Debug.LogWarning($"[VO2] 타임아웃({timeoutMs}ms) — 씬 전환은 계속 진행");
                    return false;
                }
            }

            var (code, stdout, stderr) = await runTask;

            Debug.Log($"[VO2] ExitCode={code}");
            if (!string.IsNullOrWhiteSpace(stdout))
                Debug.Log($"[VO2][OUT]\n{stdout}");
            if (!string.IsNullOrWhiteSpace(stderr))
                Debug.LogWarning($"[VO2][ERR]\n{stderr}");

            if (code != 0) return false;

            // =================================================================================
            // [NEW] 3. 파일 이름 변경 (pred_from_polar.csv -> polar_bsj_a_1.csv)
            // =================================================================================
            if (File.Exists(pythonDefaultOutput))
            {
                // 이미 같은 이름의 파일이 있으면 삭제 (덮어쓰기 위해)
                if (File.Exists(finalOutputPath))
                {
                    File.Delete(finalOutputPath);
                }

                // 이름 변경 (Move)
                File.Move(pythonDefaultOutput, finalOutputPath);
                Debug.Log($"[VO2] 파일명 변경 완료: {pythonDefaultOutput} -> {finalOutputPath}");
            }
            else
            {
                Debug.LogError($"[VO2] 파이썬 출력 파일이 없습니다: {pythonDefaultOutput}");
                return false;
            }

            // 2) 결과 확인/간단 파싱
            // var predCsv = Path.Combine(outDir, resultFileName); // ★A: 하드코딩 제거

            var predCsv = finalOutputPath; // ★ 여기가 중요! 바뀐 경로를 읽습니다.
            
            Debug.Log($"[VO2] 결과 파일 확인: {predCsv}");
            if (!File.Exists(predCsv))
            {
                // 보조 로그: 폴더에 뭐가 있는지 한번 보여줌
                var hint = string.Join(", ", Directory.GetFiles(outDir, "*.csv").Select(Path.GetFileName));
                Debug.LogError($"[VO2] 결과 파일 없음: {predCsv}  (폴더 내 CSV: {hint})");
                return false;
            }

            var lines = File.ReadAllLines(predCsv);
            if (lines.Length > 1)
            {
                var header = lines[0].Split(',');
                int idxTime = System.Array.IndexOf(header, "time");
                int idxPred = System.Array.IndexOf(header, "VO2_PRED_Lmin");
                int idxAct  = System.Array.IndexOf(header, "ACTIVE");

                var times = new System.Collections.Generic.List<string>();
                var preds = new System.Collections.Generic.List<float>();
                var acts  = new System.Collections.Generic.List<int>();

                // ★C: InvariantCulture로 파싱
                var nf = CultureInfo.InvariantCulture;
                foreach (var line in lines.Skip(1))
                {
                    var cols = line.Split(',');
                    if (cols.Length <= 1) continue;
                    if (idxTime >= 0) times.Add(cols[idxTime]);

                    if (idxPred >= 0 && float.TryParse(cols[idxPred], NumberStyles.Float, nf, out var v))
                        preds.Add(v);

                    if (idxAct  >= 0 && int.TryParse(cols[idxAct], NumberStyles.Integer, nf, out var a))
                        acts.Add(a);
                }
                Debug.Log($"[VO2] 예측 로드 OK: {preds.Count}개 (첫 값={(preds.Count>0?preds[0]:0f)})");
            }

            return true;
        }
        catch (System.Exception ex)
        {
            Debug.LogException(ex);
            return false;
        }
    }
}
