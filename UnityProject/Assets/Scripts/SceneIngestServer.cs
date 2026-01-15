using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Threading;
using System.Globalization;
using System.Linq;

public class SceneIngestServer : MonoBehaviour
{
    [Header("PC Ingest Server")]
    public int ingestPort = 5555;                           // http://PC:5555/ingest

    [Header("File Settings")]
    public string saveDirectory = "C:/rehab_logs";          // 폴더만 지정
    public bool   useDateFolder = false;                    // 날짜별 하위 폴더 생성 (예: 2025-09-10)
    public string resultFileName = "result.csv";            // 항상 이 이름으로 저장

    private HttpListener listener;
    private Thread httpThread;
    private volatile bool running;

    // ⬇️ 세션 파일 경로
    private string sessionFilePath;     // C:\rehab_logs\[날짜]\result.csv (옵션)

    private string currentExerciseCode = "unknown";

    private string currentSubject = "배상준";

    void Awake()
    {
        // StartServer();   // 씬 들어오면 자동 시작
    }

    void OnDestroy()
    {
        StopServer();    // 씬 나가면 자동 종료
    }

    // --- helpers ---
    private static int PhaseToMotionId(string phase)
    {
        if (string.IsNullOrEmpty(phase)) return 0;
        switch (phase.Trim().ToUpperInvariant())
        {
            case "ACTIVE": return 1;
            case "REST":   return 0;
            default:       return 0;
        }
    }

    // 운동 한글 이름을 코드(a/b/c/d)로 맵핑
    private static string MapExerciseNameToCode(string exerciseName)
    {
        if (string.IsNullOrEmpty(exerciseName))
            return "unknown";

        exerciseName = exerciseName.Trim();

        // 원하는대로 매핑
        if (exerciseName.Contains("오픈캔"))
            return "a";   // 오픈캔
        if (exerciseName.Contains("검빼듯"))
            return "b";   // 검빼듯
        if (exerciseName.Contains("내회전"))
            return "c";   // 내회전
        if (exerciseName.Contains("외회전"))
            return "d";   // 외회전 (둘 다 c로 쓰고 싶으면 이 줄도 "c"로 바꾸면 됨)

        // 매핑 안 된 건 원래 이름 그대로
        return exerciseName;
    }


    private static string ToIso(long epochMs)
    {
        return DateTimeOffset.FromUnixTimeMilliseconds(epochMs)
                             .ToLocalTime()
                             .ToString("yyyy-MM-dd HH:mm:ss");
    }

    /// <summary>
    /// 안드로이드에서 보낸 body(여러 줄 가능)를
    /// isoTime(초단위)을 맨 앞에, phase와 motion_id를 맨 뒤에 붙인 CSV 라인들로 변환
    /// </summary>
    public string TransformBodyToIsoCsvWithPhase(string body, string phase, string exerciseCode, string currentSubject)
    {
        if (string.IsNullOrEmpty(body)) return string.Empty;

        var sb = new StringBuilder();
        var lines = body.Split('\n');
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (string.IsNullOrEmpty(line)) continue;
            if (line.StartsWith("epochMs", StringComparison.OrdinalIgnoreCase)) continue;

            var parts = line.Split(',');
            if (parts.Length < 1) continue;

            // epochMs 파싱
            if (!long.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var epochMs))
                continue;

            // isoTime: 초 단위
            var iso = DateTimeOffset.FromUnixTimeMilliseconds(epochMs)
                                    .ToLocalTime()
                                    .ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

            // 최종 라인 구성:
            // isoTime,epochMs,(원본의 elapsedSec~sex),phase,motion_id
            sb.Append(iso)
            .Append(',')
            .Append(epochMs.ToString(CultureInfo.InvariantCulture))
            .Append(',')
            .Append(string.Join(",", parts.Skip(1)))
            .Append(',')
            .Append(phase)          // "REST" / "ACTIVE"
            .Append(',')
            .Append(exerciseCode)   // "a" / "b" / "c" / "d"
            .Append(',')
            .Append(currentSubject) // subject 추가
            .Append('\n');
        }
        return sb.ToString();
    }

    public void StartServer()
    {
        if (running) return;

        // ⬇️ 디렉터리 구성 (옵션: 날짜 폴더)
        var dir = saveDirectory;
        if (useDateFolder)
        {
            var dateFolder = DateTime.Now.ToString("yyyy-MM-dd");
            dir = Path.Combine(saveDirectory, dateFolder);
        }
        Directory.CreateDirectory(dir);


        // 1) 사용자 이름 읽기 (SetCount 씬에서 PlayerPrefs.SetString("UserName", ...) 저장해둔 값)
        var userName = PlayerPrefs.GetString("UserName", "noname");

        // 2) 운동 이름(한글) 읽기 — Video.LoadClip에서 SharedPhase.CurrentVideoLogName에 이미 세팅됨
        var exerciseName = SharedPhase.CurrentVideoLogName;

        // 비어 있으면 기본값 처리
        if (string.IsNullOrEmpty(exerciseName))
        {
            exerciseName = Path.GetFileNameWithoutExtension(resultFileName); // "result" 같은 기본값
        }

        // 🔹 운동 이름 → 코드(a/b/c/d)로 변환
        var exerciseCode = MapExerciseNameToCode(exerciseName);

        // 🔹 이번 세션 운동 코드를 필드에 저장
        currentExerciseCode = exerciseCode;

        currentSubject = userName;  // 사용자 이름을 서브젝트로 설정

        // 3) 파일 이름에 쓸 수 없는 문자 제거 (한글은 그대로 사용 가능)
        foreach (var c in Path.GetInvalidFileNameChars())
        {
            userName     = userName.Replace(c.ToString(), "_");
            exerciseCode = exerciseCode.Replace(c.ToString(), "_");
            currentSubject = currentSubject.Replace(c.ToString(), "_"); // 서브젝트도 필터링
        }

        // 4) 최종 baseName: "홍길동_a" 이런 형태
        var baseName = $"{userName}_{exerciseCode}";

        // 5) 확장자 붙이기
        var fileName = baseName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
            ? baseName
            : baseName + ".csv";

        // 6) 최종 경로
        sessionFilePath = Path.Combine(dir, fileName);

        // 동일 이름의 파일이 이미 있으면 삭제하고 새로 시작
        try
        {
            if (File.Exists(sessionFilePath))
            {
                File.Delete(sessionFilePath);
                Debug.Log($"[Ingest] 기존 파일 삭제 후 새로 생성: {sessionFilePath}");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Ingest] 기존 파일 삭제 실패: {sessionFilePath}\n{e.Message}");
        }

        // 헤더 한 줄 새로 쓰기 (여기는 이어쓰기 말고 처음에만 한 번)
        try
        {
            File.WriteAllText(
                sessionFilePath,
                "isoTime,epochMs,elapsedSec,hr,avgHr,hrr,vo2Rel_mlkgmin,vo2Abs_Lmin,intervalSec,weight,vo2max,restingHr,height,age,sex,phase,motion_id,subject\n",
                Encoding.UTF8
            );
        }
        catch (Exception e)
        {
            Debug.LogError($"[Ingest] 헤더 쓰기 실패: {sessionFilePath}\n{e.Message}");
        }

        // ✅ 여기 추가: 이번 세션 CSV 경로를 PlayerPrefs에 저장
        PlayerPrefs.SetString("LastPolarCsvPath", sessionFilePath);
        PlayerPrefs.Save();


        // ⬇️ 기존 파일 있으면 삭제 후 헤더 생성
        try
        {
            if (File.Exists(sessionFilePath))
                File.Delete(sessionFilePath);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[Ingest] 기존 파일 삭제 실패: {sessionFilePath}\n{e.Message}");
        }

        try
        {
            File.AppendAllText(
                sessionFilePath,
                "isoTime,epochMs,elapsedSec,hr,avgHr,hrr,vo2Rel_mlkgmin,vo2Abs_Lmin,intervalSec,weight,vo2max,restingHr,height,age,sex,phase,motion_id,subject\n",
                Encoding.UTF8
            );
        }
        catch (Exception e)
        {
            Debug.LogError($"[Ingest] 헤더 생성 실패: {sessionFilePath}\n{e.Message}");
        }

        // 서버 시작
        listener = new HttpListener();
        listener.Prefixes.Add($"http://+:{ingestPort}/");
        listener.IgnoreWriteExceptions = true;

        try
        {
            listener.Start();
        }
        catch (Exception e)
        {
            Debug.LogError($"[Ingest] Listener start 실패: {e.Message}\n" +
                           "• 포트 중복/방화벽 확인\n• 이전 씬에서 서버 Stop 누락 여부 확인\n• 관리자 권한/URLACL 권한 필요 여부(netsh http add urlacl...)");
            return;
        }

        running = true;
        httpThread = new Thread(HttpLoop) { IsBackground = true };
        httpThread.Start();

        Debug.Log($"[Ingest] Listening: http://0.0.0.0:{ingestPort}/ingest");
        Debug.Log($"[Ingest] Session file -> {sessionFilePath}");
    }

    public void StopServer()
    {
        if (!running) return;
        running = false;

        try { listener?.Stop(); } catch { }
        try { httpThread?.Join(500); } catch { }

        listener = null;
        httpThread = null;

        Debug.Log("[Ingest] Stopped");
    }

    private void HttpLoop()
    {
        while (running)
        {
            HttpListenerContext ctx = null;
            try
            {
                ctx = listener.GetContext();
            }
            catch
            {
                if (!running) break;
                else continue;
            }

            try
            {
                var req = ctx.Request;
                var res = ctx.Response;

                res.ProtocolVersion = System.Net.HttpVersion.Version11;
                res.SendChunked = false;
                res.AddHeader("Connection", "close");

                if (req.HttpMethod == "POST" && req.Url.AbsolutePath == "/ingest")
                {
                    string body;
                    using (var reader = new StreamReader(req.InputStream, req.ContentEncoding))
                        body = reader.ReadToEnd();

                    // 현재 phase 라벨 (예: SharedPhase.Current = "REST"/"ACTIVE")
                    string phase = SharedPhase.Current;

                    // isoTime + phase + motion_id 붙여 변환
                    var converted = TransformBodyToIsoCsvWithPhase(body, phase, currentExerciseCode, currentSubject);

                    if (!string.IsNullOrEmpty(converted))
                    {
                        File.AppendAllText(sessionFilePath, converted, Encoding.UTF8);
                    }

                    // (선택) 화면 디스플레이 훅
                    TryEnqueueForDisplay(converted);

                    // 응답
                    byte[] ok = Encoding.UTF8.GetBytes("OK");
                    res.StatusCode = 200;
                    res.ContentType = "text/plain; charset=utf-8";
                    res.ContentLength64 = ok.Length;
                    res.OutputStream.Write(ok, 0, ok.Length);
                    res.OutputStream.Flush();
                    res.OutputStream.Close();
                    res.Close();
                }
                else
                {
                    byte[] notFound = Encoding.UTF8.GetBytes("Not Found");
                    res.StatusCode = 404;
                    res.ContentType = "text/plain; charset=utf-8";
                    res.ContentLength64 = notFound.Length;
                    res.OutputStream.Write(notFound, 0, notFound.Length);
                    res.OutputStream.Flush();
                    res.OutputStream.Close();
                    res.Close();
                }
            }
            catch (Exception e)
            {
                try
                {
                    var res = ctx?.Response;
                    if (res != null && res.OutputStream != null)
                    {
                        byte[] err = Encoding.UTF8.GetBytes("ERR");
                        res.StatusCode = 500;
                        res.SendChunked = false;
                        res.AddHeader("Connection", "close");
                        res.ContentType = "text/plain; charset=utf-8";
                        res.ContentLength64 = err.Length;
                        res.OutputStream.Write(err, 0, err.Length);
                        res.OutputStream.Flush();
                        res.OutputStream.Close();
                        res.Close();
                    }
                }
                catch { /* ignore */ }

                Debug.LogError("[Ingest] " + e.Message);
            }
        }
    }

    // 디스플레이 컴포넌트가 있을 때만 안전하게 호출
    private static void TryEnqueueForDisplay(string converted)
    {
        try
        {
            IngestDisplayHRVO2.EnqueueBody(converted);
            // ↑ 존재하는 프로젝트라면 주석 해제
        }
        catch { /* 화면 모듈이 없으면 무시 */ }
    }
}
