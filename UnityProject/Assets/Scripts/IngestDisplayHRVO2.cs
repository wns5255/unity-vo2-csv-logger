using UnityEngine;
using System.Collections.Concurrent;
using System.Globalization;
using TMPro;

public class IngestDisplayHRVO2 : MonoBehaviour
{
    [Header("UI (TextMeshPro)")]
    public TMP_Text hrText;        // 심박수
    public TMP_Text vo2RelText;    // VO2 (ml/kg/min)
    public TMP_Text vo2AbsText;    // VO2 (L/min)

    private static readonly ConcurrentQueue<string> queue = new ConcurrentQueue<string>();
    private static readonly CultureInfo ci = CultureInfo.InvariantCulture;

    /// <summary>
    /// 서버에서 전달된 본문(여러 줄 가능)을 통째로 큐에 넣습니다.
    /// SceneIngestServer에서 converted 문자열을 넘겨주세요.
    /// </summary>
    public static void EnqueueBody(string body)
    {
        if (!string.IsNullOrEmpty(body))
            queue.Enqueue(body);
    }

    void Update()
    {
        // 마지막 유효 라인만 화면에 반영 (누적 처리 중복 방지)
        string pending = null;

        while (queue.TryDequeue(out var body))
        {
            var lines = body.Split('\n');
            foreach (var raw in lines)
            {
                var line = raw.Trim();
                if (string.IsNullOrEmpty(line)) continue;
                if (line.StartsWith("epochMs")) continue; // 옛 헤더
                if (line.StartsWith("isoTime")) continue; // 새 헤더

                pending = line; // 가장 마지막 라인을 채택
            }
        }

        if (string.IsNullOrEmpty(pending)) return;

        var parts = pending.Split(',');
        // 새 포맷: isoTime,epochMs,elapsedSec,hr,avgHr,hrr,vo2Rel_mlkgmin,vo2Abs_Lmin,...
        // 옛 포맷: epochMs,elapsedSec,hr,avgHr,hrr,vo2Rel_mlkgmin,vo2Abs_Lmin,...
        bool firstIsEpoch = long.TryParse(parts[0], NumberStyles.Integer, ci, out _);

        // 인덱스 계산
        int hrIdx     = firstIsEpoch ? 2 : 3;
        int vo2RelIdx = firstIsEpoch ? 5 : 6;
        int vo2AbsIdx = firstIsEpoch ? 6 : 7;

        // 방어: 인덱스 범위 확인
        if (parts.Length <= Mathf.Max(hrIdx, vo2RelIdx, vo2AbsIdx)) return;

        // 안전 파싱
        if (int.TryParse(parts[hrIdx], NumberStyles.Integer, ci, out var hr) &&
            double.TryParse(parts[vo2RelIdx], NumberStyles.Float, ci, out var vo2Rel) &&
            double.TryParse(parts[vo2AbsIdx], NumberStyles.Float, ci, out var vo2Abs))
        {
            if (hrText)     hrText.text     = $"HR : {hr} bpm";
            // if (vo2RelText) vo2RelText.text = $"VO₂(rel): {vo2Rel:F2} ml/kg/min";
            // if (vo2AbsText) vo2AbsText.text = $"VO₂(abs): {vo2Abs:F3} L/min";
        }
    }
}
