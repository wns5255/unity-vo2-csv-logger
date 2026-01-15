using UnityEngine;
using UnityEngine.UI;   // UI 바인딩 시 사용 (선택)
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class HrCollector : MonoBehaviour
{
    [Header("PC Ingest Server (유니티가 여는 수집 서버)")]
    public int ingestPort = 5555;                          // http://PC_IP:5555/ingest
    public string savePath = "C:/rehab_logs/rehab_stream.csv";

    [Header("Android Control (유니티 → 폰)")]
    public string phoneIp = "192.168.0.50";                // 안드로이드 기기 IP
    public int controlPort = 6000;                         // 폰에서 리스닝하는 UDP 포트

    [Header("User Info (UI로 바인딩해도 되고 코드로 세팅해도 됨)")]
    public float weight = 70f;      // kg
    public float vo2max = 45f;      // ml/kg/min
    public int restingHr = 60;      // bpm
    public int height = 175;        // cm
    public int age = 25;
    public int sex = 1;             // 남=1, 여=0

    // (선택) UI 바인딩용
    public InputField weightInput, vo2maxInput, restingInput, heightInput, ageInput;
    public Dropdown sexDropdown; // 0=여, 1=남

    private HttpListener listener;
    private Thread httpThread;
    private volatile bool running;

    void Start()
    {
        // 저장 폴더 준비
        var dir = Path.GetDirectoryName(savePath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        // 파일 헤더 없으면 생성
        if (!File.Exists(savePath))
        {
            File.AppendAllText(savePath,
                "epochMs,elapsedSec,hr,avgHr,hrr,vo2Rel_mlkgmin,vo2Abs_Lmin,intervalSec,weight,vo2max,restingHr,height,age,sex\n",
                Encoding.UTF8);
        }

        // PC 수집 서버 시작
        listener = new HttpListener();
        listener.Prefixes.Add($"http://+:{ingestPort}/");
        listener.Start();
        running = true;
        httpThread = new Thread(HttpLoop) { IsBackground = true };
        httpThread.Start();

        Debug.Log($"[Unity] Ingest listening: http://0.0.0.0:{ingestPort}/ingest");
        Debug.Log($"[Unity] Saving -> {savePath}");
    }

    void OnDestroy()
    {
        running = false;
        try { listener?.Stop(); } catch { }
        try { httpThread?.Join(500); } catch { }
    }

    private void HttpLoop()
    {
        while (running)
        {
            HttpListenerContext ctx = null;
            try { ctx = listener.GetContext(); }
            catch { if (!running) break; else continue; }

            var req = ctx.Request;
            var res = ctx.Response;

            if (req.HttpMethod == "POST" && req.Url.AbsolutePath == "/ingest")
            {
                using (var reader = new StreamReader(req.InputStream, req.ContentEncoding))
                {
                    string body = reader.ReadToEnd();
                    File.AppendAllText(savePath, body, Encoding.UTF8);
                }
                var bytes = Encoding.UTF8.GetBytes("OK");
                res.StatusCode = 200;
                res.OutputStream.Write(bytes, 0, bytes.Length);
            }
            else
            {
                res.StatusCode = 404;
            }
            res.Close();
        }
    }

    // ====== UI 버튼에서 바인딩 ======

    public void OnClickStart()
    {
        // (선택) UI 값 읽어 반영
        PullUi();

        string pcUrl = $"http://{GetLocalIPv4()}:{ingestPort}/ingest";
        string payload = $"START|{pcUrl}|{weight}|{vo2max}|{restingHr}|{height}|{age}|{sex}";
        SendUdp(payload);
        Debug.Log("[Unity] START sent: " + payload);
    }

    public void OnClickStop()
    {
        SendUdp("STOP");
        Debug.Log("[Unity] STOP sent");
    }

    // (선택) 구간 마커 보내고 싶을 때
    public void SendMark(string side, int rep, string phase) // phase: push/rest, *_start/*_end 등
    {
        string payload = $"MARK|{side}|{rep}|{phase}";
        SendUdp(payload);
        Debug.Log("[Unity] MARK sent: " + payload);
    }

    // ====== 내부 유틸 ======

    private void SendUdp(string msg)
    {
        try
        {
            using var udp = new UdpClient();
            var ep = new IPEndPoint(IPAddress.Parse(phoneIp), controlPort);
            var data = Encoding.UTF8.GetBytes(msg);
            udp.Send(data, data.Length, ep);
        }
        catch (Exception e)
        {
            Debug.LogError("[Unity] UDP send error: " + e.Message);
        }
    }

    private string GetLocalIPv4()
    {
        foreach (var ni in System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces())
        {
            if (ni.OperationalStatus != System.Net.NetworkInformation.OperationalStatus.Up) continue;
            var ipProps = ni.GetIPProperties();
            foreach (var ua in ipProps.UnicastAddresses)
            {
                if (ua.Address.AddressFamily == AddressFamily.InterNetwork)
                {
                    return ua.Address.ToString(); // 첫 IPv4 반환
                }
            }
        }
        return "127.0.0.1";
    }

    private void PullUi()
    {
        if (weightInput)  float.TryParse(weightInput.text, out weight);
        if (vo2maxInput)  float.TryParse(vo2maxInput.text, out vo2max);
        if (restingInput) int.TryParse(restingInput.text, out restingHr);
        if (heightInput)  int.TryParse(heightInput.text, out height);
        if (ageInput)     int.TryParse(ageInput.text, out age);
        if (sexDropdown)  sex = sexDropdown.value; // 0=여, 1=남 으로 설계했다면
    }
}
