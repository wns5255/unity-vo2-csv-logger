using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using TMPro;
using System.Diagnostics;
using System.Linq;
using UnityEngine.SceneManagement;

public class Video : MonoBehaviour
{
    public static Video Instance { get; private set; } // 싱글톤 인스턴스
    
    [Header("Core Components")]
    public VideoPlayer video;
    public UDPReceiver udpReceiver;
    public ChangeScene sceneChange;

    [Header("UI Components")]
    public RawImage raw;
    public RawImage rawmask;
    public TMP_Text countdownText;
    public TMP_Text setText;
    public GameObject startbutton;
    public GameObject pausebutton;

    [Header("Silhouette Check")]
    public bool useSilhouetteCheck = true;
    public Material invertedMaskMaterial;
    public Texture2D humanSilhouetteForCheck;
    public Color outlineDefaultColor = Color.red;
    public Color outlineReadyColor = Color.green;

    [Header("달성 결과")]
    public TMP_Text achivedText;
    public CircleFill circleFill;

    // --- 내부 상태 관리 ---
    private IRehabilitationEvaluator rehabilitation;
    //private static Process pythonProcess; // static으로 변경
    private Texture2D receivedTexture;
    private bool isDestroyed = false; // 파괴 상태 플래그 추가

    private int setCount;
    private int currentSet = 1;
    private int prepareSec;
    private int restSec;
    private string exerciseId;
    private bool alternatingSides;

    private enum Side { Intro, L, R }
    private Side currentSide = Side.Intro;
    private Coroutine currentRoutine;
    private bool introPlayed = false;
    private bool readymotion = false;
    private bool evaluatorinit = false;
    private bool evaluatorstart = false;
    private bool initialMessageDisplayed = false;
    private float firstFrameTimestamp = -1f; // -1은 타이머 시작 전을 의미
    private const float READY_CHECK_DELAY = 1.0f; // 준비 자세 체크 시작 전 딜레이 (초)

    private int setStartPerfectCount;
    private int setStartGoodCount;
    private int setStartNotgoodCount;
    private int setStartAchivedCount;
    
    public TMP_Text phaseDebug;

    private static readonly Dictionary<string, string> LogNameMap = new Dictionary<string, string>
    {
        { "3_10_left", "오픈캔자세외전" },
        { "3_11_left", "검빼듯대각선리프트" },
        { "1_9_left",  "내회전" },
        { "1_10_left", "외회전" },
        // 필요하면 여기 계속 추가
        // { "3_10_right", "오픈캔자세외전(우측)" }, 이런 식으로도 확장 가능
    };

    private void SetPhase(string p)
    {
        SharedPhase.Current = p == "ACTIVE" ? "ACTIVE" : "REST";
        if (phaseDebug) phaseDebug.text = $"PHASE: {SharedPhase.Current}";
        UnityEngine.Debug.Log($"[Video] Phase -> {SharedPhase.Current}");
    }

    // --- 생명주기 및 이벤트 ---
    void OnEnable() { video.loopPointReached += OnVideoEnd; }
    void OnDisable() 
    { 
        video.loopPointReached -= OnVideoEnd; 
    }

    void Awake()
    {
        UnityEngine.Debug.Log($"[Video] Awake 시작 - GameObject: {gameObject.name}");

        
        // 싱글톤 패턴 구현
        if (Instance != null && Instance != this)
        {
            UnityEngine.Debug.LogWarning($"[Video] 중복된 Video 인스턴스 감지 - 현재 인스턴스를 파괴합니다.");
            Destroy(gameObject);
            return;
        }
        
        Instance = this;
        UnityEngine.Debug.Log($"[Video] Video 인스턴스가 설정되었습니다.");
        
        receivedTexture = new Texture2D(640, 720, TextureFormat.RGBA32, false);
        
        // // 오디오 출력을 위한 AudioSource 설정
        // if (video != null)
        // {
        //     video.audioOutputMode = VideoAudioOutputMode.AudioSource;

        //     // AudioSource 컴포넌트가 있는지 확인하고 없으면 추가
        //     AudioSource audioSource = GetComponent<AudioSource>();
        //     if (audioSource == null)
        //     {
        //         audioSource = gameObject.AddComponent<AudioSource>();
        //         audioSource.playOnAwake = false; // 비디오 플레이어가 제어하므로 자동 재생 비활성화
        //     }
            
        //     // AudioSource 기본 설정
        //     audioSource.volume = 1.0f;
        //     audioSource.mute = false;
            
        //     // VideoPlayer에 AudioSource 연결
        //     video.SetTargetAudioSource(0, audioSource);
            
        //     // 오디오 트랙 활성화
        //     video.EnableAudioTrack(0, true);
            
        //     UnityEngine.Debug.Log($"[Video] 오디오 설정 완료 - Volume: {audioSource.volume}, Mute: {audioSource.mute}");
        // }
    }

    void Start()
    {

        SetPhase("REST"); // 시작은 REST

        UnityEngine.Debug.Log("[Video] Start 호출됨 - 초기화 시작");
        
        // GameObject의 모든 컴포넌트 확인
        Component[] components = gameObject.GetComponents<Component>();
        UnityEngine.Debug.Log($"[Video] Manager GameObject의 컴포넌트 개수: {components.Length}");
        for (int i = 0; i < components.Length; i++)
        {
            UnityEngine.Debug.Log($"[Video] 컴포넌트 #{i}: {components[i].GetType().Name}");
        }
        
        // PlayerPrefs.SetInt("PerfectCount", 0);
        // PlayerPrefs.SetInt("NotgoodCount", 0);
        // PlayerPrefs.SetInt("GoodCount", 0);
        // PlayerPrefs.SetInt("BadCount", 0);

        // 필수 참조들이 할당되어 있는지 확인
        UnityEngine.Debug.Log($"[Video] video: {(video != null ? "할당됨" : "NULL")}");
        UnityEngine.Debug.Log($"[Video] udpReceiver: {(udpReceiver != null ? "할당됨" : "NULL")}");
        UnityEngine.Debug.Log($"[Video] sceneChange: {(sceneChange != null ? "할당됨" : "NULL")}");
        UnityEngine.Debug.Log($"[Video] raw: {(raw != null ? "할당됨" : "NULL")}");
        UnityEngine.Debug.Log($"[Video] countdownText: {(countdownText != null ? "할당됨" : "NULL")}");
        UnityEngine.Debug.Log($"[Video] setText: {(setText != null ? "할당됨" : "NULL")}");
        
        if (startbutton != null) startbutton.SetActive(false);
        if (pausebutton != null) pausebutton.SetActive(false);
        // if(achivedText!=null) achivedText.gameObject.SetActive(false);
        // if(circleFill!=null) circleFill.gameObject.SetActive(false);

        // 단순하게 Python 시작
        //StartPython();

        // PlayerPrefs에서 운동 정보 로드
        setCount = PlayerPrefs.GetInt("SetCount", 1);
        exerciseId = PlayerPrefs.GetString("exerciseId", "");
        prepareSec = PlayerPrefs.GetInt("preparesec", 10);
        restSec = PlayerPrefs.GetInt("restsec", 60);
        alternatingSides = PlayerPrefs.GetString("alternatingSides", "true").ToLower() == "true";

        // 초기 안내 메시지 클리어
        countdownText.text = "";

        setText.text = $"{currentSet} / {setCount} 세트";
        video.isLooping = false;

        // 첫 시작은 Intro 영상 로드 시도
        if (!LoadClip(Side.Intro))
        {
            // Intro 영상이 없으면 바로 Left로 시작 준비
            introPlayed = true;
            LoadClip(Side.L);
        }

        // ✅ 여기서 비디오 이름이 SharedPhase.CurrentVideoName에 이미 들어간 상태
        var ingest = FindObjectOfType<SceneIngestServer>();
        if (ingest != null)
        {
            ingest.StartServer();
        }

        click_start();
    }

    void Update()
    {
        // 파괴된 상태라면 더 이상 처리하지 않음
        if (isDestroyed) return;
        
        // GameObject가 비활성화되었는지 체크
        if (!gameObject.activeInHierarchy)
        {
            UnityEngine.Debug.LogWarning($"[Video] GameObject가 비활성화됨: {gameObject.name}");
            return;
        }
        
        // if (udpReceiver.keypoints_dataReceived)
        // {
        //     ObjectData objdata = udpReceiver.keypoints.objects[0];
        //     HandleKeypointsData(objdata);
        //     if (udpReceiver.image_dataReceived)
        //     {
        //         HandleImageData(objdata);
        //     }
        // }
    }

    private void OnVideoEnd(VideoPlayer vp)
    {
        // Intro 영상이 끝났을 때만 바로 다음 단계로 넘어갑니다.
        if (currentSide == Side.Intro)
        {
            introPlayed = true;
            if (currentRoutine != null) StopCoroutine(currentRoutine);
            currentRoutine = StartCoroutine(RunPrepareThenExercise(Side.L));
            return;
        }

        if (video.isLooping)
        {
            return;
        }

        // --- 여기부터는 isLooping이 false일 때만 실행됩니다. ---
        
        //evaluatorstart = false; // 평가 중지
        if(currentSide == Side.L||currentSide == Side.R){
            SetPhase("REST"); // 운동 클립 종료 시점에 REST
            if(achivedText!=null) achivedText.gameObject.SetActive(false);
            if(circleFill!=null) circleFill.gameObject.SetActive(false);
        }
        
        // 평가자 상태를 깨끗하게 초기화합니다.
        if (rehabilitation != null)
        {
            rehabilitation.ResetState();
        }

        if (currentRoutine != null) StopCoroutine(currentRoutine);

        if (currentSide == Side.L)
        {
            if (alternatingSides)
            {
                currentRoutine = StartCoroutine(RunPrepareThenExercise(Side.R));
            }
            else
            {
                // 좌우 교대가 아니면 바로 다음 세트로
                currentSet++;
                if (currentSet <= setCount)
                {
                    setText.text = $"{currentSet} / {setCount} 세트";
                    currentRoutine = StartCoroutine(RestThenNextSet());
                }
                else
                {
                    EndExercise();
                }
            }
            return;
        }

        if (currentSide == Side.R)
        {
            currentSet++;
            if (currentSet <= setCount)
            {
                setText.text = $"{currentSet} / {setCount} 세트";
                currentRoutine = StartCoroutine(RestThenNextSet());
            }
            else
            {
                EndExercise();
            }
        }
    }

    // --- 버튼 이벤트 ---
    public void click_start()
    {
        countdownText.text = ""; // 안내 문구 지우기

        if (currentRoutine != null) StopCoroutine(currentRoutine);

        if (!introPlayed)
            currentRoutine = StartCoroutine(RunIntroThenExercise());
        else
            currentRoutine = StartCoroutine(RunPrepareThenExercise(Side.L));
    }

    public void click_pause()
    {
        if (video.isPlaying) video.Pause();
        evaluatorstart = false;
        if (currentRoutine != null) StopCoroutine(currentRoutine);
        SetPhase("REST");
    }

    public void RetryCurrentSet()
    {
        if(!readymotion){
            countdownText.text = "아직 첫 세트를 시작하지 않았습니다. 조금만 기다려주세요.";
            return;
        }
        if (currentRoutine != null) StopCoroutine(currentRoutine);
        
        RestoreSetStartCounts();

        //evaluatorstart = false;
        if (rehabilitation != null)
        {
            rehabilitation.ResetState();
        }

        // if(achivedText!=null) achivedText.gameObject.SetActive(false);
        // if(circleFill!=null) circleFill.gameObject.SetActive(false);
        
        // 현재 세트를 처음(L)부터 다시 시작
        currentRoutine = StartCoroutine(RunPrepareThenExercise(Side.L));

        setText.text = $"{currentSet} / {setCount} 세트";
    }

    private void SaveSetStartCounts()
    {
        // setStartPerfectCount = PlayerPrefs.GetInt("PerfectCount", 0);
        // setStartGoodCount = PlayerPrefs.GetInt("GoodCount", 0);
        // setStartNotgoodCount = PlayerPrefs.GetInt("NotgoodCount", 0);
        // setStartAchivedCount = PlayerPrefs.GetInt("AchivedCount", 0);
    }

    private void RestoreSetStartCounts()
    {
        // PlayerPrefs.SetInt("PerfectCount", setStartPerfectCount);
        // PlayerPrefs.SetInt("GoodCount", setStartGoodCount);
        // PlayerPrefs.SetInt("NotgoodCount", setStartNotgoodCount);
        // PlayerPrefs.SetInt("AchivedCount", setStartAchivedCount);
    }

    // --- 메인 시퀀스 코루틴 ---
    private IEnumerator RunIntroThenExercise()
    {
        yield return new WaitForSecondsRealtime(2f);
        currentSide = Side.Intro;
        video.isLooping = false; // 인트로 영상은 반복하지 않음
        video.Play();
        yield break; // Intro 영상이 끝나면 OnVideoEnd에서 이어짐
    }

    private IEnumerator RunPrepareThenExercise(Side side)
    {
        if (side == Side.L)
        {
            SaveSetStartCounts();
        }

        currentSide = side;
        LoadClip(side);

        // 1초 지점으로 이동해서 준비자세 미리보기
        yield return SeekToAndHold(1.0);
        yield return new WaitForSecondsRealtime(1.5f);
        SetPhase("REST");
        yield return GuideCountdown(10, "운동 준비 자세를 취해주세요");
        yield return new WaitForSecondsRealtime(1.5f);
        video.isLooping = false; // 운동 영상은 반복하지 않음
        
        // 비디오 재생 전 오디오 설정 재확인
        if (video.audioTrackCount > 0)
        {
            video.EnableAudioTrack(0, true);
            AudioSource audioSource = GetComponent<AudioSource>();
            if (audioSource != null)
            {
                audioSource.volume = 1.0f;
                audioSource.mute = false;
                UnityEngine.Debug.Log($"[Video] 재생 시작 - 오디오 강제 활성화");
            }
        }
        
        video.Play();
        SetPhase("ACTIVE");

        // if(achivedText!=null) achivedText.gameObject.SetActive(true);
        // if(circleFill!=null) circleFill.gameObject.SetActive(true);
        //evaluatorstart = true; // 평가 시작
    }

    private IEnumerator RestThenNextSet()
    {
        SetPhase("REST"); // 휴식 시작
        yield return MessageCountdown(60, "휴식 시간입니다.");
        currentRoutine = StartCoroutine(RunPrepareThenExercise(Side.L));
    }

    // 지정된 시간 t(초)로 이동해서 해당 프레임을 보이게 하고 Pause 상태로 유지
    private IEnumerator SeekToAndHold(double t)
    {
        if (!video.isPrepared)
        {
            video.Prepare();
            yield return new WaitUntil(() => video.isPrepared);
        }

        video.time = t;
        video.Play();

        // 실제로 재생 시점이 반영될 때까지 대기
        yield return new WaitUntil(() => video.time >= t - 0.02f);

        yield return null; // 1프레임 대기 → 확실히 반영
        video.Pause();
    }

    private void EndExercise()
    {
        SetPhase("REST");
        setText.text = "Completed!";
        countdownText.text = "";
        //StopPython();
        //sceneChange.toResult();
    }

    // --- 유틸리티 코루틴 ---
    private IEnumerator GuideCountdown(int seconds, string message)
    {
        string almostMsg = "곧 시작합니다";
        int s = Mathf.Max(0, seconds);
        while (s > 0)
        {
            string header = (s <= 3) ? almostMsg : message;
            countdownText.text = $"{header}\n{s}";
            yield return new WaitForSecondsRealtime(1f);
            s--;
        }
        countdownText.text = "";
    }

    private IEnumerator MessageCountdown(int seconds, string message)
    {
        int s = Mathf.Max(0, seconds);
        while (s > 0)
        {
            countdownText.text = $"{message}\n{s}";
            yield return new WaitForSecondsRealtime(1f);
            s--;
        }
        countdownText.text = "";
    }

    // --- 헬퍼 함수 ---
    private bool LoadClip(Side side)
    {
        string type = PlayerPrefs.GetString("Type", "");
        if (string.IsNullOrEmpty(exerciseId) || string.IsNullOrEmpty(type))
        {
            UnityEngine.Debug.LogError("[Video] exerciseId나 Type이 PlayerPrefs에 없습니다.");
            return false;
        }

        string suffix = "";
        switch (side)
        {
            case Side.Intro: suffix = "_intro"; break;
            case Side.L: suffix = "_left"; break;
            case Side.R: suffix = "_right"; break;
        }

        string path = $"Video/{type}/{exerciseId}{suffix}";
        var clip = Resources.Load<VideoClip>(path);

        if (clip == null)
        {
            UnityEngine.Debug.LogError($"[Video] 비디오 클립을 찾을 수 없음: Resources/{path}");
            return false;
        }

        video.Stop();
        video.clip = clip;
        SharedPhase.CurrentVideoName = clip.name;
        video.time = 0;
        video.Pause();
        video.Prepare();

        // ✅ 여기서 “파일명 → 한글 이름” 매핑
        // key는 네가 예로 든 그대로: "3_10_left", "3_11_left" 등
        string key = $"{exerciseId}{suffix}";   // 예) "3_10_left"
        if (!LogNameMap.TryGetValue(key, out var logName))
        {
            // 매핑 안 되어 있으면 기본은 원래 키 또는 clip.name 사용
            logName = key;  // 또는 logName = clip.name;
        }

        // 공용 상태에 저장 → 다른 스크립트에서 파일 이름으로 사용
        SharedPhase.CurrentVideoLogName = logName;

        UnityEngine.Debug.Log($"[Video] 현재 영상 키: {key}, 로그 파일 이름: {logName}");


        
        // 비디오 클립 로드 후 오디오 설정 재확인
        if (video.audioTrackCount > 0)
        {
            video.EnableAudioTrack(0, true);
            
            // AudioSource 상태 재확인
            AudioSource audioSource = GetComponent<AudioSource>();
            if (audioSource != null)
            {
                UnityEngine.Debug.Log($"[Video] AudioSource 상태 - Volume: {audioSource.volume}, Mute: {audioSource.mute}, Enabled: {audioSource.enabled}");
            }
            
            UnityEngine.Debug.Log($"[Video] 비디오 클립 로드됨 - 오디오 트랙 수: {video.audioTrackCount}, 오디오 출력 모드: {video.audioOutputMode}");
        }
        else
        {
            UnityEngine.Debug.LogWarning($"[Video] 비디오 클립에 오디오 트랙이 없습니다: {path}");
        }
        
        return true;
    }

    // #region 로직 (UDP, Python, 실루엣)
    // private void HandleKeypointsData(ObjectData objdata)
    // {
    //     if (objdata == null || objdata.GetAllPositions() == null) return;

    //     // 1. 사용자가 준비 자세를 취할 때까지 대기
    //     if (!readymotion)
    //     {
    //         // 첫 이미지 수신 후 딜레이가 지나야 준비 자세 체크를 시작 (실루엣 모드만 해당)
    //         if (!useSilhouetteCheck || (firstFrameTimestamp > 0f && Time.time - firstFrameTimestamp >= READY_CHECK_DELAY))
    //         {
    //             if (useSilhouetteCheck) UpdateReadyStatusWithSilhouette(objdata);
    //             else UpdateReadyStatusSimple(objdata);
    //         }

    //         // 준비 자세가 방금 완료되었다면, 자동으로 운동 시작
    //         if (readymotion)
    //         {
    //             click_start();
    //         }
    //         return; // 준비 단계에서는 평가 로직을 실행하지 않음
    //     }

    //     // 2. 사용자가 준비되었고, 운동이 시작되면 평가자(evaluator)를 한 번만 초기화
    //     if (evaluatorinit == false)
    //     {
    //         // 운동이 시작되었으므로 일시정지 버튼 활성화
    //         //pausebutton.SetActive(true);

    //         string exerciseName = PlayerPrefs.GetString("exerciseName", "");
    //         rehabilitation = EvaluatorSelector.CreateEvaluator(exerciseName, objdata, setCount, achivedText, circleFill);
    //         PlayerPrefs.SetInt("TotalSet", setCount);
    //         rehabilitation.StartEvaluation();
    //         evaluatorinit = true;
    //     }
        
    //     // 3. 평가가 시작되면(evaluatorstart == true), 실시간으로 자세 평가
    //     if (evaluatorstart)
    //     {
    //         // 평가 직전에 현재 영상에 맞는 평가 방향을 설정합니다.
    //         if (rehabilitation is BaseRehabilitationEvaluator baseEvaluator)
    //         {
    //             EvaluationSide evalSide = (currentSide == Side.L) ? EvaluationSide.Left : EvaluationSide.Right;
    //             baseEvaluator.SetEvaluationSide(evalSide);
    //         }

    //         rehabilitation.UpdateData(objdata);
    //         rehabilitation.RealtimeEvaluate(currentSet);
    //         udpReceiver.keypoints_dataReceived = false;
    //     }
    // }

    // private void HandleImageData(ObjectData objdata)
    // {
    //     if (udpReceiver.image_data != null && udpReceiver.image_data.Length > 0)
    //     {
    //         if (receivedTexture.LoadImage(udpReceiver.image_data))
    //         {
    //             raw.texture = receivedTexture;

    //             // 첫 이미지 수신 시, 초기 안내 메시지 표시 및 딜레이 타이머 시작
    //             if (useSilhouetteCheck && !initialMessageDisplayed)
    //             {
    //                 UnityEngine.Debug.Log("첫 이미지 수신 시, 초기 안내 메시지 표시");
    //                 countdownText.text = "표시된 위치에 바르게 서주세요.";
    //                 initialMessageDisplayed = true;
    //                 firstFrameTimestamp = Time.time; // 타이머 시작
    //             }

    //             if (useSilhouetteCheck && !readymotion)
    //             {
    //                 if (invertedMaskMaterial != null)
    //                 {
    //                     rawmask.material = invertedMaskMaterial;
    //                     rawmask.enabled = true;
    //                 }
    //             }
    //         }
    //     }
    // }

    // private void UpdateReadyStatusWithSilhouette(ObjectData objdata)
    // {
    //     if (humanSilhouetteForCheck == null || invertedMaskMaterial == null) return;
    //     var allPositions = objdata.GetAllPositions();
    //     if (allPositions == null || !allPositions.Any()) return;

    //     Vector2 maskScale = invertedMaskMaterial.GetVector("_MaskScale");
    //     Vector2 maskOffset = invertedMaskMaterial.GetVector("_MaskOffset");
    //     if (maskScale.x == 0) maskScale.x = 1f;
    //     if (maskScale.y == 0) maskScale.y = 1f;

    //     var keypointPositions = objdata.GetPositionsAsDictionary();
    //     int outsideCount = 0;
    //     foreach (var keypointPair in keypointPositions)
    //     {
    //         float[] objJoint = keypointPair.Value;
    //         if (objJoint == null || objJoint.Length < 2) continue;
            
    //         float u = objJoint[0] / 640f;
    //         float v = 1.0f - (objJoint[1] / 720f);
    //         Vector2 transformedUV = new Vector2((u - 0.5f) / maskScale.x - maskOffset.x + 0.5f, (v - 0.5f) / maskScale.y - maskOffset.y + 0.5f);

    //         int texX = (int)(transformedUV.x * humanSilhouetteForCheck.width);
    //         int texY = (int)(transformedUV.y * humanSilhouetteForCheck.height);
            
    //         if (texX < 0 || texX >= humanSilhouetteForCheck.width || texY < 0 || texY >= humanSilhouetteForCheck.height || humanSilhouetteForCheck.GetPixel(texX, texY).a < 0.5f)
    //         {
    //             outsideCount++;
    //         }
    //     }

    //     bool allPointsInside = (outsideCount <= 1);
    //     if (allPointsInside && allPositions.Count > 10)
    //     {
    //         readymotion = true;
    //         if (rawmask != null) rawmask.enabled = false;
    //     }
    //     if (rawmask.enabled)
    //     {
    //         invertedMaskMaterial.SetColor("_OutlineColor", allPointsInside ? outlineReadyColor : outlineDefaultColor);
    //     }
    // }

    // private void UpdateReadyStatusSimple(ObjectData objdata)
    // {
    //     var allPositions = objdata.GetAllPositions();
    //     if (allPositions != null && allPositions.Count > 10 && allPositions.All(pos => pos != null && pos.Length > 2 && pos[2] >= 0.5f))
    //     {
    //         readymotion = true;
    //     }
    // }



    private void OnDestroy() 
    {
        isDestroyed = true;
        UnityEngine.Debug.Log($"[Video] OnDestroy 호출됨 - GameObject: {gameObject.name}, Scene: {gameObject.scene.name}");
        
        // Python 프로세스 종료
        //StopPython();
        
        // 싱글톤 인스턴스 정리
        if (Instance == this)
        {
            Instance = null;
            UnityEngine.Debug.Log("[Video] 싱글톤 인스턴스가 정리되었습니다.");
        }
        
        // 코루틴 정리
        if (currentRoutine != null)
        {
            StopCoroutine(currentRoutine);
            currentRoutine = null;
        }
        
        // 평가자 정리
        if (rehabilitation != null)
        {
            rehabilitation.ResetState();
            rehabilitation = null;
        }
        
        UnityEngine.Debug.Log("[Video] OnDestroy 완료");
    }

    private void OnApplicationQuit()
    {
        UnityEngine.Debug.Log("[Video] OnApplicationQuit 호출됨");
        //StopPython();
    }

    // public static void StartPython()
    // {
    //     if (pythonProcess != null && !pythonProcess.HasExited) 
    //     {
    //         UnityEngine.Debug.Log("[Video] Python 프로세스가 이미 실행 중입니다.");
    //         return;
    //     }
        
    //     // 기존 프로세스가 있다면 정리 후 재시작
    //     if (pythonProcess != null)
    //     {
    //         StopPython();
    //     }
        
    //     string projectPath = Application.dataPath.Replace("/Assets", "").Replace("\\Assets", "");
    //     string scriptPath = System.IO.Path.Combine(projectPath, "Python", "vitpose", "inference.py");
    //     ProcessStartInfo startInfo = new ProcessStartInfo
    //     {
    //         FileName = "C:/Users/user/anaconda3/envs/vitpose/python.exe",
    //         Arguments = $"\"{scriptPath}\"",
    //         UseShellExecute = false,
    //         RedirectStandardOutput = true,
    //         RedirectStandardError = true,
    //         CreateNoWindow = true
    //     };
    //     pythonProcess = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
    //     pythonProcess.OutputDataReceived += (sender, args) => { 
    //         if (!string.IsNullOrEmpty(args.Data)) 
    //             UnityEngine.Debug.Log($"[Python] {args.Data}"); 
    //     };
    //     pythonProcess.ErrorDataReceived += (sender, args) => { 
    //         if (!string.IsNullOrEmpty(args.Data)) 
    //             UnityEngine.Debug.LogError($"[Python Error] {args.Data}"); 
    //     };
    //     try
    //     {
    //         pythonProcess.Start();
    //         pythonProcess.BeginOutputReadLine();
    //         pythonProcess.BeginErrorReadLine();
    //         UnityEngine.Debug.Log("[Video] Python 프로세스 시작 성공");
    //     }
    //     catch (System.Exception ex)
    //     {
    //         UnityEngine.Debug.LogError($"Python 스크립트 실행 오류: {ex.Message}");
    //         pythonProcess = null;
    //     }
    // }

    // public static void StopPython()
    // {
    //     if (pythonProcess == null)
    //     {
    //         UnityEngine.Debug.Log("[Video] 종료할 Python 프로세스가 없습니다.");
    //         return;
    //     }
        
    //     try
    //     {
    //         if (!pythonProcess.HasExited)
    //         {
    //             UnityEngine.Debug.Log("[Video] Python 프로세스 종료 시작");
    //             pythonProcess.Kill();
    //             pythonProcess.WaitForExit(2000); // 최대 2초 대기
    //             UnityEngine.Debug.Log("[Video] Python 프로세스 종료 완료");
    //         }
    //         pythonProcess.Close();
    //     }
    //     catch (System.Exception ex)
    //     {
    //         UnityEngine.Debug.LogError($"Python 스크립트 중지 오류: {ex.Message}");
    //     }
    //     finally
    //     {
    //         pythonProcess = null;
    //     }
    // }
    // #endregion
}