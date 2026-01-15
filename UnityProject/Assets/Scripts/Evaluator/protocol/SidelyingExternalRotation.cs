using System;
using System.Collections.Generic;
using System.Net.NetworkInformation;
using UnityEngine;
using TMPro;


public class SidelyingExternalRotation : BaseRehabilitationEvaluator
{
    // 운동 단계를 정의합니다.
    private enum ExerciseState
    {
        Idle,       // 운동 시작 대기 (및 1회 완료 후 대기)
        Holding,    // 목표 자세에서 3초 유지
        Returning   // 시작 자세로 복귀 대기
    }

    private ExerciseState currentState = ExerciseState.Idle;
    private float holdTimer = 0f;
    private float idleTimer = 0f;
    private float idleTimer_2=0f;
    
    
    // --- 운동 설정값 ---
    private const float HOLD_DURATION = 5f;         // 자세 유지 시간
    private const float POSITION_TOLERANCE = 0.2f;  // 위치 허용 오차 (값이 작을수록 정확해야 함)  

    // --- 상태 관리 변수 ---
    private bool hasreached=false;
    private float lastUpdateTime = 0f;
    private float startTimerLastUpdate = 0f;
    private int completedReps = 0; // 완료 횟수 추적 변수
    private EvaluationSide lastEvaluatedSide = EvaluationSide.None; // 좌/우 변경 감지를 위한 변수


    // --- 초기 움직임 감지용 변수 ---
    private Queue<Vector2> wristPosQueue = new Queue<Vector2>();
    private int maxQueueSize = 10;
    private bool ismove = false;
    private float starttimer = 0f;

    private float totalPoseScore = 0f;
    private float poseScoreSampleCount = 0f;

    public SidelyingExternalRotation(ObjectData objdata, int totalset,TMP_Text achivedText,CircleFill circleFill)
        : base(objdata, totalset,achivedText,circleFill)
    {
    }

    public override void ResetState()
    {
        base.ResetState(); // 부모 클래스의 기본 초기화 로직을 먼저 실행

        currentState = ExerciseState.Idle;
        holdTimer = 0f;
        ismove = false;
        starttimer = 0f;
        startTimerLastUpdate = 0f;
        wristPosQueue.Clear();
        completedReps = 0;
        idleTimer = 0f;
        totalPoseScore = 0f;
        poseScoreSampleCount = 0f;
        hasreached=false;
        lastEvaluatedSide = EvaluationSide.None;
    }

    public override void RealtimeEvaluate(int currentset)
    {
        if (currentEvaluationSide != lastEvaluatedSide)
        {
            completedReps = 0;
            lastEvaluatedSide = currentEvaluationSide;
            int Num = PlayerPrefs.GetInt("Num", 0);
            if (circleFill != null)
            {
                circleFill.value = completedReps;
                circleFill.fillValue = ((Num / 2) > 0) ? (float)completedReps / (Num / 2) : 0;
            }
        }
        // --- 설정된 방향에 따라 평가할 팔의 데이터를 선택 ---
        var armData = (currentEvaluationSide == EvaluationSide.Left) ? normalizedLeftArm : normalizedRightArm;
        var armData_opposite = (currentEvaluationSide == EvaluationSide.Left) ? normalizedRightArm : normalizedLeftArm;
        var legData = (currentEvaluationSide == EvaluationSide.Left) ? normalizedLeftLeg : normalizedRightLeg;
        var legData_opposite = (currentEvaluationSide == EvaluationSide.Left) ? normalizedRightLeg : normalizedLeftLeg;
        var bodyData = normalizedBody;
        
        Vector2 currentElbowPos = new Vector2(armData.elbow[0], armData.elbow[1]);
        Vector2 currentWristPos = new Vector2(armData.wrist[0], armData.wrist[1]);
        Vector2 currentElbowPos_opposite = new Vector2(armData_opposite.elbow[0], armData_opposite.elbow[1]);
        Vector2 currentWristPos_opposite = new Vector2(armData_opposite.wrist[0], armData_opposite.wrist[1]);
        Vector2 currentKneePos = new Vector2(legData.knee[0], legData.knee[1]);
        Vector2 currentKneePos_opposite = new Vector2(legData_opposite.knee[0], legData_opposite.knee[1]);
        Vector2 currentAnklePos = new Vector2(legData.ankle[0], legData.ankle[1]);
        Vector2 currentAnklePos_opposite = new Vector2(legData_opposite.ankle[0], legData_opposite.ankle[1]);
        Vector2 currentHeadPos = new Vector2(bodyData.head[0], bodyData.head[1]);

        // 1. 운동 시작을 위한 초기 움직임 감지
        if (!ismove)
        {
            wristPosQueue.Enqueue(currentWristPos);
            if (wristPosQueue.Count > maxQueueSize) wristPosQueue.Dequeue();
            
            if (wristPosQueue.Count == maxQueueSize)
            {
                if (Vector2.Distance(wristPosQueue.Peek(), currentWristPos) > 0.1)
                {
                    ismove = true;
                    idleTimer_2=0f;
                    idleTimer=0f;
                    return;
                }
            }
            
            if (startTimerLastUpdate == 0f) startTimerLastUpdate = Time.time;
            float startTimerDelta = Time.time - startTimerLastUpdate;
            starttimer += startTimerDelta;
            startTimerLastUpdate = Time.time;
            
            if (starttimer > 4f)
            {
                 MessageManager.Instance.DisplayMessage("동작을 시작해주세요.");
                 starttimer = 0f;
                 idleTimer_2=0f;
                 idleTimer=0f;
            }
            return;
        }

        // --- 목표 및 복귀 위치 설정 ---
        Vector2 wristTargetPosition;
        Vector2 elbowReturnPosition;
        Vector2 wristReturnPosition;
        Vector2 kneeReturnPosition;
        Vector2 kneeReturnPosition_opposite;
        Vector2 ankleReturnPosition;
        Vector2 ankleReturnPosition_opposite;

        float angleRad_30 = 30f * Mathf.Deg2Rad;
        float cos30 = Mathf.Cos(angleRad_30);
        float sin30 = Mathf.Sin(angleRad_30);
        float angleRad_60 = 60f * Mathf.Deg2Rad;
        float cos60 = Mathf.Cos(angleRad_60);
        float sin60 = Mathf.Sin(angleRad_60);

        if (currentEvaluationSide == EvaluationSide.Left)
        {
            wristTargetPosition = new Vector2(-0.5f, 0.4f);

            // 기본 준비 자세
            elbowReturnPosition=new Vector2(-0.5f,0f);
            wristReturnPosition = new Vector2(-0.5f, -0.5f);
            kneeReturnPosition = new Vector2(-0.4f, -0.2f);
            kneeReturnPosition_opposite = new Vector2(-0.4f, 0f);
            ankleReturnPosition = new Vector2(-0.85f, -0.1f);
            ankleReturnPosition_opposite = new Vector2(-0.85f, 0.1f);
        }
        else // EvaluationSide.Right
        {
            wristTargetPosition = new Vector2(0.5f, 0.4f);

            // 기본 준비 자세
            elbowReturnPosition = new Vector2(0.5f,0f);
            wristReturnPosition = new Vector2(0.5f, -0.5f);
            kneeReturnPosition = new Vector2(0.4f, -0.2f);
            kneeReturnPosition_opposite = new Vector2(0.4f, 0f);
            ankleReturnPosition = new Vector2(0.85f, -0.1f);
            ankleReturnPosition_opposite = new Vector2(0.85f, 0.1f);
        }
        if (lastUpdateTime == 0f) lastUpdateTime = Time.time;
        float realDeltaTime = Time.time - lastUpdateTime;
        lastUpdateTime = Time.time;
        idleTimer_2 += realDeltaTime;
        // --- 현재 자세 정확도 판단 (팔) ---
        bool isTargetPoseMaintained = 
            Vector2.Distance(currentWristPos, wristTargetPosition) < POSITION_TOLERANCE;
        bool isReturnPoseReached =
            Vector2.Distance(currentWristPos, wristReturnPosition) < POSITION_TOLERANCE;
        
        bool isLegPoseCorrect = currentKneePos.y>POSITION_TOLERANCE||
                currentKneePos_opposite.y>POSITION_TOLERANCE||
                currentAnklePos.y>POSITION_TOLERANCE||
                currentAnklePos_opposite.y>POSITION_TOLERANCE;
        if(idleTimer_2>3.0f){ 
            if (isLegPoseCorrect)
            {
                MessageManager.Instance.DisplayMessage("다리를 바르게 해주세요.");
                // FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            if (Vector2.Distance(currentElbowPos, elbowReturnPosition) > POSITION_TOLERANCE)
            {
                MessageManager.Instance.DisplayMessage("팔꿈치를 옆구리에 고정하세요.");
                // FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }

            if(Math.Abs(currentElbowPos_opposite.x)- 0.3f > POSITION_TOLERANCE||Math.Abs(currentWristPos_opposite.x)-0.3f > POSITION_TOLERANCE){
                MessageManager.Instance.DisplayMessage("팔을 머리에 받쳐주세요.");
                // FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
        }
        switch (currentState)
        {
            case ExerciseState.Idle:
                if (isTargetPoseMaintained&&!hasreached)
                {
                    MessageManager.Instance.DisplayMessage("5초를 세며 천천히 내려주세요.");
                    currentState = ExerciseState.Holding;
                    holdTimer = 0f;
                    idleTimer = 0f; // 자세를 잡았으므로 타이머 초기화
                    hasreached=true;
                }
                else if(!hasreached)
                {
                    if (ismove)
                    {
                        idleTimer += realDeltaTime;
                        if (idleTimer > 4.0f)
                        {
                            if (Mathf.Abs(currentWristPos.y)<(Math.Abs(wristTargetPosition.y)-POSITION_TOLERANCE) &&Math.Abs(currentWristPos.y)>0&&Math.Abs(currentWristPos.x-wristTargetPosition.x)<POSITION_TOLERANCE) 
                            {
                                MessageManager.Instance.DisplayMessage("팔을 더 올려주세요.");
                                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                                PlayerPrefs.SetInt("BadCount", badCount);
                                idleTimer = 0f;
                            }
                            else
                            {
                                MessageManager.Instance.DisplayMessage("잘못된 행동이에요");
                                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                                PlayerPrefs.SetInt("BadCount", badCount);
                                idleTimer = 0f;
                            }
                        }
                    }
                }
                break;

            case ExerciseState.Holding:
                holdTimer += realDeltaTime;
                if (holdTimer<=HOLD_DURATION+2.0f&&holdTimer >= HOLD_DURATION-1.0f&&isReturnPoseReached)
                {
                    currentState = ExerciseState.Returning;
                }
                else if(holdTimer>=HOLD_DURATION+2.0f&&!isReturnPoseReached)
                {
                    MessageManager.Instance.DisplayMessage("조금 더 빨리 내려주세요.");
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    currentState = ExerciseState.Returning;
                }
                if(holdTimer<HOLD_DURATION-1.0f&&isReturnPoseReached){
                    MessageManager.Instance.DisplayMessage("조금 천천히 내려주세요.");
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    currentState = ExerciseState.Returning;
                }
                break;

            case ExerciseState.Returning:
                if (isReturnPoseReached)
                {
                    if(calculateScore(holdTimer)>80){
                        FeedbackImageManager.Instance.ShowPerfectImage();
                        int perfectCount = PlayerPrefs.GetInt("PerfectCount", 0) + 1;
                        PlayerPrefs.SetInt("PerfectCount", perfectCount);
                    }
                    else if(calculateScore(holdTimer)>60){
                        FeedbackImageManager.Instance.ShowGoodImage();
                        int goodCount = PlayerPrefs.GetInt("GoodCount", 0) + 1;
                        PlayerPrefs.SetInt("GoodCount", goodCount);
                    }
                    else{
                        FeedbackImageManager.Instance.ShowNotGoodImage();
                        int notgoodCount = PlayerPrefs.GetInt("NotgoodCount", 0) + 1;
                        PlayerPrefs.SetInt("NotgoodCount", notgoodCount);
                    }
                    completedReps++;
                    MessageManager.Instance.DisplayMessage($"{completedReps}회 완료");
                    int achivedCount = PlayerPrefs.GetInt("AchivedCount", 0) + 1;
                    PlayerPrefs.SetInt("AchivedCount", achivedCount);
                    Debug.Log("AchivedCount"+achivedCount);
                    currentState = ExerciseState.Idle;
                    holdTimer = 0f;
                    idleTimer = 0f;
                    totalPoseScore = 0f;
                    poseScoreSampleCount = 0f;
                    hasreached = false;
                    starttimer = 0f;
                    startTimerLastUpdate = 0f; // 타이머 초기화
                    wristPosQueue.Clear();
                    ismove=false; // 다음 동작 감지를 위해 false로 초기화
                    idleTimer_2=0f;
                    PlayerPrefs.SetInt("BadCount", 0);
                    int Num = PlayerPrefs.GetInt("Num", 0);
                    if (circleFill != null)
                    {
                        circleFill.value = completedReps;
                        circleFill.fillValue = ((Num/2) > 0) ? (float)completedReps / (Num/2) : 0;
                    }
                }
                break;
        }
    }
    private float calculateScore(float holdTimer)
    {
        Debug.Log("holdTimer"+holdTimer);
        holdTimer = (holdTimer - HOLD_DURATION) < -0.5f ? holdTimer + 0.5f : (holdTimer - HOLD_DURATION) > 1.0f ? holdTimer - 1.0f : holdTimer;
        float timerScore = (HOLD_DURATION-Math.Abs(holdTimer - HOLD_DURATION))/HOLD_DURATION*100f;

        // 4. 감점 계산
        int badCount = PlayerPrefs.GetInt("BadCount", 0);
        float badPenalty = badCount * 5;
        Debug.Log("TimeScore"+timerScore+"BadPenalty"+badPenalty);

        float finalScore = timerScore - badPenalty;
        Debug.Log("Score"+finalScore);

        return Mathf.Max(0, finalScore);
    }
}

