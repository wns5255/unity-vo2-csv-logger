using System;
using System.Collections.Generic;
using System.Net.NetworkInformation;
using UnityEngine;
using TMPro;


public class SleeperStretch : BaseRehabilitationEvaluator
{
    // 운동 단계를 정의합니다.
    private enum ExerciseState
    {
        Idle,       // 운동 시작 대기 (및 1회 완료 후 대기)
        Holding,    // 목표 자세에서 3초 유지
        Returning,   // 시작 자세로 복귀 대기
        Rest,
        RestHolding
    }

    private ExerciseState currentState = ExerciseState.Idle;
    private float holdTimer = 0f;
    private float holdTimer_2=0f;
    private float idleTimer = 0f;
    private float idleTimer_2=0f;
    // --- 운동 설정값 ---
    private const float HOLD_DURATION = 30f;         // 자세 유지 시간
    private const float POSITION_TOLERANCE = 0.2f;  // 위치 허용 오차 (값이 작을수록 정확해야 함)
    
    // --- 목표 좌표 ---
    private readonly Vector2 elbowStartPosition = new Vector2(0f, -0.5f);

    // --- 상태 관리 변수 ---
    private bool holdMessageDisplayed = false;
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

    public SleeperStretch(ObjectData objdata, int totalset,TMP_Text achivedText,CircleFill circleFill)
        : base(objdata, totalset,achivedText,circleFill)
    {
    }

    public override void ResetState()
    {
        base.ResetState(); // 부모 클래스의 기본 초기화 로직을 먼저 실행

        // SleeperStretch만의 고유한 상태 변수들을 초기화
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
    }

    public override void RealtimeEvaluate(int currentset)
    {
        // 좌/우가 변경되었는지 확인하고 화면 표시용 카운터만 리셋합니다.
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
        
        Vector2 currentElbowPos = new Vector2(armData.elbow[0], armData.elbow[1]);
        Vector2 currentWristPos = new Vector2(armData.wrist[0], armData.wrist[1]);
        Vector2 currentElbowPos_opposite = new Vector2(armData_opposite.elbow[0], armData_opposite.elbow[1]);
        Vector2 currentWristPos_opposite = new Vector2(armData_opposite.wrist[0], armData_opposite.wrist[1]);
        Vector2 currentKneePos = new Vector2(legData.knee[0], legData.knee[1]);
        Vector2 currentKneePos_opposite = new Vector2(legData_opposite.knee[0], legData_opposite.knee[1]);
        Vector2 currentAnklePos = new Vector2(legData.ankle[0], legData.ankle[1]);
        Vector2 currentAnklePos_opposite = new Vector2(legData_opposite.ankle[0], legData_opposite.ankle[1]);
        
        float distance=0f;
        float accuracyRatio=0f;
        // 1. 운동 시작을 위한 초기 움직임 감지
        if (!ismove)
        {
            wristPosQueue.Enqueue(currentWristPos);
            if (wristPosQueue.Count > maxQueueSize) wristPosQueue.Dequeue();
            
            if (wristPosQueue.Count == maxQueueSize)
            {
                if (Vector2.Distance(wristPosQueue.Peek(), currentWristPos) > 0.05f)
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
            
            if (starttimer > 5f)
            {
                 MessageManager.Instance.DisplayMessage("동작을 시작해주세요.");
                 starttimer = 0f;
                 idleTimer_2=0f;
                 idleTimer=0f;
            }
            return;
        }

        // --- 목표 및 복귀 위치 설정 ---
        Vector2 elbowTargetPosition;
        Vector2 wristTargetPosition;
        Vector2 elbowTargetPosition_opposite;
        Vector2 wristTargetPosition_opposite;
        Vector2 wristReturnPosition;

        float angleRad_25 = 25f * Mathf.Deg2Rad;
        float cos25 = Mathf.Cos(angleRad_25);
        float sin25 = Mathf.Sin(angleRad_25);
        float angleRad_60 = 60f * Mathf.Deg2Rad;
        float cos60 = Mathf.Cos(angleRad_60);
        float sin60 = Mathf.Sin(angleRad_60);

        if (currentEvaluationSide == EvaluationSide.Left)
        {
            // 운동 기준
            elbowTargetPosition = new Vector2(0.2f, -0.1f);
            wristTargetPosition = new Vector2(currentElbowPos.x + 0.5f * cos25, currentElbowPos.y + 0.5f * sin25);
            elbowTargetPosition_opposite = new Vector2(0.5f, 0f);
            wristTargetPosition_opposite = new Vector2(0.5f, -0.3f);

            // 기본 준비 자세
            wristReturnPosition = new Vector2(0f, 0.3f);
        }
        else // EvaluationSide.Right
        {
            // 오른쪽 운동 기준
            elbowTargetPosition = new Vector2(-0.2f, -0.1f);
            wristTargetPosition = new Vector2(currentElbowPos.x + -0.5f * cos25, currentElbowPos.y + 0.5f * sin25);
            elbowTargetPosition_opposite = new Vector2(-0.5f, 0f);
            wristTargetPosition_opposite = new Vector2(-0.5f, -0.3f);

            // 기본 준비 자세
            wristReturnPosition = new Vector2(0f, 0.3f);
        }
        if (lastUpdateTime == 0f) lastUpdateTime = Time.time;
        float realDeltaTime = Time.time - lastUpdateTime;
        lastUpdateTime = Time.time;
        idleTimer_2 += realDeltaTime;
        // --- 현재 자세 정확도 판단 (팔) ---
        bool isTargetPoseMaintained = 
            Vector2.Distance(currentElbowPos, elbowTargetPosition) < 0.25f &&
            Vector2.Distance(currentWristPos, wristTargetPosition) < 0.25f &&
            Vector2.Distance(currentElbowPos_opposite, elbowTargetPosition_opposite) < 0.25f &&
            Vector2.Distance(currentWristPos_opposite, wristTargetPosition_opposite) < 0.25f;
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
            if(currentWristPos_opposite.y>POSITION_TOLERANCE){
                if (currentEvaluationSide == EvaluationSide.Left){
                    MessageManager.Instance.DisplayMessage("오른팔을 바로 해주세요.");
                }
                else{
                    MessageManager.Instance.DisplayMessage("왼팔을 바로 해주세요.");
                }
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
                    MessageManager.Instance.DisplayMessage("30초간 유지하세요.");
                    holdTimer = 0f;
                    idleTimer = 0f; // 자세를 잡았으므로 타이머 초기화
                    currentState = ExerciseState.Holding;
                    hasreached=true;
                }
                else if(!hasreached)
                {
                    idleTimer += realDeltaTime;
                    holdTimer+=realDeltaTime;
                    if (idleTimer > 4.0f)
                    {
                        if (currentEvaluationSide == EvaluationSide.Left)
                        {
                            if (Mathf.Abs(currentWristPos.x) <= Mathf.Abs(new Vector2(currentElbowPos.x + 0.5f * cos60, currentElbowPos.y + 0.5f * sin60).x) &&
                                currentWristPos.x >= 0 && 
                                currentWristPos_opposite.y <= 0) 
                            {
                                MessageManager.Instance.DisplayMessage("팔을 더 내려주세요.");
                                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                                PlayerPrefs.SetInt("BadCount", badCount);
                                idleTimer = 0f;
                            }
                            else
                            {
                                //FeedbackImageManager.Instance.ShowBadImage();
                                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                                PlayerPrefs.SetInt("BadCount", badCount);
                                idleTimer = 0f;
                            }
                        }
                        else if (holdTimer > HOLD_DURATION - 1.0f)
                        {
                            idleTimer = 0f;
                            MessageManager.Instance.DisplayMessage("영상을 따라 준비자세로 복귀하세요.");
                            currentState = ExerciseState.Rest;
                        }
                        else if(currentEvaluationSide == EvaluationSide.Right)
                        {
                            if (Mathf.Abs(currentWristPos.x) <= Mathf.Abs(new Vector2(currentElbowPos.x + 0.5f * cos60, currentElbowPos.y - 0.5f * sin60).x) &&
                                currentWristPos.x <= 0 && 
                                currentWristPos_opposite.y <= 0) 
                            {
                                MessageManager.Instance.DisplayMessage("팔을 더 내려주세요.");
                                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                                PlayerPrefs.SetInt("BadCount", badCount);
                                idleTimer = 0f;
                            }
                            else
                            {
                                // FeedbackImageManager.Instance.ShowBadImage();
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
                idleTimer+=realDeltaTime;
                distance = (Vector2.Distance(currentElbowPos, elbowTargetPosition)
                +Vector2.Distance(currentWristPos, wristTargetPosition)
                +Vector2.Distance(currentElbowPos_opposite, elbowTargetPosition_opposite)
                +Vector2.Distance(currentWristPos_opposite, wristTargetPosition_opposite)
                )/4f;
                accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                if (holdTimer >= HOLD_DURATION-1.0f)
                {
                    MessageManager.Instance.DisplayMessage("영상을 따라 준비자세로 복귀하세요.");
                    currentState = ExerciseState.Rest;
                    starttimer = 0f;
                    startTimerLastUpdate = 0f; // 타이머 초기화
                    wristPosQueue.Clear();
                    ismove=false; // 다음 동작 감지를 위해 false로 초기화
                }
                else if(holdTimer<HOLD_DURATION-3.0f&&!isTargetPoseMaintained)
                {
                    MessageManager.Instance.DisplayMessage("자세를 유지해주세요");
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    idleTimer+=realDeltaTime;
                }
                break;
            case ExerciseState.Rest:
                if(ismove){
                    currentState=ExerciseState.RestHolding;
                }
                else if(isTargetPoseMaintained)
                {
                    holdTimer += realDeltaTime;
                }
                break;

            case ExerciseState.RestHolding:
                holdTimer_2 += realDeltaTime;
                distance = Vector2.Distance(currentWristPos, wristReturnPosition);
                accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                if (holdTimer_2 >= HOLD_DURATION+1.0f)
                {
                    currentState = ExerciseState.Returning;
                }
                break;
            case ExerciseState.Returning:
                if(calculateScore(holdTimer+holdTimer_2)>80){
                    FeedbackImageManager.Instance.ShowPerfectImage();
                    int perfectCount = PlayerPrefs.GetInt("PerfectCount", 0) + 1;
                    PlayerPrefs.SetInt("PerfectCount", perfectCount);
                }
                else if(calculateScore(holdTimer+holdTimer_2)>60){
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
                holdTimer_2=0f;
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
                
                break;
        }
    }
    private float calculateScore(float holdTimer)
    {
        Debug.Log("holdTimer"+holdTimer);
        float holdduration=HOLD_DURATION*2;
        holdTimer = (holdTimer - holdduration) < -0.5f ? holdTimer + 0.5f : (holdTimer - holdduration) > 1.0f ? holdTimer - 1.0f : holdTimer;
        float timerScore = (holdduration-Math.Abs(holdTimer - holdduration))/holdduration*100f;
        float poseScore = (poseScoreSampleCount>0)?(totalPoseScore / poseScoreSampleCount):0f;

        float averageScore = (timerScore + poseScore) / 2f;

        // 4. 감점 계산
        int badCount = PlayerPrefs.GetInt("BadCount", 0);
        float badPenalty = badCount * 5;
        Debug.Log("TimeScore"+timerScore+"PoseScore"+poseScore+"BadPenalty"+badPenalty);

        float finalScore = averageScore - badPenalty;
        Debug.Log("Score"+finalScore);

        return Mathf.Max(0, finalScore);
    }
}

