using System;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PassiveExternalRotaion : BaseRehabilitationEvaluator
{
    // 운동 단계를 정의합니다.
    private enum ExerciseState
    {
        Idle,       // 운동 시작 대기 (및 1회 완료 후 대기)
        Holding,    // 목표 자세에서 3초 유지
        Rest,
        RestHolding,
        Returning   // 시작 자세로 복귀 대기
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

    public PassiveExternalRotaion(ObjectData objdata, int totalset,TMP_Text achivedText,CircleFill circleFill)
        : base(objdata, totalset,achivedText,circleFill)
    {
    }

    public override void ResetState()
    {
        base.ResetState();

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
        var bodyData = normalizedBody;
        bool isReturnConditionMet;
        
        // --- 선택된 팔의 데이터를 사용하여 현재 좌표 계산 ---
        Vector2 currentElbowPos = new Vector2(armData.elbow[0], armData.elbow[1]);
        Vector2 currentWristPos = new Vector2(armData.wrist[0], armData.wrist[1]);
        Vector2 currentElbowPos_opposite = new Vector2(armData_opposite.elbow[0], armData_opposite.elbow[1]);
        Vector2 currentWristPos_opposite = new Vector2(armData_opposite.wrist[0], armData_opposite.wrist[1]);
        Vector2 currentKneePos = new Vector2(legData.knee[0], legData.knee[1]);
        Vector2 currentKneePos_opposite = new Vector2(legData_opposite.knee[0], legData_opposite.knee[1]);
        Vector2 currentAnklePos = new Vector2(legData.ankle[0], legData.ankle[1]);
        Vector2 currentAnklePos_opposite = new Vector2(legData_opposite.ankle[0], legData_opposite.ankle[1]);
        Vector2 currentHeadPos = new Vector2(bodyData.head[0], bodyData.head[1]);
        
        float distance=0f;
        float accuracyRatio=0f;
        // 1. 운동 시작을 위한 초기 움직임 감지
        if (!ismove)
        {
            wristPosQueue.Enqueue(currentWristPos);
            if (wristPosQueue.Count > maxQueueSize) wristPosQueue.Dequeue();
            
            if (wristPosQueue.Count == maxQueueSize)
            {
                if (Vector2.Distance(wristPosQueue.Peek(), currentWristPos) > 0.025f)
                {
                    ismove = true;
                    idleTimer_2=0f;
                    idleTimer=0f;
                    return; // 움직임이 감지되었으므로 즉시 다음 프레임에서 평가 시작
                }
            }
            
            // 실제 시간을 사용한 시작 대기 타이머
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
        float angleRad_45 = 45f * Mathf.Deg2Rad;
        float cos45 = Mathf.Cos(angleRad_45);
        float sin45 = Mathf.Sin(angleRad_45);

        Vector2 wristTargetPosition;
        Vector2 kneeReturnPosition;
        Vector2 kneeReturnPosition_opposite;
        Vector2 ankleReturnPosition;
        Vector2 ankleReturnPosition_opposite;
        Vector2 headReturnPosition;


        if (currentEvaluationSide == EvaluationSide.Left)
        {
            wristTargetPosition = new Vector2(-0.45f, -0.5f);
            isReturnConditionMet = currentWristPos.x >= -0.15f;

            kneeReturnPosition = new Vector2(0f, -0.5f);
            kneeReturnPosition_opposite = new Vector2(0f, -0.5f);
            ankleReturnPosition = new Vector2(0f, -1.0f);
            ankleReturnPosition_opposite = new Vector2(0f, -1.0f);
            headReturnPosition = new Vector2(0f, 1.0f);
        }
        else // EvaluationSide.Right
        {
            wristTargetPosition = new Vector2(0.45f, -0.5f);
            isReturnConditionMet = currentWristPos.x <= 0.15f;

            kneeReturnPosition = new Vector2(0f, -0.5f);
            kneeReturnPosition_opposite = new Vector2(0f, -0.5f);
            ankleReturnPosition = new Vector2(0f, -1.0f);
            ankleReturnPosition_opposite = new Vector2(0f, -1.0f);
            headReturnPosition = new Vector2(0f, 1.0f);
        }

        if (lastUpdateTime == 0f) lastUpdateTime = Time.time;
        float realDeltaTime = Time.time - lastUpdateTime;
        lastUpdateTime = Time.time;
        idleTimer_2 += realDeltaTime;
        // --- 현재 자세 정확도 판단 ---
        bool isTargetPoseMaintained = Vector2.Distance(currentWristPos, wristTargetPosition) < POSITION_TOLERANCE&&Math.Abs(currentWristPos_opposite.y)-0.5f<POSITION_TOLERANCE;
        bool isReturnPoseReached = isReturnConditionMet && Mathf.Abs(currentWristPos.y - wristTargetPosition.y) < POSITION_TOLERANCE;
        
        if(idleTimer_2>3.0f){
            if(Vector2.Distance(currentKneePos, kneeReturnPosition) > POSITION_TOLERANCE ||
            Vector2.Distance(currentAnklePos, ankleReturnPosition) > POSITION_TOLERANCE)
            {
                if (currentEvaluationSide == EvaluationSide.Left){
                    MessageManager.Instance.DisplayMessage("왼쪽 다리를 바로 해주세요.");
                }
                else{
                    MessageManager.Instance.DisplayMessage("오른쪽 다리를 바로 해주세요.");
                }
                // FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            if(Vector2.Distance(currentKneePos_opposite, kneeReturnPosition_opposite) > POSITION_TOLERANCE ||
            Vector2.Distance(currentAnklePos_opposite, ankleReturnPosition_opposite) > POSITION_TOLERANCE)
            {
                if (currentEvaluationSide == EvaluationSide.Left){
                    MessageManager.Instance.DisplayMessage("오른쪽 다리를 바로 해주세요.");
                }
                else{
                    MessageManager.Instance.DisplayMessage("왼쪽 다리를 바로 해주세요.");
                }
                // FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            // if(Vector2.Distance(currentHeadPos, headReturnPosition) > POSITION_TOLERANCE)
            // {
            //     MessageManager.Instance.DisplayMessage("상체를 바로 해주세요.");
            //     // FeedbackImageManager.Instance.ShowBadImage();
            //     int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
            //     PlayerPrefs.SetInt("BadCount", badCount);
            //     idleTimer_2 = 0f;
            // }
            if (Vector2.Distance(currentElbowPos, elbowStartPosition) > POSITION_TOLERANCE)
            {
                MessageManager.Instance.DisplayMessage("양팔 팔꿈치를 옆구리에 고정하세요.");
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
                    currentState = ExerciseState.Holding;
                    hasreached=true;
                    idleTimer = 0f;
                    holdTimer = 0f;
                }
                else if (!hasreached)
                {   
                    idleTimer += realDeltaTime;
                    holdTimer+=realDeltaTime;
                    if (idleTimer > 4.0f)
                    {
                        if(Math.Abs(currentWristPos.x)-0.5f<POSITION_TOLERANCE&&Math.Abs(currentWristPos.y)-0.5f<POSITION_TOLERANCE&&Math.Abs(currentWristPos_opposite.y)-0.5f<POSITION_TOLERANCE){
                            MessageManager.Instance.DisplayMessage("봉을 끝까지 밀어주세요.");
                        }
                        else if(Math.Abs(currentWristPos.y)-0.5f>POSITION_TOLERANCE&&Math.Abs(currentWristPos_opposite.y)-0.5f>POSITION_TOLERANCE){
                            MessageManager.Instance.DisplayMessage("봉을 올바른 위치에 두세요.");
                        }
                        else{
                            MessageManager.Instance.DisplayMessage("잘못된 행동이에요.");
                        }
                        int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                        PlayerPrefs.SetInt("BadCount", badCount);
                        idleTimer = 0f;
                    }
                    else if(holdTimer>HOLD_DURATION-1.0f){
                        idleTimer = 0f;
                        MessageManager.Instance.DisplayMessage("영상을 따라 준비자세로 복귀하세요.");
                        currentState = ExerciseState.Rest;
                    }
                }
                break;
                

            case ExerciseState.Holding:
                holdTimer += realDeltaTime;
                idleTimer+=realDeltaTime;
                distance = (Vector2.Distance(currentWristPos, wristTargetPosition)+Math.Abs(currentWristPos_opposite.y)-0.5f)/2f;
                accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                if (holdTimer>=HOLD_DURATION-1.0f)
                {
                    MessageManager.Instance.DisplayMessage("영상을 따라 준비자세로 복귀하세요.");
                    currentState = ExerciseState.Rest;
                    starttimer = 0f;
                    startTimerLastUpdate = 0f; // 타이머 초기화
                    wristPosQueue.Clear();
                    ismove=false; // 다음 동작 감지를 위해 false로 초기화
                }
                else if(holdTimer<HOLD_DURATION-3.0f&&Vector2.Distance(currentWristPos, wristTargetPosition) >= POSITION_TOLERANCE&&idleTimer>4.0f)
                {
                    if (currentEvaluationSide == EvaluationSide.Left){
                        MessageManager.Instance.DisplayMessage("왼팔을 조금 더 밀어주세요.");
                    }
                    else{
                        MessageManager.Instance.DisplayMessage("오른팔을 조금 더 밀어주세요.");
                    }
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    idleTimer = 0f;
                }
                else if(holdTimer<HOLD_DURATION-3.0f&&Math.Abs(currentWristPos_opposite.y+0.5f)>POSITION_TOLERANCE&&idleTimer>4.0f)
                {
                    if (currentEvaluationSide == EvaluationSide.Left){
                        MessageManager.Instance.DisplayMessage("오른팔을 바로 해주세요.");
                    }
                    else{
                        MessageManager.Instance.DisplayMessage("왼팔을 바로 해주세요.");
                    }
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    idleTimer = 0f;
                }
                break;

            case ExerciseState.Rest:
                if(ismove){
                    currentState=ExerciseState.RestHolding;
                }
                else if(isTargetPoseMaintained){
                    holdTimer += realDeltaTime;
                }
                break;
            case ExerciseState.RestHolding:
                holdTimer_2 += realDeltaTime;
                idleTimer+=realDeltaTime;
                distance = Mathf.Abs(currentWristPos.y - wristTargetPosition.y);
                accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                if (holdTimer_2 >= HOLD_DURATION+1.0f)
                {
                    currentState = ExerciseState.Returning;
                }
                else if(holdTimer_2<HOLD_DURATION-3.0f&&!isReturnConditionMet&&idleTimer>4.0f)
                {
                    if (currentEvaluationSide == EvaluationSide.Left){
                        MessageManager.Instance.DisplayMessage("왼팔을 안으로 조금 더 넣어주세요.");
                    }
                    else{
                        MessageManager.Instance.DisplayMessage("오른팔을 안으로 조금 더 넣어주세요.");
                    }
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    idleTimer = 0f;
                }
                else if(holdTimer_2<HOLD_DURATION-3.0f&&Math.Abs(currentWristPos_opposite.y+0.5f)>POSITION_TOLERANCE&&idleTimer>4.0f)
                {
                    if (currentEvaluationSide == EvaluationSide.Left){
                        MessageManager.Instance.DisplayMessage("오른팔을 바로 해주세요.");
                    }
                    else{
                        MessageManager.Instance.DisplayMessage("왼팔을 바로 해주세요.");
                    }
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                    idleTimer = 0f;
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
                currentState = ExerciseState.Idle;
                holdTimer = 0f;
                idleTimer = 0f;
                holdTimer_2=0f;
                totalPoseScore = 0f;
                poseScoreSampleCount = 0f;
                hasreached=false;
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

