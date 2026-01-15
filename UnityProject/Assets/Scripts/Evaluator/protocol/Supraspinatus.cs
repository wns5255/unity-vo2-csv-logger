using System;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class Supraspinatus : BaseRehabilitationEvaluator
{
    // --- 다른 스크립트와 통신하기 위한 이벤트 ---

    
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
    private const float HOLD_DURATION = 3f;         // 자세 유지 시간
    private const float POSITION_TOLERANCE = 0.3f;  // 위치 허용 오차 (값이 작을수록 정확해야 함)

    // --- 상태 관리 변수 ---
    private float totalPoseScore = 0f;
    private float poseScoreSampleCount = 0f;
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

    public Supraspinatus(ObjectData objdata, int totalset,TMP_Text achivedText,CircleFill circleFill)
        : base(objdata, totalset,achivedText,circleFill)
    {
    }
    public override void ResetState()
    {
        base.ResetState(); // 부모 클래스의 기본 초기화 로직을 먼저 실행

        // ExternalRotation만의 고유한 상태 변수들을 초기화
        currentState = ExerciseState.Idle;
        holdTimer = 0f;
        totalPoseScore = 0f;
        poseScoreSampleCount = 0f;
        ismove = false;
        starttimer = 0f;
        startTimerLastUpdate = 0f;
        wristPosQueue.Clear();
        completedReps = 0;
        idleTimer = 0f;
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
            
            if (starttimer > 5f)
            {
                 MessageManager.Instance.DisplayMessage("동작을 시작해주세요.");
                 starttimer = 0f;
                 idleTimer_2=0f;
                 idleTimer=0f;
            }
            return;
        }

        float angleRad_20 = 20f * Mathf.Deg2Rad;
        float cos20 = Mathf.Cos(angleRad_20);
        float sin20 = Mathf.Sin(angleRad_20);
        float angleRad = 45f * Mathf.Deg2Rad;
        float cos45 = Mathf.Cos(angleRad);
        float sin45 = Mathf.Sin(angleRad);

        Vector2 elbowTargetPosition;
        Vector2 wristTargetPosition;
        Vector2 elbowReturnPosition;
        Vector2 elbowReturnPosition_opposite;
        Vector2 wristReturnPosition;
        Vector2 wristReturnPosition_opposite;
        Vector2 kneeReturnPosition;
        Vector2 kneeReturnPosition_opposite;
        Vector2 ankleReturnPosition;
        Vector2 ankleReturnPosition_opposite;
        Vector2 headReturnPosition;

        if (currentEvaluationSide == EvaluationSide.Left)
        {
            // 첫번쨰 운동
            elbowTargetPosition = new Vector2(-0.5f * cos20, 0.5f * sin20);
            wristTargetPosition = new Vector2(-0.95f * cos20, 0.95f * sin20);

            // 기본 준비 자세
            elbowReturnPosition = new Vector2(-0.5f * cos45, -0.5f * sin45);
            wristReturnPosition = new Vector2(-1.0f * cos45, -1.0f * sin45);

            elbowReturnPosition_opposite = new Vector2(0f, -0.5f);
            wristReturnPosition_opposite = new Vector2(0f, -1.0f);
            kneeReturnPosition = new Vector2(0f, -0.5f);
            kneeReturnPosition_opposite = new Vector2(0f, -0.5f);
            ankleReturnPosition = new Vector2(0f, -1.0f);
            ankleReturnPosition_opposite = new Vector2(0f, -1.0f);
            headReturnPosition = new Vector2(0f, 1.0f);
        }
        else // EvaluationSide.Right
        {
            elbowTargetPosition = new Vector2(0.5f * cos20, 0.5f * sin20);
            wristTargetPosition = new Vector2(0.95f * cos20, 0.95f * sin20);

            // 기본 준비 자세
            elbowReturnPosition = new Vector2(0.5f * cos45, -0.5f * sin45);
            wristReturnPosition = new Vector2(0.95f * cos45, -1.0f * sin45);

            elbowReturnPosition_opposite = new Vector2(0f, -0.5f);
            wristReturnPosition_opposite = new Vector2(0f, -1.0f);
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
        bool isTargetPoseMaintained = 
            Vector2.Distance(currentElbowPos, elbowTargetPosition) < POSITION_TOLERANCE &&
            Vector2.Distance(currentWristPos, wristTargetPosition) < POSITION_TOLERANCE;

        bool isReturnPoseReached = Vector2.Distance(currentElbowPos, elbowReturnPosition) < POSITION_TOLERANCE
         && Vector2.Distance(currentWristPos, wristReturnPosition) < POSITION_TOLERANCE;

        if(idleTimer_2>3.0f){
            if (Vector2.Distance(currentElbowPos_opposite, elbowReturnPosition_opposite) > POSITION_TOLERANCE ||
                Vector2.Distance(currentWristPos_opposite, wristReturnPosition_opposite) > POSITION_TOLERANCE)
            {
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
            if(Vector2.Distance(currentKneePos, kneeReturnPosition) > POSITION_TOLERANCE ||
                Vector2.Distance(currentAnklePos, ankleReturnPosition) > POSITION_TOLERANCE
                ){
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
            if(Vector2.Distance(currentKneePos_opposite, kneeReturnPosition_opposite) > POSITION_TOLERANCE||
            Vector2.Distance(currentAnklePos_opposite, ankleReturnPosition_opposite) > POSITION_TOLERANCE){
                if (currentEvaluationSide == EvaluationSide.Left){
                    MessageManager.Instance.DisplayMessage("왼쪽 다리를 바로 해주세요.");
                }
                else{
                    MessageManager.Instance.DisplayMessage("오른쪽 다리를 바로 해주세요.");
                }
                //FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            // if(Vector2.Distance(currentHeadPos, headReturnPosition) > POSITION_TOLERANCE){
            //     MessageManager.Instance.DisplayMessage("상체를 바로 해주세요.");
            //     // FeedbackImageManager.Instance.ShowBadImage();
            //     int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
            //     PlayerPrefs.SetInt("BadCount", badCount);
            //     idleTimer_2 = 0f;
            // }
        }
        switch (currentState)
        {
            case ExerciseState.Idle:
                Debug.Log("Idle");
                if (isTargetPoseMaintained&&!hasreached)
                {
                    MessageManager.Instance.DisplayMessage("3초간 유지하세요.");
                    holdTimer = 0f;
                    idleTimer = 0f;
                    currentState = ExerciseState.Holding;
                    hasreached=true;
                }
                else if (!hasreached)
                {
                    idleTimer += realDeltaTime;
                    if (idleTimer > 4.0f)
                    {
                        Vector2 pathStart = wristReturnPosition;
                        Vector2 pathEnd = wristTargetPosition;
                        
                        Vector2 pathVector = pathEnd - pathStart;
                        Vector2 pointVector = currentWristPos - pathStart;

                        float t = Vector2.Dot(pointVector, pathVector) / pathVector.sqrMagnitude;
                        Vector2 closestPointOnSegment = pathStart + Mathf.Clamp01(t) * pathVector;
                        
                        float distanceToPath = Vector2.Distance(currentWristPos, closestPointOnSegment);
                        if (distanceToPath > POSITION_TOLERANCE)
                        {
                            MessageManager.Instance.DisplayMessage("잘못된 행동이에요.");
                            // FeedbackImageManager.Instance.ShowBadImage();
                            int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                            PlayerPrefs.SetInt("BadCount", badCount);
                            idleTimer = 0f;
                        }
                        else
                        {
                            MessageManager.Instance.DisplayMessage("밴드를 끝까지 당겨주세요.");
                            int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                            PlayerPrefs.SetInt("BadCount", badCount);
                            idleTimer = 0f;
                        }
                    }
                }
                break;
            case ExerciseState.Holding:
                holdTimer += realDeltaTime;
                Debug.Log("Holding");
                float distance = (Vector2.Distance(currentElbowPos, elbowTargetPosition)+Vector2.Distance(currentWristPos, wristTargetPosition))/2f;
                float accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                if (holdTimer >= HOLD_DURATION-1.0f)
                {
                    currentState=ExerciseState.Returning;
                    // MessageManager.Instance.DisplayMessage("준비자세로 복귀하세요.");
                    // if(idleTimer>1.0f){
                        
                    //     idleTimer=0f;
                    // }
                }
                else if(holdTimer>=HOLD_DURATION-1.0f&&!isTargetPoseMaintained)
                {
                    MessageManager.Instance.DisplayMessage("자세를 유지해주세요.");
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                }
                break;

            case ExerciseState.Returning:
                Debug.Log("Returning");
                if(isReturnPoseReached){
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
                }
                break;
        }
    }
    private float calculateScore(float holdTimer)
    {
        Debug.Log("holdTimer"+holdTimer);
        holdTimer = (holdTimer - HOLD_DURATION) < -0.5f ? holdTimer + 0.5f : (holdTimer - HOLD_DURATION) > 1.0f ? holdTimer - 1.0f : holdTimer;
        float timerScore = (HOLD_DURATION-Math.Abs(holdTimer - HOLD_DURATION))/HOLD_DURATION*100f;
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

