using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;


public class ExternalRotation : BaseRehabilitationEvaluator
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
    private const float HOLD_DURATION = 3f;         // 자세 유지 시간
    private const float POSITION_TOLERANCE = 0.3f;  // 위치 허용 오차 (값이 작을수록 정확해야 함)
    
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

    public ExternalRotation(ObjectData objdata, int totalset,TMP_Text achivedText,CircleFill circleFill)
        : base(objdata, totalset,achivedText,circleFill)
    {
    }

    public override void ResetState()
    {
        base.ResetState(); // 부모 클래스의 기본 초기화 로직을 먼저 실행

        // ExternalRotation만의 고유한 상태 변수들을 초기화
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
            wristPosQueue.Enqueue(currentWristPos); // 선택된 팔의 데이터 사용
            if (wristPosQueue.Count > maxQueueSize) wristPosQueue.Dequeue();
            
            if (wristPosQueue.Count == maxQueueSize)
            {
                if (Vector2.Distance(wristPosQueue.Peek(), currentWristPos) > 0.1) // 선택된 팔의 데이터 사용
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
        Debug.Log("start"+starttimer+"idleTimer_2"+idleTimer_2);
        
        float angleRad_45 = 45f * Mathf.Deg2Rad;
        float cos45 = Mathf.Cos(angleRad_45);
        float sin45 = Mathf.Sin(angleRad_45);
        
        Vector2 wristTargetPosition;
        Vector2 elbowReturnPosition_opposite;
        Vector2 wristReturnPosition_opposite;
        Vector2 kneeReturnPosition;
        Vector2 kneeReturnPosition_opposite;
        Vector2 ankleReturnPosition;
        Vector2 ankleReturnPosition_opposite;
        Vector2 headReturnPosition;

        
        if (currentEvaluationSide == EvaluationSide.Left)
        {
            wristTargetPosition = new Vector2(-0.5f, -0.5f);
            // 기본 준비 자세
            isReturnConditionMet = currentWristPos.x >= -0.1f;

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
            wristTargetPosition = new Vector2(0.5f, -0.5f);

            // 기본 준비 자세
            isReturnConditionMet = currentWristPos.x <= 0.1f;

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
        bool isTargetPoseMaintained = Vector2.Distance(currentWristPos, wristTargetPosition) < POSITION_TOLERANCE;
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
                //FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            if(Vector2.Distance(currentKneePos, kneeReturnPosition) > POSITION_TOLERANCE ||
                Vector2.Distance(currentAnklePos, ankleReturnPosition) > POSITION_TOLERANCE
                ){
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
            if(Vector2.Distance(currentKneePos_opposite, kneeReturnPosition_opposite) > POSITION_TOLERANCE||
            Vector2.Distance(currentAnklePos_opposite, ankleReturnPosition_opposite) > POSITION_TOLERANCE){
                if (currentEvaluationSide == EvaluationSide.Left){
                    MessageManager.Instance.DisplayMessage("오른쪽 다리를 바로 해주세요.");
                }
                else{
                    MessageManager.Instance.DisplayMessage("완쪽 다리를 바로 해주세요.");
                }
                //FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
            // if(Vector2.Distance(currentHeadPos, headReturnPosition) > POSITION_TOLERANCE){
            //     MessageManager.Instance.DisplayMessage("상체를 바로 해주세요.");
            //     //FeedbackImageManager.Instance.ShowBadImage();
            //     int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
            //     PlayerPrefs.SetInt("BadCount", badCount);
            //     idleTimer_2 = 0f;
            // }


            if (Vector2.Distance(currentElbowPos, elbowStartPosition) > POSITION_TOLERANCE)
            {
                MessageManager.Instance.DisplayMessage("팔꿈치를 옆구리에 고정하세요.");
                //FeedbackImageManager.Instance.ShowBadImage();   

                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2=0f;
            }
            if(currentEvaluationSide==EvaluationSide.Left&&currentWristPos.x>(POSITION_TOLERANCE+0.1f)&&Math.Abs(currentWristPos.y)-0.5f<POSITION_TOLERANCE||
            currentEvaluationSide==EvaluationSide.Right&&currentWristPos.x<-(POSITION_TOLERANCE+0.1f)&&Math.Abs(currentWristPos.y)-0.5f<POSITION_TOLERANCE){
                MessageManager.Instance.DisplayMessage("팔을 너무 안으로 굽히지 말아주세요.");
                //FeedbackImageManager.Instance.ShowBadImage();
                int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                PlayerPrefs.SetInt("BadCount", badCount);
                idleTimer_2 = 0f;
            }
        }
        // 실제 경과 시간 계산
        switch (currentState)
        {
            case ExerciseState.Idle:
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
                        if(Math.Abs(currentWristPos.y)-0.5f<POSITION_TOLERANCE){
                            MessageManager.Instance.DisplayMessage("밴드를 끝까지 당겨주세요.");
                            int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                            PlayerPrefs.SetInt("BadCount", badCount);
                            idleTimer = 0f;
                        }
                        else{
                            int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                            PlayerPrefs.SetInt("BadCount", badCount);
                            idleTimer = 0f;
                        }
                    }
                }
                break;
                

            case ExerciseState.Holding:
                holdTimer += realDeltaTime;
                float distance = Vector2.Distance(currentWristPos, wristTargetPosition);
                float accuracyRatio = 1 - distance;
                totalPoseScore += accuracyRatio * 100f;
                poseScoreSampleCount++;
                idleTimer+=realDeltaTime;
                if (holdTimer>=HOLD_DURATION-1.0f)
                {
                    currentState=ExerciseState.Returning;
                    // MessageManager.Instance.DisplayMessage("준비자세로 복귀하세요.");
                    // if(idleTimer>1.0f){
                        
                    //     idleTimer=0f;
                    // }
                        
                    }
                
                if(holdTimer<HOLD_DURATION-1.0f&&!isTargetPoseMaintained)
                {
                    MessageManager.Instance.DisplayMessage("자세를 유지해주세요.");
                    int badCount = PlayerPrefs.GetInt("BadCount", 0) + 1;
                    PlayerPrefs.SetInt("BadCount", badCount);
                }
                break;

            case ExerciseState.Returning:
                // if(isTargetPoseMaintained){
                //     holdTimer += realDeltaTime;
                // }
                // 위에서 설정된 복귀 조건을 사용합니다.
                if (isReturnConditionMet && Mathf.Abs(currentWristPos.y - wristTargetPosition.y) < POSITION_TOLERANCE)
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
                    completedReps++; // 완료 횟수 증가
                    MessageManager.Instance.DisplayMessage($"{completedReps}회 완료");
                    int achivedCount = PlayerPrefs.GetInt("AchivedCount", 0) + 1;
                    PlayerPrefs.SetInt("AchivedCount", achivedCount);
                    Debug.Log("AchivedCount"+achivedCount);
                    // 상태를 초기화하여 다음 동작을 준비합니다.
                    currentState = ExerciseState.Idle;
                    holdTimer = 0f;
                    idleTimer = 0f;
                    idleTimer_2=0f;
                    totalPoseScore = 0f;
                    poseScoreSampleCount = 0f;
                    starttimer = 0f;
                    startTimerLastUpdate = 0f; // 타이머 초기화
                    hasreached=false;
                    wristPosQueue.Clear();
                    ismove=false; // 다음 동작 감지를 위해 false로 초기화
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

