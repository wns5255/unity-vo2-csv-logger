using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using TMPro;

// 모든 파일에서 접근 가능하도록 enum을 클래스 밖에 선언합니다.
public enum EvaluationSide { Left, Right, None }

public abstract class BaseRehabilitationEvaluator : IRehabilitationEvaluator
{
    protected ObjectData objdata;
    protected ObjectData currentobjdata;
    protected List<List<List<float>>> realtimedata = new List<List<List<float>>>();
    protected int totalset;
    protected (float[] shoulder, float[] elbow, float[] wrist) normalizedLeftArm;
    protected (float[] shoulder, float[] elbow, float[] wrist) normalizedRightArm;
    protected (float[] pelvis, float[] thorax, float[] head) normalizedBody;
    protected (float[] hip, float[] knee, float[] ankle) normalizedLeftLeg;
    protected (float[] hip, float[] knee, float[] ankle) normalizedRightLeg;
    protected EvaluationSide currentEvaluationSide = EvaluationSide.Left; // 기본값을 왼쪽으로 설정
    protected TMP_Text achivedText;
    protected CircleFill circleFill;

    public BaseRehabilitationEvaluator(ObjectData objdata, int totalset, TMP_Text achivedText, CircleFill circleFill)
    {
        this.objdata = objdata;
        this.totalset = totalset;
        this.achivedText = achivedText;
        this.circleFill = circleFill;
        StartEvaluation();
    }

    public virtual void StartEvaluation()
    {
        InitializeRealtimeData();
    }

    protected virtual void InitializeRealtimeData()
    {
        for (int j = 0; j < totalset; j++)
        {
            realtimedata.Add(new List<List<float>>
            {
                new List<float>(),
                new List<float>()
            });
        }
    }

    public virtual void UpdateData(ObjectData newObjData)
    {
        this.currentobjdata = newObjData;
        // 데이터를 업데이트할 때 공통 정규화 로직을 미리 실행
        NormalizeAllParts();
    }

    protected virtual void NormalizeAllParts()
    {
        if (objdata == null || currentobjdata == null) return;

        normalizedLeftArm = Util.NormalizePositions(objdata.left_shoulder, objdata.left_elbow, objdata.left_wrist, currentobjdata.left_shoulder, currentobjdata.left_elbow, currentobjdata.left_wrist);
        normalizedRightArm = Util.NormalizePositions(objdata.right_shoulder, objdata.right_elbow, objdata.right_wrist, currentobjdata.right_shoulder, currentobjdata.right_elbow, currentobjdata.right_wrist);

        normalizedBody = Util.NormalizePositions(objdata.pelvis, objdata.thorax, objdata.head, currentobjdata.pelvis, currentobjdata.thorax, currentobjdata.head, false);

        normalizedLeftLeg = Util.NormalizePositions(objdata.left_hip, objdata.left_knee, objdata.left_ankle, currentobjdata.left_hip, currentobjdata.left_knee, currentobjdata.left_ankle);
        normalizedRightLeg = Util.NormalizePositions(objdata.right_hip, objdata.right_knee, objdata.right_ankle, currentobjdata.right_hip, currentobjdata.right_knee, currentobjdata.right_ankle);
    }

    // 평가 방향을 설정하는 가상 함수 추가
    public virtual void SetEvaluationSide(EvaluationSide side)
    {
        currentEvaluationSide = side;
    }

    public abstract void RealtimeEvaluate(int currentset);
    //public abstract void CalculateAccuracy();

    public virtual void ResetState()
    {
    }

    public virtual void ResetSet(int currentSet)
    {
        if (currentSet >= 0 && currentSet < realtimedata.Count)
        {
            realtimedata[currentSet][0].Clear();
            realtimedata[currentSet][1].Clear();
        }
    }
    public virtual void PrintRealtimeData(int currentset)
    {
        Debug.Log($"set : {currentset}, realdata length : {realtimedata[currentset][0].Count}");
    }
}