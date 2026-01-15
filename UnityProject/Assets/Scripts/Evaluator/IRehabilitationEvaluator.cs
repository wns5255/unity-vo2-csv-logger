using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IRehabilitationEvaluator 
{
    void StartEvaluation();
    void UpdateData(ObjectData objectData);
    void RealtimeEvaluate(int setIndex);
    void ResetState(); // 상태 초기화 함수 추가
    //void CalculateAccuracy();
    void ResetSet(int currentSet);
    void PrintRealtimeData(int currentset);
}
