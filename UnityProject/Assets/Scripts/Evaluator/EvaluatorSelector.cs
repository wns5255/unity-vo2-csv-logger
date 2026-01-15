using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;
using TMPro;

public static class EvaluatorSelector
{ 
    public static IRehabilitationEvaluator CreateEvaluator(string exerciseType, ObjectData objdata,int setNum,TMP_Text achivedText,CircleFill circleFill)
    {
        switch (exerciseType)
        {
            case "슬리퍼스트레치 운동":
                return new SleeperStretch(objdata,setNum,achivedText,circleFill);
            case "측와위외회전 운동":
                return new SidelyingExternalRotation(objdata,setNum,achivedText,circleFill);
            case "수동 외회전 운동":
                return new PassiveExternalRotaion(objdata,setNum,achivedText,circleFill);
            case "외회전 운동":
                return new ExternalRotation(objdata, setNum,achivedText,circleFill);
            case "내회전 운동":
                return new InternalRotation(objdata, setNum,achivedText,circleFill);
            case "오픈캔자세외전":
                return new Supraspinatus(objdata,setNum,achivedText,circleFill);
            case "검빼듯대각선리프트":
                return new ShoulderDiagonals(objdata, setNum,achivedText,circleFill);
            default:
                throw new ArgumentException("올바른 운동 타입이 아닙니다.");
        }
    }
}
