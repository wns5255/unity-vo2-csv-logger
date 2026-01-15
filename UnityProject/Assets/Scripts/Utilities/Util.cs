using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using UnityEditor;
using UnityEngine;
using FastDtw.CSharp.Implementations;
using FastDtw.CSharp;

public class Util
{
    private const float WIDTH = 960f;
    private const float HEIGHT = 720f;
    public static double DtwGetScore(Span<double> arrayA, Span<double> arrayB)
    {
        return UnweightedDtw.GetScore(arrayA, arrayB);
    }

    public static float DtwGetScore(Span<float> arrayA, Span<float> arrayB)
    {
        return UnweightedDtw.GetScoreF(arrayA, arrayB);
    }
    public static PathResult DtwGetPath(float[] arrayA, float[] arrayB)
    {
        return UnweightedDtwPath.GetPath(arrayA, arrayB);
    }

    public static (float[], float[], float[]) NormalizePositions(float[] point1, float[] point2, float[] point3,float[] shoulder, float[] elbow, float[] wrist, bool isLimbPart = true)
    {
        float padding = 10f;
        float distance1 = CalculateEuclideanDistance(point1, point2);
        float distance2 = CalculateEuclideanDistance(point2, point3);
        
        float scaleDistance;
        if (isLimbPart) // 팔이나 다리 부분
        {
            scaleDistance = Mathf.Max(distance1, distance2) * 2f + padding;
        }
        else // 몸통/상체 부분
        {
            scaleDistance = distance1 + distance2 + padding;
        }

        float scaleFactor = 1f / scaleDistance;
        float refX = shoulder[0];
        float refY = shoulder[1];
        float[][] points = new float[][] { shoulder, elbow, wrist };
        float[][] normalizedPoints = new float[3][];

        for (int i = 0; i < points.Length; i++)
        {
            normalizedPoints[i] = new float[2];
            normalizedPoints[i][0] = (points[i][0] - refX) * scaleFactor;
            normalizedPoints[i][1] = (points[i][1] - refY) * scaleFactor;
        }
        return (normalizedPoints[0], normalizedPoints[1], normalizedPoints[2]);
    }
    public static float CalculateEuclideanDistance(float[] point1, float[] point2)
    {
        if (point1 == null || point2 == null || point1.Length < 2 || point2.Length < 2)
            return 0f;

        float dx = point1[0] - point2[0];
        float dy = point1[1] - point2[1];
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    public static float TrigonometricFunction(float degree, float distance)
    {
        float radians = degree * Mathf.Deg2Rad;
        float cosValue = Mathf.Cos(radians);
        float result = cosValue - Mathf.Abs(distance);
        return result;
    }
    public static float SimilarityDistance(float[] point1, float[] point2, float[] point3, float[] point4)
    {
        float result;
        float uclideanDistance1 = Util.CalculateEuclideanDistance(point1, point2);
        float uclideanDistance2 = Util.CalculateEuclideanDistance(point3, point4);
        result = (uclideanDistance1 + uclideanDistance2) / 2;
        return result;
    }

    // 점수 계산 유틸리티
    public static float CalculateExerciseScore(float holdTimer, float holdDuration, float totalPoseScore, float poseScoreSampleCount)
    {
        // 타이머 조정
        float adjustedTimer = (holdTimer - holdDuration) > 2.0f ? holdTimer - 2.0f : 
                             (holdTimer - holdDuration) > 1.0f ? holdTimer - 1.0f : holdTimer;
        
        float timerScore = (holdDuration - Math.Abs(adjustedTimer - holdDuration)) / holdDuration * 100f;
        float poseScore = (poseScoreSampleCount > 0) ? (totalPoseScore / poseScoreSampleCount) : 0f;
        float averageScore = (timerScore + poseScore) / 2f;
        
        // 감점 계산
        int badCount = PlayerPrefs.GetInt("BadCount", 0);
        float badPenalty = badCount * 5;
        
        float finalScore = averageScore - badPenalty;
        return Mathf.Max(0, finalScore);
    }

    // 복합 운동 점수 계산 (PassiveExternalRotation, SleeperStretch 등용)
    public static float CalculateComplexExerciseScore(float totalHoldTimer, float totalHoldDuration, float totalPoseScore, float poseScoreSampleCount)
    {
        float adjustedTimer = (totalHoldTimer - totalHoldDuration) > 2.0f ? totalHoldTimer - 2.0f : 
                             (totalHoldTimer - totalHoldDuration) > 1.0f ? totalHoldTimer - 1.0f : totalHoldTimer;
        
        float timerScore = (totalHoldDuration - Math.Abs(adjustedTimer - totalHoldDuration)) / totalHoldDuration * 100f;
        float poseScore = (poseScoreSampleCount > 0) ? (totalPoseScore / poseScoreSampleCount) : 0f;
        float averageScore = (timerScore + poseScore) / 2f;
        
        int badCount = PlayerPrefs.GetInt("BadCount", 0);
        float badPenalty = badCount * 5;
        
        float finalScore = averageScore - badPenalty;
        return Mathf.Max(0, finalScore);
    }

    // 각도 기반 위치 계산
    public static Vector2 GetPositionWithAngle(Vector2 center, float distance, float angleDegrees)
    {
        float angleRad = angleDegrees * Mathf.Deg2Rad;
        return new Vector2(
            center.x + distance * Mathf.Cos(angleRad),
            center.y + distance * Mathf.Sin(angleRad)
        );
    }

    // 운동 경로상의 점까지의 거리 계산
    public static float CalculateDistanceToPath(Vector2 currentPos, Vector2 pathStart, Vector2 pathEnd)
    {
        Vector2 pathVector = pathEnd - pathStart;
        Vector2 pointVector = currentPos - pathStart;
        
        float t = Vector2.Dot(pointVector, pathVector) / pathVector.sqrMagnitude;
        Vector2 closestPointOnSegment = pathStart + Mathf.Clamp01(t) * pathVector;
        
        return Vector2.Distance(currentPos, closestPointOnSegment);
    }
}
