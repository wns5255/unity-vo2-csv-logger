using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

[System.Serializable]
public class Keypoints
{
    public ObjectData[] objects;


}
[System.Serializable]
public class ObjectData
{
    public float[] right_ankle;
    public float[] right_knee;
    public float[] right_hip;
    public float[] left_hip;
    public float[] left_knee;
    public float[] left_ankle;
    public float[] pelvis;
    public float[] thorax;
    public float[] neck;
    public float[] head;
    public float[] right_wrist;
    public float[] right_elbow;
    public float[] right_shoulder;
    public float[] left_shoulder;
    public float[] left_elbow;
    public float[] left_wrist;

    public List<float[]> GetAllPositions()
    {
        return new List<float[]>
        {
            right_ankle, right_knee, right_hip ,
            left_hip, left_knee, left_ankle,
            pelvis, thorax, neck, head,
            right_wrist, right_elbow, right_shoulder,
            left_shoulder, left_elbow,left_wrist
        };
    }

    public Dictionary<string, float[]> GetPositionsAsDictionary()
    {
        return new Dictionary<string, float[]>
        {
            { "right_ankle", right_ankle },
            { "right_knee", right_knee },
            { "right_hip", right_hip },
            { "left_hip", left_hip },
            { "left_knee", left_knee },
            { "left_ankle", left_ankle },
            { "pelvis", pelvis },
            { "thorax", thorax },
            { "neck", neck },
            { "head", head },
            { "right_wrist", right_wrist },
            { "right_elbow", right_elbow },
            { "right_shoulder", right_shoulder },
            { "left_shoulder", left_shoulder },
            { "left_elbow", left_elbow },
            { "left_wrist", left_wrist }
        };
    }
}
//[System.Serializable]
//public class ObjectData
//{
//    public float[] nose;
//    public float[] left_eye;
//    public float[] right_eye;
//    public float[] left_shoulder;
//    public float[] right_shoulder;
//    public float[] left_elbow;
//    public float[] right_elbow;
//    public float[] left_wrist;
//    public float[] right_wrist;
//    public float[] left_hip;
//    public float[] right_hip;
//    public float[] left_knee;
//    public float[] right_knee;
//    public float[] left_ankle;
//    public float[] right_ankle;
//    public float[] middle_hip;

//    public List<float[]> GetAllPositions()
//    {
//        return new List<float[]>
//        {
//            nose, left_eye, right_eye, left_shoulder, right_shoulder, 
//            left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, 
//            left_knee, right_knee, left_ankle, right_ankle, middle_hip
//        };
//    }
//}

//[System.Serializable]
//public class ObjectData
//{
//    public float[] middle_hip;
//    public float[] right_hip;
//    public float[] right_knee;
//    public float[] right_foot;
//    public float[] left_hip;
//    public float[] left_knee;
//    public float[] left_foot;
//    public float[] spine;
//    public float[] neck;
//    public float[] nose;
//    public float[] head;
//    public float[] left_shoulder;
//    public float[] left_elbow;
//    public float[] left_wrist;
//    public float[] right_shoulder;
//    public float[] right_elbow;
//    public float[] right_wrist;

//    public List<float[]> GetAllPositions()
//    {
//        return new List<float[]>
//        {
//            middle_hip, right_hip, right_knee ,right_foot, left_hip,
//            left_knee, left_foot, spine, neck,
//            nose, head, left_shoulder, left_elbow,
//            left_wrist, right_shoulder, right_elbow, right_wrist
//        };
//    }
//}




