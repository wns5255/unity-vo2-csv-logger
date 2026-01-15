using System;
using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting.FullSerializer;
using UnityEngine;
public class ArmPositions
{
    public int Elbow_Col { get; set; }
    public int Elbow_Row { get; set; }
    public int Wrist_Col { get; set; }
    public int Wrist_Row { get; set; }

    public ArmPositions(int elbow_Col, int elbow_Row, int wrist_Col, int wrist_Row)
    {
        Elbow_Col = elbow_Col;
        Elbow_Row = elbow_Row;
        Wrist_Col = wrist_Col;
        Wrist_Row = wrist_Row;
    }
}
public class BodyPartCalculator
{
    public static ArmPositions Arms(float[] shoulder,float[] elbow,float[] wrist, float[] current_shoulder, float[] current_elbow, float[] current_wrist, float cal, float row)
    {
        Vector2 vshoulder = new Vector2(shoulder[0], shoulder[1]);
        Vector2 velbow = new Vector2(elbow[0], elbow[1]);
        Vector2 vwrist = new Vector2(wrist[0], wrist[1]);
        Vector2 vcurrent_shoulder = new Vector2(current_shoulder[0], current_shoulder[1]);
        Vector2 vcurrent_elbow = new Vector2(current_elbow[0], current_elbow[1]);
        Vector2 vcurrent_wrist = new Vector2(current_wrist[0], current_wrist[1]);

        float padding = 5f;
        float magnitude = Mathf.Max(Vector2.Distance(vshoulder,velbow), Vector2.Distance(vshoulder,vwrist));
        float x_min = vcurrent_shoulder.x-magnitude - padding;
        float x_max = vcurrent_shoulder.x+magnitude + padding;
        float y_min = vcurrent_shoulder.y-magnitude - padding;
        float y_max = vcurrent_shoulder.y+magnitude + padding;

        float cell_width = (magnitude * 2) / cal;
        float cell_height = (magnitude * 2) / row;

        int elbow_col = Mathf.FloorToInt((vcurrent_elbow.x - x_min) / cell_width);
        int elbow_row = Mathf.FloorToInt((vcurrent_elbow.y - y_min) / cell_height);
        int wrist_col = Mathf.FloorToInt((vcurrent_wrist.x - x_min) / cell_width);
        int wrist_row = Mathf.FloorToInt((vcurrent_wrist.y - y_min) / cell_height);

        elbow_col = Mathf.Clamp(elbow_col, 0, 7);
        elbow_row = Mathf.Clamp(elbow_row, 0, 7);
        wrist_col = Mathf.Clamp(wrist_col, 0, 7);
        wrist_row = Mathf.Clamp(wrist_row, 0, 7);

        return new ArmPositions(elbow_col,elbow_row, wrist_col,wrist_row);

    }
    public static void Body(float[] neck, float[] middle_hip, float[] current_neck, float[] current_middle_hip)
    {
        Vector2 vneck = new Vector2(neck[0], neck[1]);
        Vector2 vmiddle_hip = new Vector2(middle_hip[0], middle_hip[1]);
        Vector2 vcurrent_neck = new Vector2(current_neck[0], current_neck[1]);
        Vector2 vcurrent_middle_hip = new Vector2(current_middle_hip[0], current_middle_hip[1]);

        float padding = 5f;
        float magnitude = Vector2.Distance(vneck ,vmiddle_hip);
        float x_min = vcurrent_middle_hip.x - magnitude -padding;
        float x_max = vcurrent_middle_hip.x + magnitude + padding;
        float y_min = vcurrent_middle_hip.y - padding;
        float y_max = vcurrent_middle_hip.y + magnitude + padding;

        float cell_width = (magnitude * 2) / 8f;
        float cell_height = magnitude / 4f;

        int neck_col = Mathf.FloorToInt((vcurrent_neck.x - x_min) / cell_width);
        int neck_row = Mathf.FloorToInt((vcurrent_neck.y - y_min) / cell_height);

        neck_col = Mathf.Clamp(neck_col, 0, 3);
        neck_row = Mathf.Clamp(neck_row, 0, 3);
    }
    public static void Legs(float[] hip, float[] knee, float[] ankle, float[] current_hip, float[] current_knee, float[] current_ankle)
    {
        Vector2 vhip = new Vector2(hip[0], hip[1]);
        Vector2 vknee = new Vector2(knee[0], knee[1]);
        Vector2 vankle = new Vector2(ankle[0], ankle[1]);
        Vector2 vcurrent_hip = new Vector2(current_hip[0], current_hip[1]);
        Vector2 vcurrent_knee = new Vector2(current_knee[0], current_knee[1]);
        Vector2 vcurrent_ankle = new Vector2(current_ankle[0], current_ankle[1]);

        float padding = 5f;
        float magnitude = Mathf.Max(Vector2.Distance(vhip,vknee), Vector2.Distance(vhip , vankle));
        float x_min = vcurrent_hip.x - magnitude - padding;
        float x_max = vcurrent_hip.x + magnitude + padding;
        float y_min = vcurrent_hip.y - magnitude - padding;
        float y_max = vcurrent_hip.y + magnitude + padding;

        float cell_width = (magnitude * 2) / 8f;
        float cell_height = magnitude / 8f;

        int knee_col = Mathf.FloorToInt((vcurrent_knee.x - x_min) / cell_width);
        int knee_row = Mathf.FloorToInt((vcurrent_knee.y - y_min) / cell_height);
        int ankle_col = Mathf.FloorToInt((vcurrent_ankle.x - x_min) / cell_width);
        int ankle_row = Mathf.FloorToInt((vcurrent_ankle.y - y_min) / cell_height);

        knee_col = Mathf.Clamp(knee_col, 0, 7);
        knee_row = Mathf.Clamp(knee_row, 0, 7);
        ankle_col = Mathf.Clamp(ankle_col, 0, 7);
        ankle_row = Mathf.Clamp(ankle_row, 0, 7);

    }
}
