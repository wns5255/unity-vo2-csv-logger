using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ViTPose : MonoBehaviour
{
    public UDPReceiver udpReceive;
    public GameObject[] point;

    void Start()
    {
        
    }

    //Update is called once per frame
    void Update()
    {
        if (udpReceive.keypoints_dataReceived == true)
        {
            List<float[]> positions = udpReceive.keypoints.objects[0].GetAllPositions();
            int i = 0;
            foreach (float[] position in positions)
            {
                if (i < point.Length && position.Length == 3)
                {
                    point[i].transform.localPosition = new Vector3(position[0]*2, position[1]*2, position[2]*2);
                    print(position[0]);
                    i++;
                }
            }
        }
    }

}
