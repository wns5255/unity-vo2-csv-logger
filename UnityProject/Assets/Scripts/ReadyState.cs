using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReadyState : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Invoke("send_True", 5);
    }

    public bool send_True()
    {
        return true;
    }

}
