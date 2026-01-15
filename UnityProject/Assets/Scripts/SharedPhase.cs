using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Assets/Scripts/SharedPhase.cs
public static class SharedPhase
{
    // 간단하고 호환성 좋은 락 방식 (asmdef/플랫폼 상관없이 잘 작동)
    private static readonly object _lockObj = new object();
    private static string _current = "REST";   // 기본값

    public static string CurrentVideoName = "result"; 
    public static string CurrentVideoLogName = "result";   // 기본값
    public static string Current
    {
        get { lock (_lockObj) return _current; }
        set
        {
            var v = (value == "ACTIVE") ? "ACTIVE" : "REST";
            lock (_lockObj) _current = v;
        }
    }
}
