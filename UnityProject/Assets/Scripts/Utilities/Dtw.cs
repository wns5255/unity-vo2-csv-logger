using System;
using FastDtw.CSharp.Implementations;

namespace FastDtw.CSharp
{
    public static class Dtw
    {
        // Unweighted DTW using Span<T>
        public static double GetScore(Span<double> arrayA, Span<double> arrayB)
        {
            return UnweightedDtw.GetScore(arrayA, arrayB);
        }

        public static float GetScore(Span<float> arrayA, Span<float> arrayB)
        {
            return UnweightedDtw.GetScoreF(arrayA, arrayB);
        }
        public static PathResult GetPath(float[] arrayA, float[] arrayB)
        {
            return UnweightedDtwPath.GetPath(arrayA, arrayB);
        }
    }
}