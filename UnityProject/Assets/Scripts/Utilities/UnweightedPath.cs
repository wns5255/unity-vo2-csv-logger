using System;
using FastDtw.CSharp.Implementations.Shared;

namespace FastDtw.CSharp.Implementations
{
    public static class UnweightedDtwPath
    {
        public static PathResult GetPath(float[] arrayA, float[] arrayB)
        {
            var aLength = arrayA.Length;
            var bLength = arrayB.Length;
            var tCostMatrix = new double[aLength, bLength];

            double lastMin;
            double lastCalculatedCost = 0;

            for (var i = 0; i < aLength; i++)
            {
                for (var j = 0; j < bLength; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        lastMin = 0;
                    }
                    else if (i == 0)
                    {
                        lastMin = lastCalculatedCost;
                    }
                    else if (j == 0)
                    {
                        lastMin = tCostMatrix[i - 1, j];
                    }
                    else
                    {
                        lastMin = NumericHelpers.FindMinimum(
                            ref tCostMatrix[i - 1, j],
                            ref tCostMatrix[i - 1, j - 1],
                            ref lastCalculatedCost);
                    }

                    var absDifference = Math.Abs(arrayA[i] - arrayB[j]);

                    lastCalculatedCost = absDifference + lastMin;
                    tCostMatrix[i, j] = lastCalculatedCost;
                }
            }

            return new PathResult(tCostMatrix[aLength - 1, bLength - 1],
                DtwShared.GetPathFromCostMatrix(tCostMatrix, aLength, bLength));
        }
    }
}