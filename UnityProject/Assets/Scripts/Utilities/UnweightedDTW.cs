using System;
using FastDtw.CSharp.Implementations.Shared;
namespace FastDtw.CSharp.Implementations
{
    internal static class UnweightedDtw
    {
        internal static double GetScore(Span<double> arrayA, Span<double> arrayB)
        {
            InputArrayValidator.ValidateLength<double>(arrayA, arrayB);

            int aLength = arrayA.Length;
            int bLength = arrayB.Length;
            double[] tCostMatrix = new double[2 * bLength];

            int previousRow = 0;
            int currentRow = -bLength;
            int tPathLength = tCostMatrix.Length;

            double lastMin = 0;
            double lastCalculatedCost = 0;

            for (int i = 0; i < aLength; i++)
            {
                currentRow += bLength;
                if (currentRow == tPathLength)
                {
                    currentRow = 0;
                }

                for (int j = 0; j < bLength; j++)
                {
                    DtwShared.UpdateLastMin(i, j, previousRow, lastCalculatedCost, tCostMatrix, ref lastMin);

                    double absDifference = Math.Abs(arrayA[i] - arrayB[j]);

                    lastCalculatedCost = absDifference + lastMin;
                    tCostMatrix[currentRow + j] = lastCalculatedCost;
                }

                previousRow = currentRow;
            }

            return tCostMatrix[currentRow + bLength - 1];
        }

        internal static float GetScoreF(Span<float> arrayA, Span<float> arrayB)
        {
            InputArrayValidator.ValidateLength<float>(arrayA, arrayB);

            int aLength = arrayA.Length;
            int bLength = arrayB.Length;
            float[] tCostMatrix = new float[2 * bLength];

            int previousRow = 0;
            int currentRow = -bLength;
            int tPathLength = tCostMatrix.Length;

            float lastMin = 0;
            float lastCalculatedCost = 0;

            for (int i = 0; i < aLength; i++)
            {
                currentRow += bLength;
                if (currentRow == tPathLength)
                {
                    currentRow = 0;
                }

                for (int j = 0; j < bLength; j++)
                {
                    DtwShared.UpdateLastMinF(i, j, previousRow, lastCalculatedCost, tCostMatrix, ref lastMin);

                    float absDifference = Math.Abs(arrayA[i] - arrayB[j]);

                    lastCalculatedCost = absDifference + lastMin;
                    tCostMatrix[currentRow + j] = lastCalculatedCost;
                }

                previousRow = currentRow;
            }

            return tCostMatrix[currentRow + bLength - 1];
        }
    }
}