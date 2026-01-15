using System;
using System.Collections.Generic;

namespace FastDtw.CSharp.Implementations.Shared
{
    internal static class DtwShared
    {
        // 최적 경로를 추출하는 메서드
        internal static List<Tuple<int, int>> GetPathFromCostMatrix(double[,] matrix, int aLength, int bLength)
        {
            var result = new List<Tuple<int, int>>(Math.Max(aLength, bLength));
            int i = aLength - 1;
            int j = bLength - 1;

            result.Add(Tuple.Create(i, j));
            while (i != 0 || j != 0)
            {
                if (i == 0 && j != 0)
                {
                    j--;
                }
                else if (j == 0 && i != 0)
                {
                    i--;
                }
                else
                {
                    double leftRowTop = matrix[i - 1, j];
                    double currentRowBottom = matrix[i, j - 1];
                    double leftRowBottom = matrix[i - 1, j - 1];

                    if (leftRowBottom <= leftRowTop)
                    {
                        if (leftRowBottom <= currentRowBottom)
                        {
                            i--;
                            j--;
                        }
                        else
                        {
                            j--;
                        }
                    }
                    else
                    {
                        if (leftRowTop <= currentRowBottom)
                        {
                            i--;
                        }
                        else
                        {
                            j--;
                        }
                    }
                }

                result.Add(Tuple.Create(i, j));
            }

            result.Reverse();

            return result;
        }

        // 최소 비용을 업데이트하는 메서드 (double 버전)
        internal static void UpdateLastMin(int idxArrayA, int idxArrayB, int previousRow, double lastCalculatedCost,
            double[] costMatrix, ref double lastMin)
        {
            if (idxArrayA == 0 && idxArrayB == 0)
            {
                lastMin = 0;
            }
            else if (idxArrayA == 0)
            {
                lastMin = lastCalculatedCost;
            }
            else if (idxArrayB == 0)
            {
                lastMin = costMatrix[previousRow];
            }
            else
            {
                lastMin = NumericHelpers.FindMinimum(
                    ref costMatrix[previousRow + idxArrayB],
                    ref costMatrix[previousRow + idxArrayB - 1],
                    ref lastCalculatedCost);
            }
        }

        // 최소 비용을 업데이트하는 메서드 (float 버전)
        internal static void UpdateLastMinF(int idxArrayA, int idxArrayB, int previousRow, float lastCalculatedCost,
            float[] costMatrix, ref float lastMin)
        {
            if (idxArrayA == 0 && idxArrayB == 0)
            {
                lastMin = 0;
            }
            else if (idxArrayA == 0)
            {
                lastMin = lastCalculatedCost;
            }
            else if (idxArrayB == 0)
            {
                lastMin = costMatrix[previousRow];
            }
            else
            {
                lastMin = NumericHelpers.FindMinimumF(
                    ref costMatrix[previousRow + idxArrayB],
                    ref costMatrix[previousRow + idxArrayB - 1],
                    ref lastCalculatedCost);
            }
        }
    }
}