using System;

namespace FastDtw.CSharp.Implementations.Shared
{
    internal static class InputArrayValidator
    {
        // 배열 길이 검증 (가중치 없는 경우)
        internal static void ValidateLength<T>(Span<T> arrayA, Span<T> arrayB) where T : struct
        {
            if (arrayA.Length < 2)
            {
                throw new ArgumentException("Array length should be at least 2", nameof(arrayA));
            }

            if (arrayB.Length < 2)
            {
                throw new ArgumentException("Array length should be at least 2", nameof(arrayB));
            }
        }
    }
}