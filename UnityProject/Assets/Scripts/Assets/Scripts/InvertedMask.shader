Shader "UI/InvertedMask"
{
    Properties
    {
        [PerRendererData] _MainTex ("Sprite Texture", 2D) = "white" {}
        _MaskTex ("Mask Texture", 2D) = "white" {}
        _Color ("Tint", Color) = (1,1,1,1)
        _MaskBackgroundColor ("Mask Background Color", Color) = (0, 0, 0, 0.5) // 마스크 배경색 (검정, 반투명)
        _OutlineColor ("Outline Color", Color) = (1, 0, 0, 1) // 외곽선 색 (기본 빨강)
        _OutlineThickness("Outline Thickness", Range(0, 10)) = 5 // 외곽선 두께
        
        [Header(Mask Transform)]
        _MaskScale ("Mask Scale (X Y)", Vector) = (1, 1, 0, 0) // 마스크 크기 (가로, 세로)
        _MaskOffset ("Mask Offset (X Y)", Vector) = (0, 0, 0, 0) // 마스크 위치 오프셋

        _StencilComp ("Stencil Comparison", Float) = 8
        _Stencil ("Stencil ID", Float) = 0
        _StencilOp ("Stencil Operation", Float) = 0
        _StencilWriteMask ("Stencil Write Mask", Float) = 255
        _StencilReadMask ("Stencil Read Mask", Float) = 255

        _ColorMask ("Color Mask", Float) = 15

        [Toggle(UNITY_UI_ALPHACLIP)] _UseUIAlphaClip ("Use Alpha Clip", Float) = 0
    }

    SubShader
    {
        Tags
        {
            "Queue"="Transparent"
            "RenderType"="Transparent"
            "IgnoreProjector"="True"
            "CanUseSpriteAtlas"="True"
        }

        Pass
        {
            Blend SrcAlpha OneMinusSrcAlpha
            Cull Off
            ZWrite Off
            ZTest [unity_GUIZTestMode]

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            #include "UnityUI.cginc"

            struct appdata
            {
                float4 vertex   : POSITION;
                float4 color    : COLOR;
                float2 texcoord : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex   : SV_POSITION;
                fixed4 color    : COLOR;
                float2 texcoord : TEXCOORD0;
                float2 maskCoord : TEXCOORD1;
            };
            
            sampler2D _MaskTex;
            float4 _MaskTex_ST;
            float4 _MaskTex_TexelSize;
            fixed4 _MaskBackgroundColor;
            fixed4 _OutlineColor;
            float _OutlineThickness;
            float4 _MaskScale;
            float4 _MaskOffset;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.texcoord = v.texcoord;
                o.maskCoord = TRANSFORM_TEX(v.texcoord, _MaskTex);
                o.color = v.color;
                return o;
            }

            float2 transformMaskCoord(float2 uv)
            {
                // 중심점을 (0.5, 0.5)로 이동
                uv -= 0.5;
                
                // 스케일 적용 (역수를 사용하여 스케일 값이 클수록 마스크가 커짐)
                uv /= _MaskScale.xy;
                
                // 오프셋 적용
                uv -= _MaskOffset.xy;
                
                // 다시 (0.5, 0.5)를 중심으로 복원
                uv += 0.5;
                
                return uv;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                // 마스크 좌표 변환 적용
                float2 transformedMaskCoord = transformMaskCoord(i.maskCoord);
                
                // 변환된 좌표가 범위를 벗어나면 배경으로 처리
                if (transformedMaskCoord.x < 0 || transformedMaskCoord.x > 1 || 
                    transformedMaskCoord.y < 0 || transformedMaskCoord.y > 1)
                {
                    return _MaskBackgroundColor;
                }

                float maskAlpha = tex2D(_MaskTex, transformedMaskCoord).a;

                // Outline Detection with transformed coordinates
                float2 texelSize = _MaskTex_TexelSize.xy;
                float maxAlpha = 0.0;
                
                float2 offsetUp = transformMaskCoord(i.maskCoord + _OutlineThickness * float2(0, texelSize.y));
                float2 offsetDown = transformMaskCoord(i.maskCoord - _OutlineThickness * float2(0, texelSize.y));
                float2 offsetRight = transformMaskCoord(i.maskCoord + _OutlineThickness * float2(texelSize.x, 0));
                float2 offsetLeft = transformMaskCoord(i.maskCoord - _OutlineThickness * float2(texelSize.x, 0));
                
                // 경계 체크를 포함한 알파값 샘플링
                if (offsetUp.x >= 0 && offsetUp.x <= 1 && offsetUp.y >= 0 && offsetUp.y <= 1)
                    maxAlpha = max(maxAlpha, tex2D(_MaskTex, offsetUp).a);
                if (offsetDown.x >= 0 && offsetDown.x <= 1 && offsetDown.y >= 0 && offsetDown.y <= 1)
                    maxAlpha = max(maxAlpha, tex2D(_MaskTex, offsetDown).a);
                if (offsetRight.x >= 0 && offsetRight.x <= 1 && offsetRight.y >= 0 && offsetRight.y <= 1)
                    maxAlpha = max(maxAlpha, tex2D(_MaskTex, offsetRight).a);
                if (offsetLeft.x >= 0 && offsetLeft.x <= 1 && offsetLeft.y >= 0 && offsetLeft.y <= 1)
                    maxAlpha = max(maxAlpha, tex2D(_MaskTex, offsetLeft).a);

                // If current pixel is background, but a nearby pixel is foreground -> it's an outline
                if (maskAlpha < 0.1 && maxAlpha > 0.1)
                {
                    return _OutlineColor;
                }

                if (maskAlpha > 0.1)
                {
                    // Completely transparent inside the silhouette
                    return fixed4(0, 0, 0, 0);
                }
                else
                {
                    // Background color
                    return _MaskBackgroundColor;
                }
            }
            ENDCG
        }
    }
} 