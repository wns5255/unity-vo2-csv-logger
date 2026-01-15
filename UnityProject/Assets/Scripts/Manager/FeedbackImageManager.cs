using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class FeedbackImageManager : MonoBehaviour
{
    public static FeedbackImageManager Instance;

    [SerializeField]
    private Image perfectImage;
    [SerializeField]
    private Image goodImage;
    [SerializeField]
    private Image notgoodImage;
    [SerializeField]
    private Image badImage;

    private const float IMAGE_DURATION = 2.0f;

    private Coroutine _hideCoroutine;

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }

        // 모든 이미지를 시작할 때 비활성화
        if (perfectImage != null) perfectImage.gameObject.SetActive(false);
        if (goodImage != null) goodImage.gameObject.SetActive(false);
        if (notgoodImage != null) notgoodImage.gameObject.SetActive(false);
        if (badImage != null) badImage.gameObject.SetActive(false);
    }

    public void ShowPerfectImage()
    {
        ShowImage(perfectImage);
    }

    public void ShowGoodImage()
    {
        ShowImage(goodImage);
    }
    public void ShowNotGoodImage()
    {
        ShowImage(notgoodImage);
    }
    public void ShowBadImage()
    {
        ShowImage(badImage);
    }

    private void ShowImage(Image imageToShow)
    {
        if (imageToShow == null) 
        {
            Debug.LogError(imageToShow.name + " 이미지가 할당되지 않았습니다. Inspector를 확인해주세요.");
            return;
        }

        // 다른 이미지가 있다면 먼저 숨깁니다.
        if (perfectImage != null && perfectImage != imageToShow) perfectImage.gameObject.SetActive(false);
        if (goodImage != null && goodImage != imageToShow) goodImage.gameObject.SetActive(false);
        if (notgoodImage != null && notgoodImage != imageToShow) notgoodImage.gameObject.SetActive(false);
        if (badImage != null && badImage != imageToShow) badImage.gameObject.SetActive(false);

        imageToShow.gameObject.SetActive(true);

        if (_hideCoroutine != null)
        {
            StopCoroutine(_hideCoroutine);
        }
        _hideCoroutine = StartCoroutine(HideImageCoroutine(imageToShow));
    }

    private IEnumerator HideImageCoroutine(Image imageToHide)
    {
        yield return new WaitForSeconds(IMAGE_DURATION);
        if (imageToHide != null)
        {
            imageToHide.gameObject.SetActive(false);
        }
        _hideCoroutine = null;
    }
}
