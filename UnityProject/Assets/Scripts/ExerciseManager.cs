using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ExerciseManager : MonoBehaviour
{
    public ExerciseContent[] exerciseList;
    public GameObject buttonPrefab;
    public Transform buttonContainer;
    public Button previousButton;
    public Button nextButton;
    public TMP_Text pageIndicatorText;
    public TextAsset contentJson;

    private int currentPage = 0;
    private int itemsPerPage = 3;
    private List<ExerciseContent> filteredExerciseList;

    public ChangeScene sceneChange;

    void Start()
    {
        LoadExerciseData();
        FilterExerciseListByType();
        UpdatePage();

        previousButton.onClick.AddListener(OnPreviousPage);
        nextButton.onClick.AddListener(OnNextPage);
    }

    private void LoadExerciseData()
    {
        if (contentJson != null)
        {
            exerciseList = JsonUtility.FromJson<ExerciseContentArrayWrapper>(
                "{\"exercises\":" + contentJson.text + "}").exercises;
        }
        else
        {
            Debug.LogError("JSON file not found");
        }
    }

    private void FilterExerciseListByType()
    {
        string selectedType = PlayerPrefs.GetString("Type", "");

        filteredExerciseList = new List<ExerciseContent>();
        foreach (var exercise in exerciseList)
        {
            if (exercise.type == selectedType)
            {
                filteredExerciseList.Add(exercise);
            }
        }

        if (filteredExerciseList.Count == 0)
        {
            Debug.LogWarning("No exercises found for type: " + selectedType);
        }
    }

    private void UpdatePage()
    {
        foreach (Transform child in buttonContainer)
        {
            Destroy(child.gameObject);
        }

        int startIndex = currentPage * itemsPerPage;
        int endIndex = Mathf.Min(startIndex + itemsPerPage, filteredExerciseList.Count);

        for (int i = startIndex; i < endIndex; i++)
        {
            var exercise = filteredExerciseList[i];

            // 버튼 생성
            GameObject newButton = Instantiate(buttonPrefab, buttonContainer);
            TMP_Text buttonText = newButton.GetComponentInChildren<TMP_Text>();

            // 버튼에는 운동 "이름"을 보여줌
            buttonText.text = exercise.exerciseName;
            buttonText.enableWordWrapping = true;
            buttonText.alignment = TextAlignmentOptions.Center;
            buttonText.margin = new Vector4(10, 70, 5, 5);

            // 이미지 로드 (id 기반)
            Image exerciseImage = newButton.transform.Find("Image").GetComponent<Image>();
            Sprite sprite = Resources.Load<Sprite>("ExerciseCapture/" + exercise.id);
            exerciseImage.sprite = sprite;

            // 버튼 클릭 이벤트
            newButton.GetComponent<Button>().onClick.AddListener(() => OnExerciseButtonClicked(exercise));
        }

        int totalPages = Mathf.CeilToInt((float)filteredExerciseList.Count / itemsPerPage);
        pageIndicatorText.text = (currentPage + 1).ToString() + " / " + totalPages.ToString();

        previousButton.interactable = (currentPage > 0);
        nextButton.interactable = (currentPage < totalPages - 1);
    }

    private void OnPreviousPage()
    {
        if (currentPage > 0)
        {
            currentPage--;
            UpdatePage();
        }
    }

    private void OnNextPage()
    {
        int totalPages = Mathf.CeilToInt((float)exerciseList.Length / itemsPerPage);
        if (currentPage < totalPages - 1)
        {
            currentPage++;
            UpdatePage();
        }
    }

    private void OnExerciseButtonClicked(ExerciseContent exercise)
    {
        // 초기화
        PlayerPrefs.DeleteKey("exerciseId");
        PlayerPrefs.DeleteKey("exerciseName");
        PlayerPrefs.DeleteKey("RecommendedRepeatNum");
        PlayerPrefs.DeleteKey("alternatingSides");
        PlayerPrefs.DeleteKey("preparesec");
        PlayerPrefs.DeleteKey("restsec");

        // 이제는 id와 name을 따로 저장
        PlayerPrefs.SetString("exerciseId", exercise.id);               // 영상/이미지 로딩용
        PlayerPrefs.SetString("exerciseName", exercise.exerciseName);   // UI 표시용 이름

        PlayerPrefs.SetInt("RecommendedRepeatNum", Mathf.Max(1, exercise.repeatNum));
        PlayerPrefs.SetString("alternatingSides", exercise.alternatingSides);
        PlayerPrefs.SetInt("preparesec", exercise.preparesec);
        PlayerPrefs.SetInt("restsec", exercise.restsec);
        PlayerPrefs.SetInt("Num", exercise.Num);

        PlayerPrefs.SetInt("AchivedCount", 0);
        PlayerPrefs.SetInt("PerfectCount", 0);
        PlayerPrefs.SetInt("BadCount", 0);
        PlayerPrefs.SetInt("GoodCount", 0);

        PlayerPrefs.Save();

        sceneChange.toRepeat();
    }

    [System.Serializable]
    public class ExerciseContent
    {
        public string id;
        public string exerciseName;
        public string type;
        public string supplies;
        public int repeatNum;
        public string alternatingSides;
        public int preparesec;
        public int restsec;
        public int Num;
    }

    [System.Serializable]
    public class ExerciseContentArrayWrapper
    {
        public ExerciseContent[] exercises;
    }
}
