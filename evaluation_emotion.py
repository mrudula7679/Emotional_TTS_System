# evaluate_emotions.py
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix  # [web:50]
from emotion_detector import detect_emotion  # import from your project

# 1) Define your emotion label set
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

def load_text_test_data():
    """
    TODO: Replace this with your real test set loader.
    Should return:
        texts: list[str]
        true_labels: list[str]  (same length as texts)
    """
    texts = [
        "I am very happy today!",
        "This is a terrible situation.",
        "That was shocking news.",
    ]
    true_labels = [
        "happy",
        "sad",
        "surprise",
    ]
    return texts, true_labels

def evaluate_text_emotion():
    texts, true_labels = load_text_test_data()
    pred_labels = []

    for t in texts:
        pred = detect_emotion(t)   # your function from emotion_detector.py
        pred_labels.append(pred)

    print("=== TEXT EMOTION EVALUATION ===")
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("Macro F1:", f1_score(true_labels, pred_labels, average="macro"))
    print("Weighted F1:", f1_score(true_labels, pred_labels, average="weighted"))

    print("\nPer-class report:")
    print(classification_report(true_labels, pred_labels, labels=EMOTIONS))

    print("Confusion matrix:")
    print(confusion_matrix(true_labels, pred_labels, labels=EMOTIONS))

if __name__ == "__main__":  # makes this runnable as a separate script [web:52][web:46]
    evaluate_text_emotion()
