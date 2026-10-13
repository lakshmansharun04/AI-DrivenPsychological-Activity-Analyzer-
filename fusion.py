
import time



psychological_map = {
    "angry": "Stress / Irritability",
    "fear": "Anxiety",
    "sad": "Depression",
    "happy": "Normal",
    "neutral": "Normal"
}



def fuse_emotions(face_label, face_score, eeg_label, eeg_score):
 

    face_weight = 0.8
    eeg_weight = 0.2

    scores = {}
    

    scores[face_label] = face_score * face_weight
    scores[eeg_label] =eeg_score * eeg_weight
    print(scores)

    final_emotion = max(scores, key=scores.get)
    final_score = scores[final_emotion]

    return final_emotion.lower(), final_score


def detect_disorder(final_emotion):
    return psychological_map.get(final_emotion, "Normal")



if __name__ == "__main__":

    userid = 1
    face_label = "Angry"
    face_score = 0.359185

    eeg_label = "happy"
    eeg_score = 0.32263502

    # Fusion
    final_emotion, final_score = fuse_emotions(face_label, face_score, eeg_label, eeg_score)

    # Disorder
    disorder = detect_disorder(final_emotion)

    # Convert to percentage
    confidence_percent = int(final_score * 100)

   
    print("Final Emotion:", final_emotion)
    print("Confidence:", confidence_percent)
    print("Disorder:", disorder)
