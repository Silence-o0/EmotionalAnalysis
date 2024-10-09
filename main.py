import re

base_emotion_dict = {
    "positive": {
        "hope": 0.8,
        "brave": 0.8,
        "relief": 0.7,
        "calm": 0.6,
        "happy": 0.8,
        "secure": 0.7,
        "bright": 0.7,
        "wonderful": 0.8,
        "inspire": 1,
        "stable": 0.75,
        "breakthrough": 0.9,
        "promising": 0.85,
        "remarkable": 0.8,
        "transformative": 0.9,
        "innovative": 0.85,
        "uplift": 0.9,
        "extraordinary": 0.88,
        "encourage": 0.83,
        "success": 0.82,
        "empower": 0.86,
        "groundbreak": 0.9,
        "sustainable": 0.84,
        "optimistic": 0.85,
        "progressive": 0.82,
        "advancement": 0.88,
        "unify": 0.9,
        "compassionate": 0.86,
        "collaborative": 0.83,
        "supportive": 0.85,
        "achievable": 0.84,
        "motivate": 0.87,
        "positive": 0.8,
    },
    "negative": {
        "lose": -0.8,
        "terrify": -0.9,
        "disaster": -0.9,
        "terrible": -0.8,
        "panic": -0.8,
        "sad": -0.6,
        "danger": -0.7,
        "devastate": -0.9,
        "emergency": -0.9,
        "catastrophic": -0.95,
        "disastrous": -0.9,
        "severe": -0.8,
        "crisis": -0.85,
        "chaos": -0.9,
        "chaotic": -0.9,
        "threaten": -0.85,
        "dire": -0.88,
        "hazardous": -0.8,
        "calamitous": -0.92,
        "explosive": -0.87,
        "grim": -0.85,
        "bleak": -0.8,
        "serious": -0.8,
        "warning": -0.8,
        "break": -0.75,
        "worry": -0.8,
        "escalate": -0.8,
        "problematic": -0.76,
        "dangerous": -0.8,
        "volatile": -0.8,
        "outbreak": -0.82,
        "collapse": -0.88,
        "failure": -0.82,
        "crucial": -0.77,
        "frightening": -0.83,
        "alarm": -0.8,
        "critical": -0.85,
        "unstoppable": -0.9,
        "evacuate": -0.8,
        "force": -0.8,
    },
    "intensifiers": {
        "absolutely": 1.3,
        "completely": 1.3,
        "extremely": 1.3,
        "highly": 1.3,
        "totally": 1.3,
        "incredibly": 1.3,
        "unbelievably": 1.3
    },
    "negative_parts": ["not", "no", "never", "none", "nobody", "nothing", "nowhere"]
}


def generate_variations(word, base_weight):
    negative_suffixes = ["less", "lessly"]
    positive_suffixes = ["ful", "fully", "ly", "ing", "ed", "ally", "al"]
    negative_prefixes = ["un", "in", "dis", "im", "ir", "non", "il"]

    variations = {word: base_weight}

    for prefix in negative_prefixes:
        variations[prefix + word] = -base_weight

    for suffix in negative_suffixes + positive_suffixes:
        new_word = word

        if word.endswith("e") and suffix in ("ing", "ed"):
            new_word = word[:-1]

        if (re.match(r'.*[^aeiou]([aeiou])[^aeiou]$', new_word) and not word.endswith("e")
                and suffix in ("ing", "ed")):
            new_word += new_word[-1]

        base_weight = -base_weight if suffix in negative_suffixes else base_weight
        variations[new_word + suffix] = base_weight

    return variations


def create_expanded_emotion_dict(base_emotion_dict):
    expanded_dict = {}

    for category, words_dict in base_emotion_dict.items():
        expanded_dict[category] = {}

        if category in ["intensifiers", "negative_prefixes"]:
            expanded_dict[category] = words_dict
            continue

        if isinstance(words_dict, dict):
            for word, weight in words_dict.items():
                expanded_dict[category][word] = weight
                variations = generate_variations(word, weight)
                for variation, var_weight in variations.items():
                    expanded_dict[category][variation] = var_weight
        elif isinstance(words_dict, list):
            for word in words_dict:
                expanded_dict[category][word] = 0

    return expanded_dict


def get_sentence_type_weight(sentence):
    if sentence.endswith("?"):
        return 0.25
    elif sentence.endswith("!"):
        return 1.5
    return 1.0


def read_text_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None


def analyze_text_emotion(text, emotion_dict, print_details=False):
    sentences = re.split(r'(?<=[.!?])\s+', text.lower())
    sentence_scores = []
    word_details_by_part = [[] for _ in range(4)]

    part_length = max(1, len(sentences) // 4)
    part_multipliers = [1.5, 1.0, 1.25, 1.75]
    part_scores = [0] * 4

    for idx, sentence in enumerate(sentences):
        words = re.findall(r'\b\w+\b', sentence)
        sentence_weight = get_sentence_type_weight(sentence)
        sentence_score = 0
        skip_indices = set()
        part_word_details = []

        for i, word in enumerate(words):
            if i in skip_indices:
                continue

            if word in emotion_dict["positive"] or word in emotion_dict["negative"]:
                weight = emotion_dict["positive"].get(word, emotion_dict["negative"].get(word))
                if i > 0 and words[i - 1] in emotion_dict["negative_parts"]:
                    word = words[i - 1] + " " + word
                    weight = -weight
                sentence_score += weight
                part_word_details.append((word, weight))

            if word in emotion_dict["intensifiers"] and i < len(words) - 1:
                next_word = words[i + 1]
                if next_word in emotion_dict["positive"] or next_word in emotion_dict["negative"]:
                    weight = emotion_dict["positive"].get(next_word, emotion_dict["negative"].get(next_word))
                    intensifier_weight = emotion_dict["intensifiers"][word]
                    adjusted_weight = weight * intensifier_weight
                    sentence_score += adjusted_weight
                    part_word_details.append((f"{word} {next_word}", adjusted_weight))
                    skip_indices.add(i + 1)

        sentence_score *= sentence_weight
        sentence_scores.append(sentence_score)

        part_index = min(idx // part_length, 3)
        part_scores[part_index] += sentence_score
        word_details_by_part[part_index].extend(part_word_details)

    final_score = sum(
        part_score * part_multiplier for part_score, part_multiplier in zip(part_scores, part_multipliers))

    print_results(final_score, part_scores, part_multipliers, word_details_by_part, print_details)


def print_results(final_score, part_scores, part_multipliers, word_details_by_part, print_details=False):
    print(f"Final Emotion Score: {final_score}")
    if final_score < -0.5:
        emotion_result = "Negative"
    elif final_score > 0.5:
        emotion_result = "Positive"
    else:
        emotion_result = "Neutral"
    print(f"Article is : {emotion_result}")
    if print_details:
        print("Sentence scores by parts:")
        for i, part_score in enumerate(part_scores):
            print(
                f"Part {i + 1} score (multiplier {part_multipliers[i]}): "
                f"{part_score} -> {part_score * part_multipliers[i]}")
            print(f"Words in Part {i + 1}:")
            seen_words = set()
            for word, weight in word_details_by_part[i]:
                if word not in seen_words:
                    print(f" - {word}: {weight}")
                    seen_words.add(word)


if __name__ == "__main__":
    expanded_emotion_dict = create_expanded_emotion_dict(base_emotion_dict)

    file_path_name = "article.txt"
    article_text = read_text_from_file(file_path_name)

    if article_text:
        analyze_text_emotion(article_text, expanded_emotion_dict, True)
