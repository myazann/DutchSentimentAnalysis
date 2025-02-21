import random

def get_emotion_labels(language="nl"):

    if language == "nl":
        return {
            0: "geen emotie",
            1: "woede", 
            2: "afkeer",
            3: "angst",
            4: "geluk",
            5: "verdriet",
            6: "verrassing"
        }

    else:
        return {
            0: "no emotion",
            1: "anger", 
            2: "disgust",
            3: "fear",
            4: "happiness",
            5: "sadness",
            6: "surprise"            
        }

def get_emotion_labels_inverse(language="nl"):

    if language == "nl":
        return {
            "geen emotie": 0,
            "woede": 1,
            "afkeer": 2, 
            "angst": 3,
            "geluk": 4,
            "verdriet": 5,
            "verrassing": 6
        }
    else:
        return {
            "no emotion": 0,
            "anger": 1,
            "disgust": 2, 
            "fear": 3,
            "happiness": 4,
            "sadness": 5,
            "surprise": 6           
        }

def get_translate_prompt():

    return [
        {
            "role": "system",
            "content": (
                "You are a translation model specialized in translating English to Dutch. "
                "When given a single sentence, translate it into Dutch and return only the translated sequence. "
                "When given a list of sentences, return a list where each element is the Dutch translation corresponding to each input sentence. "
                "Do not include any additional commentary or explanation."
            )
        },
        {
            "role": "user",
            "content": (
                "Translate the following English text into Dutch. "
                "If the input is a list, return the translations as a list."
                "Otherwise, provide only the translated sequence.\n\n"
                "Input:\n{text}\n\nOutput:"
            )
        }
    ]

def get_classifier_prompt(language="nl"):

    if language == "nl":
        return [
            {
                "role": "system",
                "content": (
                    "Je bent een model voor emotieclassificatie. "
                    "Je taak is om te bepalen welke van de volgende emoties het beste past bij de gegeven tekst: "
                    "[geen emotie, woede, afkeer, angst, geluk, verdriet, verrassing]. "
                    "Gebruik **geen andere klassen** dan deze. "
                    "Als je een enkele tekst ontvangt, geef dan uitsluitend de voorspelde emotie als output, "
                    "Indien er voorbeelden worden meegeleverd, gebruik deze als referentie, maar als er geen voorbeelden zijn, "
                    "maak dan alsnog een voorspelling. "
                    "Geef exact de vereiste output, zonder extra uitleg, whitespace of opmaak.\n\n"
                )
            },
            {
                "role": "user",
                "content": (
                    "<Voorbeelden>"
                    "{few_shot_examples}"
                    "</Voorbeelden>\n"
                    "Output:"
                    "Analyseer de emotie(s) van de volgende tekst(zen):\n\n"
                    "Input:\n{text}\n\n"
                )
            }
        ]

    else:

        return [
        {
            "role": "system",
            "content": (
                "You are a model for emotion classification. "
                "Your task is to determine which of the following emotions best fits the given text: "
                "[no emotion, anger, disgust, fear, happiness, sadness, surprise]. "
                "**Do not use any classes other than these.** "
                "If you receive a single text, provide only the predicted emotion as output, "
                "If examples are provided, use them as reference, but if there are no examples, "
                "still make a prediction. "
                "Provide exactly the required output, without extra explanation, whitespace, or formatting.\n\n"
            )
        },
        {
            "role": "user",
            "content": (
                "<Examples>"
                "{few_shot_examples}"
                "</Examples>\n"
                "Analyze the emotion(s) of the following text(s):\n\n"
                "Input:\n{text}\n\n"
                "Output:"
            )
        }
    ]

def get_few_shot_samples(dataset, args, translator=None):

    emotion_labels = get_emotion_labels(args.language)
    label_indexes = {}
    total_samples = 0

    for i, sample in enumerate(dataset["emotion"]):
        for j, turn in enumerate(sample):
            if label_indexes.get(emotion_labels[turn]) is None:
                label_indexes[emotion_labels[turn]] = []
            label_indexes[emotion_labels[turn]].append((i, j))
            total_samples += 1

    few_shot_examples = ""

    if args.proportional:
        weights = {key: len(indexes) / total_samples 
                  for key, indexes in label_indexes.items()}
        
        min_weight = min(weights.values())
        max_weight = 10
        
        sample_counts = {key: min(max_weight, max(1, (round((args.few_k * (weight / min_weight))/max_weight))))
                        for key, weight in weights.items()}
    else:
        sample_counts = {key: args.few_k for key in label_indexes.keys()}

    for key in label_indexes.keys():
        chosen_indexes = random.choices(label_indexes[key], k=sample_counts[key])
        for sample in chosen_indexes:
            example = dataset["dialog"][sample[0]][sample[1]]
            if args.language == "nl":
                params = {"text": example}
                example = translator.generate(prompt_params=params)
            few_shot_examples = f"{few_shot_examples}\n{example.strip()}: {key}"

    return few_shot_examples