import random

def get_emotion_labels():

    return {
        0: "geen emotie",
        1: "woede", 
        2: "afkeer",
        3: "angst",
        4: "geluk",
        5: "verdriet",
        6: "verrassing"
    }

def get_emotion_labels_inverse():

    return {
        "geen emotie": 0,
        "woede": 1,
        "afkeer": 2, 
        "angst": 3,
        "geluk": 4,
        "verdriet": 5,
        "verrassing": 6
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

def get_classifier_prompt_nl():

    return [
        {
            "role": "system",
            "content": (
                "Je bent een model voor emotieclassificatie. "
                "Je taak is om te bepalen welke van de volgende emoties het beste past bij de gegeven tekst: "
                "'geen emotie', 'woede', 'afkeer', 'angst', 'geluk', 'verdriet', 'verrassing'. "
                "Als je een enkele tekst ontvangt, geef dan uitsluitend de voorspelde emotie als output, "
                "correct geformatteerd als een Python-string omgeven door dubbele aanhalingstekens, bijvoorbeeld: \"geluk\". "
                "Als je een lijst met teksten ontvangt, geef dan voor iedere tekst de corresponderende voorspelling, "
                "en presenteer de resultaten als een Python-lijst van strings, bijvoorbeeld: [\"geluk\", \"verdriet\", \"geen emotie\"]. "
                "De lijst moet correct geformatteerd zijn voor Python's eval()-functie: "
                "gebruik dubbele aanhalingstekens voor elke string, gebruik komma's tussen de items zonder een komma na het laatste item, "
                "en geef geen extra uitleg, whitespace of karakters buiten de lijst. "
                "Indien er voorbeelden worden meegeleverd, gebruik deze als referentie, maar als er geen voorbeelden zijn, "
                "maak dan alsnog een voorspelling. "
                "Geef exact de vereiste output, zonder extra uitleg of opmaak."
            )
        },
        {
            "role": "user",
            "content": (
                "{few_shot_examples}"
                "Analyseer de emotie(s) van de volgende tekst(zen):\n\n"
                "Input:\n{text}\n\n"
                "*Belangrijk*: Geef **uitsluitend** een emotie uit de volgende lijst als output: "
                "[\"geen emotie\", \"woede\", \"afkeer\", \"angst\", \"geluk\", \"verdriet\", \"verrassing\"]. "
                "Gebruik **geen andere klassen** dan deze. "
                "De output moet correct geformatteerd zijn als een Python-lijst van strings, bijvoorbeeld: [\"geluk\", \"verdriet\"]. "
                "Geef exact de vereiste output, zonder extra uitleg, whitespace of opmaak.\n\n"
                "Output:"
            )
        }
    ]

def get_few_shot_samples(dataset, translator, num_sample_per_class=1):

    emotion_labels = get_emotion_labels()
    label_indexes = {}

    for i, sample in enumerate(dataset["emotion"]):
        for j, turn in enumerate(sample):
            if label_indexes.get(emotion_labels[turn]) is None:
                label_indexes[emotion_labels[turn]] = []
            label_indexes[emotion_labels[turn]].append((i, j))

    few_shot_examples = ""

    for key in label_indexes.keys():
        chosen_indexes = random.choices(label_indexes[key], k=num_sample_per_class)
        for sample in chosen_indexes:
            params = {"text": dataset["dialog"][sample[0]][sample[1]]}
            translated_sequence = translator.generate(prompt_params=params)
            few_shot_examples = f"{few_shot_examples}\n{translated_sequence}: {key}"

    return few_shot_examples    