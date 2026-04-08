import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Vocabularies
clean_subjects = ["I", "You", "They", "We", "He", "She", "The company", "The team", "My friend", "Everyone"]
clean_verbs = ["love", "like", "enjoy", "went to", "will visit", "recommend", "saw", "heard about", "think about", "learned"]
clean_objects = ["the movie", "this place", "that book", "the new update", "working here", "the weather", "the food", "this new song", "the match"]
clean_adj = ["great", "awesome", "good", "nice", "okay", "informative", "helpful", "interesting", "amazing", "beautiful"]

profanity_words = ["shit", "damn", "fuck", "bitch", "asshole", "bullshit", "crappy", "motherfucker", "fucking", "crap"]
profanity_templates = [
    "This is {profanity}.",
    "I don't give a {profanity}.",
    "What the {profanity} is this?",
    "That was a {profanity}ing disaster.",
    "Holy {profanity}!",
    "I am so {profanity}ing tired.",
    "This {profanity} is broken.",
    "Just my {profanity} luck.",
    "Are you {profanity}ing kidding me?",
    "He's a complete {profanity}."
]

hate_targets = ["blacks", "whites", "asians", "hispanics", "muslims", "jews", "christians", "gays", "women", "men", "immigrants"]
hate_slurs = ["scum", "trash", "animals", "parasites", "terrorists", "invaders", "freaks", "inferiors"]
hate_verbs = ["should be killed", "belong in jail", "ruin everything", "should be deported", "are a disease", "deserve to die", "are inferior"]
hate_templates = [
    "All {target} are {slur}.",
    "I hate {target}, they {verb}.",
    "{target} {verb}.",
    "We need to get rid of {target}.",
    "Such {slur}, typical of {target}."
]

abuse_targets = ["You", "Your mom", "Your family", "He", "She", "This guy", "The author", "The OP"]
abuse_insults = ["idiot", "moron", "loser", "ugly piece of trash", "retard", "dumbass", "coward", "failure", "pathetic joke"]
abuse_templates = [
    "{target} are a complete {insult}.",
    "Go kill yourself, {target}!",
    "Nobody likes {target}.",
    "{target} should jump off a bridge.",
    "Shut up you absolute {insult}.",
    "I hope {target} dies.",
    "You're nothing but a {insult}."
]

spam_toxic_phrases = ["LMAO UR TRASH", "KYS", "NOOB NOOB NOOB", "ez pz get rekt", "cry more kid", "git gud trash", "UR SO BAD HAHAHA", "SPAM SPAM SPAM", "uninstall the game"]
spam_toxic_templates = [
    "{phrase} {phrase}",
    "{phrase}!!!",
    "hahaha {phrase}",
    "{phrase} lmaooo",
    "{phrase} x100"
]

def generate_clean(n):
    data = []
    for _ in range(n):
        if random.random() < 0.5:
            text = f"{random.choice(clean_subjects)} {random.choice(clean_verbs)} {random.choice(clean_objects)}."
        else:
            text = f"{random.choice(clean_objects).title()} is really {random.choice(clean_adj)}."
        data.append({"text": text, "label": "clean"})
    return data

def generate_profanity(n):
    data = []
    for _ in range(n):
        template = random.choice(profanity_templates)
        text = template.replace("{profanity}", random.choice(profanity_words))
        data.append({"text": text, "label": "profanity"})
    return data

def generate_hate(n):
    data = []
    for _ in range(n):
        template = random.choice(hate_templates)
        text = template.replace("{target}", random.choice(hate_targets)).replace("{slur}", random.choice(hate_slurs)).replace("{verb}", random.choice(hate_verbs))
        data.append({"text": text, "label": "hate_speech"})
    return data

def generate_abuse(n):
    data = []
    for _ in range(n):
        template = random.choice(abuse_templates)
        text = template.replace("{target}", random.choice(abuse_targets)).replace("{insult}", random.choice(abuse_insults))
        data.append({"text": text, "label": "abuse"})
    return data

def generate_spam_toxic(n):
    data = []
    for _ in range(n):
        template = random.choice(spam_toxic_templates)
        text = template.replace("{phrase}", random.choice(spam_toxic_phrases))
        data.append({"text": text, "label": "spam_toxic"})
    return data

def main():
    print("Generating dataset...")
    # Generate 10k samples per class (Total 50k)
    samples_per_class = 10000
    
    clean_data = generate_clean(samples_per_class)
    profanity_data = generate_profanity(samples_per_class)
    hate_data = generate_hate(samples_per_class)
    abuse_data = generate_abuse(samples_per_class)
    spam_data = generate_spam_toxic(samples_per_class)
    
    all_data = clean_data + profanity_data + hate_data + abuse_data + spam_data
    
    df = pd.DataFrame(all_data)
    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    output_path = 'data/dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully generated with {len(df)} samples.")
    print("Class distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()
