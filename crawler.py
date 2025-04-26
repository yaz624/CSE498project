import requests
import random
import time

SERVER_URL = "http://127.0.0.1:5000/api/submit-feedback"  

comments = [
    "Amazing character!",
    "So creative!",
    "Love the pixel style!",
    "Perfect for my game project!",
    "Very cool design!",
    "Awesome work!",
    "I like this generator!",
    "Great color themes!",
    "Fast and easy!",
    "Impressive result!",
    "The details are incredible!",
    "Exactly what I was looking for!",
    "This will be perfect for my RPG!",
    "The AI understands my vision!",
    "Better than I expected!",
    "The textures are so realistic!",
    "Love the unique style!",
    "Would use this for my comic book!",
    "The proportions are perfect!",
    "This generator saves me so much time!",
    "The color palette is spot on!",
    "Can't believe this was AI-generated!",
    "The character has so much personality!",
    "Perfect for my indie game prototype!",
    "The shading is professionally done!",
    "I'll be using this tool daily!",
    "The armor design is fantastic!",
    "Looks like it came from a AAA game!",
    "The facial expression is perfect!",
    "Better than most paid generators!",
    "The weapon design is so creative!",
    "This exceeded all my expectations!",
    "The pose is dynamic and interesting!",
    "Would recommend to other developers!",
    "The silhouette is instantly recognizable!",
    "The fantasy elements are well-balanced!",
    "Could pass for hand-drawn artwork!",
    "The style consistency is impressive!",
    "Perfect for my tabletop game tokens!",
    "The details in the clothing are amazing!",
    "This will be my go-to character creator!",
    "The lighting effects are professional grade!",
    "Looks like it jumped out of an anime!",
    "The design inspires new story ideas!",
    "Can't wait to show my team these results!"
]


for i in range(100):
    payload = {
        "seed": str(random.randint(0, 100000)),
        "like": True,
        "comment": random.choice(comments)
    }

    try:
        response = requests.post(SERVER_URL, json=payload)
        if response.status_code == 200:
            print(f"[{i+1}] Feedback submitted successfully!")
        else:
            print(f"[{i+1}] Failed: {response.text}")
    except Exception as e:
        print(f"[{i+1}] Exception occurred:", e)

    time.sleep(0.1)
