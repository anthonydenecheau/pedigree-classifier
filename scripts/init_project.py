import os
import json

def create_structure():
    base_dirs = ['data/raw', 'data/processed', 'models', 'scripts']
    for d in base_dirs: os.makedirs(d, exist_ok=True)

    with open('countries.json', 'r') as f:
        config = json.load(f)

    all_classes = []
    for region, clubs in config.items():
        for club in clubs:
            class_name = f"{region}_{club}"
            os.makedirs(os.path.join('data/raw', class_name), exist_ok=True)
            all_classes.append(class_name)
            
    with open('models/classes.json', 'w') as f:
        json.dump(all_classes, f)
    print(f"✅ Structure prête : {len(all_classes)} classes.")

if __name__ == "__main__":
    create_structure()