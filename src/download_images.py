import os
import requests
from duckduckgo_search import DDGS
from time import sleep
from pathlib import Path

def search_images(term, max_images=30):
    print(f"Searching for '{term}'...")
    try:
        with DDGS() as ddgs:
            results = ddgs.images(term, max_results=max_images)
            return [r['image'] for r in results]
    except Exception as e:
        print(f"Search failed for {term}: {e}")
        return []

def download_image(url, dest_path):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_images_for_class(class_name, search_term, base_path, max_images=30):
    dest = Path(base_path) / class_name
    dest.mkdir(parents=True, exist_ok=True)
    
    urls = search_images(search_term, max_images=max_images)
    
    print(f"Downloading images for {class_name}...")
    success_count = 0
    for i, url in enumerate(urls):
        if success_count >= max_images:
            break
            
        ext = url.split('.')[-1].split('?')[0].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'webp']:
            ext = 'jpg'
        
        filename = f"{class_name}_{success_count}.{ext}"
        if download_image(url, dest / filename):
            success_count += 1
            sleep(0.1)
    
    print(f"Successfully downloaded {success_count} images for {class_name}.")

if __name__ == "__main__":
    # Ensure folder structure is relative to this script or root
    # assuming we run from the root of 'tennis-court-surface-classifier'
    train_data_path = Path("data/train")
    val_data_path = Path("data/val")
    classes = {
        "clay": "clay tennis court",
        "grass": "grass tennis court",
        "hard_court": "hard tennis court"
    }
    
    for class_name, search_term in classes.items():
        print(f"--- Downloading for {class_name} ---")
        download_images_for_class(class_name, search_term, train_data_path, max_images=40)
        # Search again for val with a slight variation to avoid exact duplicates
        download_images_for_class(class_name, search_term + " player", val_data_path, max_images=40)
    
    print("\nDataset download process complete!")
