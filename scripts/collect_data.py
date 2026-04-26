import os, json, time, requests
from playwright.sync_api import sync_playwright

with open('models/classes.json', 'r') as f:
    CLASSES = json.load(f)

SEARCH_QUERIES = {
    "FRANCE_SCC": ['site:.fr "pedigree" "LOF" filetype:jpg', 'site:chiens-de-france.com "pedigree"'],
    "USA_AKC": ['site:.com "AKC" "Certified Pedigree"', 'site:.edu "pedigree" "dog"'],
    "UK_KC": ['site:.co.uk "Kennel Club" "Pedigree"'],
    "DEFAULT": ['"pedigree dog certificate" scan']
}

def download_images(query, save_path, limit=30):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(f"https://www.bing.com/images/search?q={query}")
        for _ in range(2): 
            page.mouse.wheel(0, 2000)
            time.sleep(1)
        urls = page.locator('img.mimg').evaluate_all("imgs => imgs.map(img => img.src)")
        
        count = 0
        for url in urls:
            if count >= limit or not url.startswith('http'): continue
            try:
                img_data = requests.get(url, timeout=5).content
                if len(img_data) > 15000:
                    with open(f"{save_path}/img_{count}.jpg", 'wb') as f:
                        f.write(img_data)
                    count += 1
            except: continue
        browser.close()
        print(f" -> {count} images pour {query}")

if __name__ == "__main__":
    for c in CLASSES:
        path = os.path.join('data/raw', c)
        q = SEARCH_QUERIES.get(c, SEARCH_QUERIES["DEFAULT"])[0]
        download_images(q, path)