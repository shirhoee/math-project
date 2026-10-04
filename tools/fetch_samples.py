import urllib.request
import urllib.parse
import json
import ssl
import os
from PIL import Image
import numpy as np

# Create directories
os.makedirs("assets/samples", exist_ok=True)
ssl._create_default_https_context = ssl._create_unverified_context

def get_image_info(title):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url|extmetadata&titles={urllib.parse.quote(title)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                info = page_data.get("imageinfo", [{}])[0]
                url = info.get("url", "")
                if url:
                    meta = info.get("extmetadata", {})
                    author = meta.get("Artist", {}).get("value", "Unknown")
                    license = meta.get("LicenseShortName", {}).get("value", "Unknown")
                    descurl = info.get("descriptionurl", "")
                    return {"url": url, "source_page": descurl, "author": author, "license": license}
    except Exception as e:
        print(f"Error getting info for {title}: {e}")
    return None

queries = [
    "File:Living_room_(4102748829).jpg",
    "File:Still_life_fleamarket_amk.jpg",
    "File:Wooden_and_bamboo_facades_of_dwellings_with_sudare_in_a_cobbled_street_of_Gion,_perspective_effect_with_vanishing_point,_Kyoto,_Japan.jpg",
    "File:Cheops_Mountain_seen_the_Sir_Donald_Trail.jpg",
    "File:Gray_espresso_cup_with_amaretto_1.jpg",
    "File:J._Lee_Vause_Park_dog_park.jpg"
]

import sys
sys.path.insert(0, ".")
from depth_estimator import DepthEstimator

depth_model = DepthEstimator()

results = []
count = 1
for q in queries:
    if len(results) >= 6:
        break
    res = get_image_info(q)
    if res:
        # download
        name = f"sample_{count}"
        img_path = f"assets/samples/01_{name}.jpg"
        print(f"Downloading {res['url']} to {img_path}...")
        try:
            req = urllib.request.Request(res['url'], headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as response, open(img_path, 'wb') as out:
                out.write(response.read())
            
            # resize
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if img.width > 1600:
                    ratio = 1600.0 / img.width
                    new_size = (1600, int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.save(img_path, "JPEG", quality=92)
                
            # Check depth contrast
            disparity = depth_model.estimate_depth(img_path)
            d_max = disparity.max()
            if d_max > 0:
                disparity = disparity / d_max
            p95 = np.percentile(disparity, 95)
            p5 = np.percentile(disparity, 5)
            contrast = float(p95 - p5)
            print(f"Image {name} depth contrast: {contrast:.3f}")
            
            if contrast < 0.25:
                print(f"Rejecting {name} due to low contrast.")
                os.remove(img_path)
                continue
            
            res["name"] = name
            res["local_path"] = img_path
            res["contrast"] = contrast
            results.append(res)
            count += 1
        except Exception as e:
            print(f"Failed to process {res['url']}: {e}")
            if os.path.exists(img_path):
                os.remove(img_path)

print("Suitability table:")
print("Name\tContrast")
for r in results:
    print(f"{r['name']}\t{r['contrast']:.3f}")

with open("assets/samples/sources.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# Write attribution
with open("assets/samples/ATTRIBUTION.md", "w", encoding="utf-8") as f:
    f.write("# Photo credits\n\n")
    for r in results:
        f.write(f"- **{r['name']}**: [{r['source_page']}]({r['source_page']}) by {r['author']}, License: {r['license']}\n")

print(f"Successfully downloaded {len(results)} images.")
