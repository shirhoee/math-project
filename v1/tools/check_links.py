import os
import re

def check_links():
    md_files = [f for f in os.listdir('.') if f.endswith('.md')]
    all_good = True
    
    # Matches markdown links [text](path) or ![text](path)
    pattern = re.compile(r'\[.*?\]\((.*?)\)')
    
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        links = pattern.findall(content)
        for link in links:
            if link.startswith('http') or link.startswith('#'):
                continue
            
            # resolve relative to md_file
            target = os.path.join(os.path.dirname(md_file), link)
            if not os.path.exists(target):
                print(f"Broken link in {md_file}: {link}")
                all_good = False
                
    if not all_good:
        import sys
        sys.exit(1)
    else:
        print("All links OK.")
        
if __name__ == '__main__':
    check_links()
