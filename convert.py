
import markdown
import os

# CSS for nice PDF printing
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 850px;
        margin: 0 auto;
        padding: 40px;
        background-color: #ffffff;
    }
    
    h1, h2, h3, h4 { color: #2c3e50; margin-top: 1.5em; margin-bottom: 0.5em; }
    h1 { font-size: 2.5em; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    h2 { font-size: 1.8em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    
    code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 0.9em; }
    pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #eee; }
    
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background-color: #f2f2f2; font-weight: bold; }
    tr:nth-child(even) { background-color: #f9f9f9; }
    
    blockquote { border-left: 4px solid #3498db; margin: 20px 0; padding-left: 15px; color: #555; background-color: #f8fbff; padding: 10px 15px; }
    
    @media print {
        body { padding: 0; max-width: 100%; }
        a { text-decoration: none; color: #000; }
        pre, blockquote, tr, img { page-break-inside: avoid; }
    }
</style>
"""

def convert(filename):
    try:
        input_path = os.path.join('docs', filename)
        output_path = os.path.join('docs', filename.replace('.md', '.html'))
        
        print(f"Reading {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
        
        full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{filename}</title>{CSS}</head><body>{html}</body></html>"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"[+] Created {output_path}")
        
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    convert('PROJECT_DOCUMENTATION.md')
    convert('PRESENTATION_SLIDES.md')
