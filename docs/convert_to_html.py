
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
    
    h1, h2, h3, h4 {
        color: #2c3e50;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    
    h1 { font-size: 2.5em; border-bottom: 2px solid #eee; padding-bottom: 10px; }
    h2 { font-size: 1.8em; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    h3 { font-size: 1.4em; }
    
    code {
        background-color: #f8f9fa;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em;
    }
    
    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        overflow-x: auto;
        border: 1px solid #eee;
    }
    
    pre code {
        background-color: transparent;
        padding: 0;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    
    th, td {
        border: 1px solid #ddd;
        padding: 12px;
        text-align: left;
    }
    
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    
    tr:nth-child(even) { background-color: #f9f9f9; }
    
    blockquote {
        border-left: 4px solid #3498db;
        margin: 20px 0;
        padding-left: 15px;
        color: #555;
        background-color: #f8fbff;
        padding: 10px 15px;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: #ddd;
        margin: 30px 0;
    }
    
    @media print {
        body { padding: 0; max-width: 100%; }
        h1 { margin-top: 0; }
        a { text-decoration: none; color: #000; }
        pre, blockquote { page-break-inside: avoid; }
        tr, img { page-break-inside: avoid; }
    }
</style>
"""

def convert_md_to_html(md_file, html_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert Markdown to HTML with table support
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'toc'])
    
    # Add title based on filename
    title = os.path.basename(md_file).replace('.md', '').replace('_', ' ').title()
    
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        {CSS}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"[+] Converted {md_file} -> {html_file}")

if __name__ == "__main__":
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert Project Documentation
    convert_md_to_html(
        os.path.join(docs_dir, 'PROJECT_DOCUMENTATION.md'),
        os.path.join(docs_dir, 'Project_Documentation_Printable.html')
    )
    
    # Convert Presentation Slides
    convert_md_to_html(
        os.path.join(docs_dir, 'PRESENTATION_SLIDES.md'),
        os.path.join(docs_dir, 'Presentation_Slides_Printable.html')
    )
    
    print("\\n[SUCCESS] HTML files created! Open them in browser and 'Print to PDF'.")
