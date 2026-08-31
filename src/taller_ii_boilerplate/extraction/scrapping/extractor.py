import requests
from bs4 import BeautifulSoup

URL_A_EXTRAER = "https://web-scraping.dev/api/testimonials"

def descargar_html(url, pagina):
    try:
        respuesta = requests.get(f"{url}?page={pagina}", headers={"referer": "https://web-scraping.dev/testimonials"})
        if respuesta.status_code == 200:
            return respuesta.text
        else:
            return None
    except requests.exceptions.ConnectionError:
        return None

def parsear_html(html_crudo):
    parser = BeautifulSoup(html_crudo, 'html.parser')
    return parser

def ajustar_html(html_crudo):
    if "html" in html_crudo: 
        return html_crudo #ya está limpio
    return f"<html><body>{html_crudo}</body></html>"


html_crudo = descargar_html(URL_A_EXTRAER, 6)
html_crudo_limpio = ajustar_html(html_crudo)
parser = parsear_html(html_crudo_limpio)

contenidos = parser.find_all("div", {"class": "testimonial"})
for contenido in contenidos:
    print(contenido.text.strip())
